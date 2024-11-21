# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random
from pathlib import Path
import yaml
from typing import Tuple
import math

import cv2
import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

KITTI_DEPTH_MIN = 1e-3
KITTI_DEPTH_MAX = 256
NYU_DEPTH_MIN = 1e-3
NYU_DEPTH_MAX = 10

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

def pad_images(images: torch.Tensor, multiple_of: int = 32) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pads an image tensor with zeros such that its dimensions are a multiple of the given int.
    Padding is always applied at the bottom and on the right.

    Args:
        images: tensor of shape [C, H, W] or [B, C, H, W]
        multiple_of: Image dimensions after padding will be integer multiples of this argument. Defaults to 32.
    """
    height, width = images.shape[-2:]
    images_dim = images.dim()
    padding_bottom: int = int(math.ceil(float(height) / multiple_of) * multiple_of) - int(height)
    padding_right: int = int(math.ceil(float(width) / multiple_of) * multiple_of) - int(width)
    assert padding_bottom >= 0 and padding_right >= 0, f"Invalid padding: {padding_bottom}, {padding_right}"
    if images_dim == 4:
        images = F.pad(images, (0, padding_right, 0, padding_bottom), mode="constant", value=0.0)
    else:
        images = F.pad(images.unsqueeze(0), (0, padding_right, 0, padding_bottom), mode="constant", value=0.0).squeeze(
            0
        )
    return images, (padding_bottom, padding_right)


def augment_long_range(
    image,
    depth_map,
    intrinsics,
    alpha: float,
):
    """Apply long range augmentation to all cameras.
    - Performs a local optimization of alpha in range [alpha-0.1, alpha+0.1] to minimize padding for augmented image
    - Augment intrinsics and extrinsics so that depth of projected lidar points is scaled by alpha in camera frame

    Args:
        image: dict of images from each camera for the last lidar bundle
        alpha: scaling ratio for the image. Z-coordinate is scaled by (1/alpha)

    Returns:
        image: augmented dict of images from each camera for the last lidar bundle
        camera_intrinsics: augmented intrinsics
        camera_extrinsics: augmented extrinsics
        alpha: locally optimized alpha
    """
    # img_colorized = (image[..., :3] * 255).astype(np.uint8)
    # depth_map_colorized = cv2.applyColorMap(depth_map.astype(np.uint8), cv2.COLORMAP_TURBO) # img.shape = (x, y)
    # depth_overlay = cv2.addWeighted(img_colorized, 0.5, depth_map_colorized, 0.5, 0.0)
    # cv2.imwrite("depth_orig.png", depth_overlay)

    # Get pixelwise image remappings
    def get_maps(_alpha, _intrinsics, _img_size):
        map_x1, map_y1 = np.meshgrid(np.arange(_img_size[1]), np.arange(_img_size[0]))
        map_x1 = map_x1.astype(np.float32)
        map_y1 = map_y1.astype(np.float32)
        map_x1 *= _alpha
        map_y1 *= _alpha
        map_x1 += (1 - _alpha) * _intrinsics[0, 2].item()
        map_y1 += (1 - _alpha) * _intrinsics[1, 2].item()

        # Different logic needed to handle upsampling vs downsampling
        if _alpha > 1:
            map_x2, map_y2 = np.meshgrid(np.arange(_img_size[1] // _alpha), np.arange(_img_size[0] // _alpha))
            map_x2 = map_x2.astype(np.float32)
            map_y2 = map_y2.astype(np.float32)
            map_x2 -= (1 - _alpha) * _intrinsics[0, 2].item() / _alpha
            map_y2 -= (1 - _alpha) * _intrinsics[1, 2].item() / _alpha

            intrinsics_correction = torch.eye(3, dtype=_intrinsics.dtype)
            intrinsics_correction[0, 2] = (1 - _alpha) * _intrinsics[0, 2].item() / _alpha
            intrinsics_correction[1, 2] = (1 - _alpha) * _intrinsics[1, 2].item() / _alpha
            new_intrinsics = intrinsics_correction @ _intrinsics
        else:
            new_intrinsics = _intrinsics
            map_x2, map_y2 = None, None

        return map_x1, map_y1, map_x2, map_y2, new_intrinsics

    img_size = image.shape[:2]
    map_x1, map_y1, map_x2, map_y2, new_intrinsics = get_maps(alpha, torch.tensor(intrinsics), img_size)

    intrinsics = new_intrinsics.numpy()
    depth_map_mapped = depth_map * alpha
    img_mapped = cv2.remap(image, map_x1, map_y1, cv2.INTER_LINEAR)
    depth_map_mapped = cv2.remap(depth_map_mapped, map_x1, map_y1, cv2.INTER_NEAREST)
    if not isinstance(map_x2, type(None)):
        img_mapped = cv2.remap(img_mapped, map_x2, map_y2, cv2.INTER_LINEAR)
        depth_map_mapped = cv2.remap(depth_map_mapped, map_x2, map_y2, cv2.INTER_NEAREST)

    # img_colorized = (img_mapped[..., :3] * 255).astype(np.uint8)
    # depth_map_colorized = cv2.applyColorMap(depth_map_mapped.astype(np.uint8), cv2.COLORMAP_TURBO) # img.shape = (x, y)
    # depth_overlay = cv2.addWeighted(img_colorized, 0.5, depth_map_colorized, 0.5, 0.0)
    # cv2.imwrite("depth.png", depth_overlay)
    return img_mapped, depth_map_mapped, intrinsics


class DepthDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, eval=False):
        self.args = args
        assert mode in ['eval', 'train']
        if mode == 'eval':
            filenames_file = args.filenames_file_eval
        else:
            filenames_file = args.filenames_file
        with open(filenames_file, 'r') as f:
            raw_gt_tuples = f.readlines()
            raw_paths, gt_paths = zip(*[fn.split(',') for fn in raw_gt_tuples])
            self.raw_paths = [p.replace('\n', '') for p in raw_paths]
            self.gt_paths = [p.replace('\n', '') for p in gt_paths]
            
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.eval = eval
        self.image_height, self.image_width = None, None
        
        assert self.args.dataset in ['nyu', 'kitti']
        if self.args.dataset == 'nyu':
            # NOTE(james) - the matlab script I borrowed dumps 16bit depth
            # https://github.com/wangq95/NYUd2-Toolkit
            # NOTE(carter) - NYU preprocessing assigns max depth to invalid values
            self.depth_normalizer = 65535.0 / 10 # 1000
            self.depth_min = NYU_DEPTH_MIN
            self.depth_max = NYU_DEPTH_MAX
            self.intrinsics = np.array([[5.1885790117450188e+02, 0, 3.2558244941119034e+02],
                                        [0, 5.1946961112127485e+02, 2.5373616633400465e+02],
                                        [0, 0, 1]])

        elif self.args.dataset == 'kitti':
            self.depth_normalizer = 256.0
            self.depth_min = KITTI_DEPTH_MIN
            self.depth_max = KITTI_DEPTH_MAX
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        raw_path = self.raw_paths[idx]
        gt_path = self.gt_paths[idx]
        
        if self.args.dataset == 'nyu':
            intrinsics = self.intrinsics.copy()
            # TODO: sam_feats_path = ...
        else:
            intrinsics_path = Path(raw_path).parent.parent.parent.parent / 'calib_cam_to_cam.txt'
            with open(intrinsics_path, "r") as f:
                intrinsics_str = yaml.safe_load(f)
            intrinsics_str = intrinsics_str['K_02'] if 'image_02' in raw_path else intrinsics_str['K_03']
            intrinsics = np.array([float(x) for x in intrinsics_str.split(' ')]).reshape((3, 3))
            sam_feats_path = raw_path.replace("kitti-depth", "kitti-depth-sam-feats")
        focal = 0.5 * (intrinsics[0, 0] + intrinsics[1, 1])

        image = Image.open(raw_path)
        depth_gt = Image.open(gt_path)
        sam_feats = torch.load(sam_feats_path)

        if self.args.dataset == 'kitti':
            self.image_height = 250
            self.image_width = 1200
            height, width = image.height, image.width
            top_margin = int(height - self.image_height)
            left_margin = int((width - self.image_width) / 2)
            image = image.crop((left_margin, top_margin, left_margin + self.image_width, top_margin + self.image_height))
            depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + self.image_width, top_margin + self.image_height))
            sam_feats = sam_feats[..., top_margin:top_margin+self.image_height, left_margin:left_margin+self.image_width]
        else:
            depth_gt = depth_gt.crop((43, 45, 608, 472))
            image = image.crop((43, 45, 608, 472))
            self.image_height = 416
            self.image_width = 544
        sam_feats = sam_feats[0, ...].permute(1, 2, 0).numpy()
                
        if self.mode == 'train':
            # if self.args.do_random_rotate is True:
            #     random_angle = (random.random() - 0.5) * 2 * self.args.degree
            #     image = self.rotate_image(image, random_angle)
            #     depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            # image, depth_gt = self.random_crop(image, depth_gt, self.image_height, self.args.image_width)
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / self.depth_normalizer
            image = np.concatenate([image, sam_feats], axis=2)
            image, depth_gt, intrinsics = self.train_preprocess(image, depth_gt, intrinsics)
            depth_gt_mask = np.logical_and(depth_gt > self.depth_min, depth_gt < self.depth_max)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'depth_mask': depth_gt_mask, 'intrinsics': intrinsics}
        else:
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / self.depth_normalizer
            depth_gt_mask = np.logical_and(depth_gt > self.depth_min, depth_gt < self.depth_max)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'image_path': raw_path, 'depth_path': gt_path, 'depth_mask': depth_gt_mask, 'intrinsics': intrinsics}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt, intrinsics):
        """
        Applies flipping and gamma/brightness/color augmentation. Flipping currently disabled for consistent intrinsics

        Returns:
            image_aug, depth_gt_aug
        """
        # # Random flipping
        # do_flip = random.random()
        # if do_flip > 0.5:
        #     image = (image[:, ::-1, :]).copy()
        #     depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image[..., :3] = self.augment_image(image[..., :3])

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # if rank == 1:
        #     new_h, new_w = image.shape[0]*3//4, image.shape[1]*3//4
        #     image = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
        #     depth = cv2.resize(depth_gt, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
        #     intrinsics *= 0.5
        #     intrinsics[2, 2] = 1
        do_resize = random.random()
        if True:
        # if rank == 1 and do_resize() > 0.5:
            image, depth_gt, intrinsics = augment_long_range(image, depth_gt, intrinsics, alpha=1.5)
            
        # do_resize = random.random()
        # if do_resize > 0.5:
        #     if do_resize > 0.75:
        #         new_h, new_w = image.shape[0]*2//3, image.shape[1]*2//3
        #     else:
        #         new_h, new_w = image.shape[0]*3//4, image.shape[1]*3//4
        #     image = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
        #     depth = cv2.resize(depth_gt, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
        #     intrinsics *= 0.5
        #     intrinsics[2, 2] = 1
            
        return image, depth_gt, intrinsics

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.raw_paths)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image[:3, ...] = self.normalize(image[:3, ...])
        image = pad_images(image, multiple_of=32)[0]

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth, depth_mask = sample['depth'], sample['depth_mask']
        intrinsics = torch.tensor(sample['intrinsics'])
        depth = self.to_tensor(depth)
        depth_mask = self.to_tensor(depth_mask)
        depth = pad_images(depth, multiple_of=32)[0]
        depth_mask = pad_images(depth_mask, multiple_of=32)[0]
        if self.mode == 'train':
            return {'image': image, 'depth': depth, 'focal': focal, 'depth_mask': depth_mask, 'intrinsics': intrinsics}
        else:
            return {'image': image, 'depth': depth, 'focal': focal, 'image_path': sample['image_path'], 'depth_path': sample['depth_path'], 'depth_mask': depth_mask, 'intrinsics': intrinsics}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            if len(pic.shape) == 2:
                pic = pic[..., None]
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
