# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random
from pathlib import Path
import yaml
from typing import Tuple
import math
import io

from fastnumpyio import load as load_np
import cv2
import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

KITTI_DEPTH_MIN = 1e-3
KITTI_DEPTH_MAX = 150
NYU_DEPTH_MIN = 1e-3
NYU_DEPTH_MAX = 10

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

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


def augment_long_range_tensors(
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
    batch_size = image.size(0)
    new_image, new_depth_map, new_intrinsics = [], [], []
    for bi in range(batch_size):
        image_bi = image[bi, ...].cpu().permute(1, 2, 0).numpy()
        depth_map_bi = depth_map[bi, ...].cpu().permute(1, 2, 0).numpy()
        intrinsics_bi = intrinsics[bi, ...].cpu().numpy()
        
        image_bi, depth_map_bi, intrinsics_bi = augment_long_range(image_bi, depth_map_bi, intrinsics_bi, alpha)
        new_image.append(torch.from_numpy(image_bi).permute(2, 0, 1))
        new_depth_map.append(torch.from_numpy(depth_map_bi).unsqueeze(2).permute(2, 0, 1))
        new_intrinsics.append(torch.from_numpy(intrinsics_bi))
    img_mapped = torch.stack(new_image)
    depth_map_mapped = torch.stack(new_depth_map)
    intrinsics_mapped = torch.stack(new_intrinsics)
    return img_mapped, depth_map_mapped, intrinsics_mapped


# class DepthDataLoader(object):
#     def __init__(self, args, mode):
#         if mode == 'train':
#             self.training_samples = DataLoadPreprocess(args, 'train', transform=ToTensor('train'))
#             if args.distributed:
#                 self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
#             else:
#                 self.train_sampler = None

#             self.data = DataLoader(self.training_samples, args.batch_size,
#                                    shuffle=(self.train_sampler is None),
#                                    num_workers=1,
#                                    pin_memory=False,
#                                    sampler=self.train_sampler,
#                                    persistent_workers=True)

#         elif mode == 'eval':
#             self.testing_samples = DataLoadPreprocess(args, 'eval', transform=transform=ToTensor('eval'))
#             if args.distributed:  # redundant. here only for readability and to be more explicit
#                 # Give whole test set to all processes (and perform/report evaluation only on one) regardless
#                 self.eval_sampler = None
#             else:
#                 self.eval_sampler = None
#             self.data = DataLoader(self.testing_samples, 1,
#                                    shuffle=False,
#                                    num_workers=1,
#                                    pin_memory=False,
#                                    sampler=self.eval_sampler,
#                                    persistent_workers=True)

#         elif mode == 'test':
#             self.testing_samples = DataLoadPreprocess(args, 'test', transform=transform=ToTensor('test'))
#             self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=0, pin_memory=False)

#         else:
#             print('mode should be one of \'train, test, eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class CustomDataset(Dataset):
    def __init__(self, dataset_name, filenames_file, mode, transform=None, eval=False):
        assert mode in ['eval', 'train']
        with open(filenames_file, 'r') as f:
            raw_gt_tuples = f.readlines()
            raw_paths, gt_paths = zip(*[fn.split(',') for fn in raw_gt_tuples])
            self.raw_paths = [p.replace('\n', '') for p in raw_paths]
            self.gt_paths = [p.replace('\n', '') for p in gt_paths]
            
        self.mode = mode
        self.dataset_name = dataset_name
        self.transform = transform
        self.to_tensor = ToTensor
        self.eval = eval
        self.image_height, self.image_width = None, None
        
        assert dataset_name in ['nyu', 'kitti']
        if dataset_name == 'nyu':
            # NOTE(james) - the matlab script I borrowed dumps 16bit depth
            # https://github.com/wangq95/NYUd2-Toolkit
            # NOTE(carter) - NYU preprocessing assigns max depth to invalid values
            self.depth_normalizer = 65535.0 / 10 # 1000
            self.depth_min = NYU_DEPTH_MIN
            self.depth_max = NYU_DEPTH_MAX
            self.intrinsics = np.array([[5.1885790117450188e+02, 0, 3.2558244941119034e+02],
                                        [0, 5.1946961112127485e+02, 2.5373616633400465e+02],
                                        [0, 0, 1]])

        elif dataset_name == 'kitti':
            self.depth_normalizer = 256.0
            self.depth_min = KITTI_DEPTH_MIN
            self.depth_max = KITTI_DEPTH_MAX
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        raw_path = self.raw_paths[idx]
        gt_path = self.gt_paths[idx]
        
        if self.dataset_name == 'nyu':
            intrinsics = self.intrinsics.copy()
            # TODO: sam_feats_path = ...
        else:
            # intrinsics_path = Path(raw_path).parent.parent.parent.parent / 'calib_cam_to_cam.txt'
            # with open(intrinsics_path, "r") as f:
            #     intrinsics_str = yaml.safe_load(f)
            # intrinsics_str = intrinsics_str['K_02'] if 'image_02' in raw_path else intrinsics_str['K_03']
            # intrinsics = np.array([float(x) for x in intrinsics_str.split(' ')]).reshape((3, 3))
            intrinsics = np.eye(3)
            # sam_feats_path = raw_path.replace("kitti-depth", "kitti-depth-sam-feats-np").replace(".png", ".npy")
            sam_feats_path = "/mnt/remote/shared_data/datasets/kitti-depth-sam-feats-np/2011_10_03/2011_10_03_drive_0047_sync/image_03/data/0000000831_fp16.npy"
        focal = 0.0

        image = transforms.functional.pil_to_tensor(Image.open(raw_path)).float() / 255.0
        depth_gt = transforms.functional.pil_to_tensor(Image.open(gt_path)).float() / self.depth_normalizer
        with open(sam_feats_path, "rb") as f:
            buf = io.BytesIO(f.read())
            sam_feats = torch.from_numpy(load_np(buf))[0, ...]
        sam_feats = torch.zeros_like(image)

        # Crop
        if self.dataset_name == 'kitti':
            self.image_height = 256
            self.image_width = 1216
            height, width = image.shape[1:]
            top_margin = int(height - self.image_height)
            left_margin = int((width - self.image_width) / 2)
            
            image = image[:, top_margin:top_margin+self.image_height, left_margin:left_margin+self.image_width]
            depth_gt = depth_gt[:, top_margin:top_margin+self.image_height, left_margin:left_margin+self.image_width]
            sam_feats = sam_feats[:, top_margin:top_margin+self.image_height, left_margin:left_margin+self.image_width]
        else:
            depth_gt = depth_gt.crop((43, 45, 608, 472))
            image = image.crop((43, 45, 608, 472))
            self.image_height = 416
            self.image_width = 544
                
        sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'image_path': raw_path, 'depth_path': gt_path, 'intrinsics': intrinsics, 'sam_feats': sam_feats}

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

        # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # if rank == 1:
        #     new_h, new_w = image.shape[0]*3//4, image.shape[1]*3//4
        #     image = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
        #     depth = cv2.resize(depth_gt, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
        #     intrinsics *= 0.5
        #     intrinsics[2, 2] = 1
        # do_resize = random.random()
        # if rank == 1 and do_resize > 0.5:
        #     print(f"Augmented")
        #     image, depth_gt, intrinsics = augment_long_range(image, depth_gt, intrinsics, alpha=1.5)
            
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
        if self.dataset_name == 'nyu':
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
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image[:3, ...] = self.normalize(image[:3, ...])

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        intrinsics = torch.tensor(sample['intrinsics'])

        sam_feats = sample['sam_feats']
        return {'image': image, 'depth': depth, 'focal': focal, 'image_path': sample['image_path'], 'depth_path': sample['depth_path'], 'intrinsics': intrinsics, 'sam_feats': sam_feats}

    def to_tensor(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        
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
