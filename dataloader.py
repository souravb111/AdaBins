# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications
import hdf5plugin
import h5py
import os
import io
import PIL
import random
import gc
from pathlib import Path
import yaml
from typing import Tuple
import math
import io
import h5py
import hdf5plugin

import cv2
import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from fast_np import load as load_np

KITTI_DEPTH_MIN = 1e-3
KITTI_DEPTH_MAX = 150
NYU_DEPTH_MIN = 1e-3
NYU_DEPTH_MAX = 10
USE_SAM = os.environ.get("USE_SAM", False)

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
    """Apply long range augmentation. Augment intrinsics and extrinsics so that depth of projected lidar points is scaled by alpha in camera frame"""
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

    return img_mapped, depth_map_mapped, intrinsics


def augment_long_range_tensors(
    image,
    depth_map,
    intrinsics,
    alpha: float,
):
    """Same as augment_long_range but works on tensor inputs. Intended to be used on the inputs after collate_fn"""
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


class DepthDataLoader(object):
    def __init__(self, args, mode):
        dataset_cls = BothDatasets if args.both_data else DataLoadPreprocess
        collate = collate_both if args.both_data else None
        if mode == 'train':
            self.training_samples = dataset_cls(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler,
                                   collate_fn=collate,
                                   drop_last=True,
                                   )

        elif mode == 'eval':
            self.testing_samples = dataset_cls(args, mode, transform=preprocessing_transforms(mode))
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=collate,
                                   drop_last=True,
                                   )

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=0, pin_memory=False)

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
        
        self.dataset_type = args.dataset
        assert self.dataset_type in ['nyu', 'kitti']
        if self.dataset_type == 'nyu':
            # NOTE(james) - the matlab script I borrowed dumps 16bit depth
            # https://github.com/wangq95/NYUd2-Toolkit
            # NOTE(carter) - NYU preprocessing assigns max depth to invalid values
            self.depth_normalizer = 65535.0 / 10 # 1000
            self.depth_min = NYU_DEPTH_MIN
            self.depth_max = NYU_DEPTH_MAX
            self.intrinsics = np.array([[5.1885790117450188e+02, 0, 3.2558244941119034e+02],
                                        [0, 5.1946961112127485e+02, 2.5373616633400465e+02],
                                        [0, 0, 1]])

        elif self.dataset_type == 'kitti':
            self.depth_normalizer = 256.0
            self.depth_min = KITTI_DEPTH_MIN
            self.depth_max = KITTI_DEPTH_MAX
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        try:
            return self._getitem__(idx)
        except PIL.UnidentifiedImageError:
            print(f"Encountered Pillow loading error on {idx} with raw path {self.raw_paths[idx]} and gt path {self.gt_paths[idx]}")
            rand_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(rand_idx)
    
    def load_nyu_sam_feats(raw_path: str) -> torch.Tensor:
        sam_f_path = raw_path.replace("nyu_depth_v2_sync", "nyu_sam_feats_downsample").replace(".png", ".npy")
        sam_feats = np.load(sam_f_path)
        sam_feats = torch.nn.functional.interpolate(torch.from_numpy(sam_feats), scale_factor=4).permute(0,2,3,1).numpy()
        return sam_feats[0]
    
    def load_kitti_sam_feats(raw_path: str) -> torch.Tensor:
        sam_feats_path = raw_path.replace("kitti-depth", "kitti-depth-sam-feats-np").replace(".png", ".h5")
        if not os.path.exists(sam_feats_path):
            sam_feats_path = raw_path.replace("kitti-depth", "kitti-depth-sam-feats-np").replace(".png", ".npy")
        
        if sam_feats_path.endswith(".npy"):
            with open(sam_feats_path, "rb") as f:
                buf = io.BytesIO(f.read())
                sam_feats = torch.from_numpy(load_np(buf))
        else:
            h5f = h5py.File(sam_feats_path,'r')
            sam_feats = h5f['data'][:]
            h5f.close()
        
        return np.transpose(sam_feats[0], (1, 2 , 0)).astype(np.float32)

    def _getitem__(self, idx):
        raw_path = self.raw_paths[idx]
        gt_path = self.gt_paths[idx]
        
        if self.args.dataset == 'nyu':
            intrinsics = self.intrinsics.copy()
            sam_feats_path = raw_path.replace("nyu_depth_v2_sync", "nyu_sam_feats_downsample").replace(".png", ".npy")
            # sam_feats = np.load(sam_f_path)
            # # ...
            # sam_feats = torch.nn.functional.interpolate(torch.from_numpy(sam_feats), scale_factor=4).permute(0,2,3,1).numpy()
            # return sam_feats[0]
        else:
            # intrinsics_path = Path(raw_path).parent.parent.parent.parent / 'calib_cam_to_cam.txt'
            # with open(intrinsics_path, "r") as f:
            #     intrinsics_str = yaml.safe_load(f)
            # intrinsics_str = intrinsics_str['K_02'] if 'image_02' in raw_path else intrinsics_str['K_03']
            # intrinsics = np.array([float(x) for x in intrinsics_str.split(' ')]).reshape((3, 3))
            sam_feats_path = raw_path.replace("kitti-depth", "kitti-depth-sam-feats-np").replace(".png", ".h5")
            if not os.path.exists(sam_feats_path):
                sam_feats_path = raw_path.replace("kitti-depth", "kitti-depth-sam-feats-np").replace(".png", ".npy")
            # sam_feats_path = raw_path.replace("kitti-depth", "kitti-depth-sam-feats")
        intrinsics = np.eye(3)
        focal = 0.5 * (intrinsics[0, 0] + intrinsics[1, 1])

        image = Image.open(raw_path)
        depth_gt = Image.open(gt_path)
        # with open(sam_feats_path, "rb") as sam_feats_f:
        #     sam_feats = torch.load(sam_feats_f, map_location='cpu')
        if sam_feats_path.endswith(".npy"):
            sam_feats = torch.from_numpy(np.load(sam_feats_path))
            sam_feats = torch.nn.functional.interpolate(sam_feats, scale_factor=4)
            # with open(sam_feats_path, "rb") as f:
            #     buf = io.BytesIO(f.read())
            #     sam_feats = torch.from_numpy(load_np(buf))
        else:
            h5f = h5py.File(sam_feats_path,'r')
            sam_feats = torch.from_numpy(h5f['data'][:])
            h5f.close()

        if self.args.dataset == 'kitti':
            self.image_height = 250
            self.image_width = 1200
            height, width = image.height, image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            if USE_SAM:
                sam_feats = self.load_kitti_sam_feats(raw_path)[
                    top_margin:top_margin+self.image_height, left_margin:left_margin+self.image_width
                ] 
        else:
            depth_gt = depth_gt.crop((43, 45, 608, 472))
            image = image.crop((43, 45, 608, 472))
            self.image_height = 416
            self.image_width = 544
            if USE_SAM:
                sam_feats = self.load_nyu_sam_feats(raw_path)[45:472, 43:608]
                
        if self.mode == 'train':
            image = np.asarray(image, dtype=np.float32) / 255.0
            if USE_SAM:
                image = np.concatenate((image, sam_feats), axis=-1)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / self.depth_normalizer
            image, depth_gt, intrinsics = self.train_preprocess(image, depth_gt, intrinsics)
            depth_gt_mask = np.logical_and(depth_gt > self.depth_min, depth_gt < self.depth_max)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'depth_mask': depth_gt_mask, 'intrinsics': intrinsics}
        else:
            image = np.asarray(image, dtype=np.float32) / 255.0
            if USE_SAM:
                image = np.concatenate((image, sam_feats), axis=-1)
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
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image[...,:3] = self.augment_image(image[...,:3])

        # NOTE(carter): moved to train.py because samples in the same batch must be re-scaled with the same alpha
        # Long range augmentation
        # do_resize = random.random()
        # if rank == 1 and do_resize > 0.5:
        #     print(f"Augmented")
        #     image, depth_gt, intrinsics = augment_long_range(image, depth_gt, intrinsics, alpha=1.5)
            
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
        image[:3] = self.normalize(image[:3])
        image = pad_images(image, multiple_of=32)[0]

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth, depth_mask = sample['depth'], sample['depth_mask']
        intrinsics = torch.tensor(sample['intrinsics'])
        depth = self.to_tensor(depth)
        depth_mask = self.to_tensor(depth_mask)
        depth = pad_images(depth, multiple_of=32)[0]
        depth_mask = pad_images(depth_mask, multiple_of=32)[0]

        dataset = sample.get("dataset", "kitti") #["dataset"]
        if self.mode == 'train':
            return {f'image_{dataset}': image, f'depth_{dataset}': depth, 'focal': focal, f'depth_mask_{dataset}': depth_mask, 'intrinsics': intrinsics}
        else:
            return {f'image': image, f'depth': depth, 'focal': focal, 'image_path': sample['image_path'], 'depth_path': sample['depth_path'], f'depth_mask': depth_mask, 'intrinsics': intrinsics}

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


class BothDatasets(Dataset):
    def _read_split(self, split):
        with open(split, 'r') as f:
            raw_gt_tuples = f.readlines()
            raw_paths, gt_paths = zip(*[fn.split(',') for fn in raw_gt_tuples])
            raw_paths = [p.replace('\n', '') for p in raw_paths]
            gt_paths = [p.replace('\n', '') for p in gt_paths]
        return raw_paths, gt_paths

    def _get_kitti_paths(self):
        split = "kitti/kitti_train.csv" if self.mode == "train" else "kitti/kitti_val.csv"
        return self._read_split(split)
    
    def _get_nyu_paths(self):
        split = "nyu/nyu_depth_v2_train.csv" if self.mode == "train" else "nyu/nyu_depth_v2_val.csv"
        return self._read_split(split)

    def __init__(self, args, mode, transform=None, eval=False):
        self.args = args
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.eval = eval
        self.image_height, self.image_width = None, None
        self.dataset_type = "both"
        
        kitti_raw_paths, kitti_gt_paths = self._get_kitti_paths()
        nyu_raw_paths, nyu_gt_paths = self._get_nyu_paths()

        self.raw_paths = kitti_raw_paths + nyu_raw_paths
        self.gt_paths = kitti_gt_paths + nyu_gt_paths
        self.path_to_dataset = {pth:"kitti" for pth in kitti_raw_paths} | {pth:"nyu" for pth in nyu_raw_paths}

        self.nyu_normalizer = 65535 / 10.0
        self.kitti_normalizer = 256
        self.depth_min = 1e-3
        self.depth_max = 100
        # TODO intrinsics if needed

    @staticmethod
    def load_nyu_sam_feats(raw_path: str) -> torch.Tensor:
        sam_f_path = raw_path.replace("nyu_depth_v2_sync", "nyu_sam_feats_downsample").replace(".png", ".npy")
        sam_feats = np.load(sam_f_path)
        sam_feats = torch.nn.functional.interpolate(torch.from_numpy(sam_feats), scale_factor=4).permute(0,2,3,1).numpy()
        return sam_feats[0]
    
    @staticmethod
    def load_kitti_sam_feats(raw_path: str) -> torch.Tensor:
        sam_feats_path = raw_path.replace("kitti-depth", "kitti-depth-sam-feats-np").replace(".png", ".h5")
        if not os.path.exists(sam_feats_path):
            sam_feats_path = raw_path.replace("kitti-depth", "kitti-depth-sam-feats-np").replace(".png", ".npy")
        
        if sam_feats_path.endswith(".npy"):
            with open(sam_feats_path, "rb") as f:
                buf = io.BytesIO(f.read())
                sam_feats = torch.from_numpy(load_np(buf))
        else:
            h5f = h5py.File(sam_feats_path,'r')
            sam_feats = h5f['data'][:]
            h5f.close()
        
        return np.transpose(sam_feats[0], (1, 2 , 0)).astype(np.float32)

    def __getitem__(self, idx):
        try:
            return self._getitem__(idx)
        except PIL.UnidentifiedImageError:
            print(f"Encountered Pillow loading error on {idx} with raw path {self.raw_paths[idx]} and gt path {self.gt_paths[idx]}")
            rand_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(rand_idx)

    def _getitem__(self, idx):
        raw_path = self.raw_paths[idx]
        gt_path = self.gt_paths[idx]
        dataset = self.path_to_dataset[raw_path]
        depth_norm = self.kitti_normalizer if dataset == "kitti" else self.nyu_normalizer

        image = Image.open(raw_path)
        depth_gt = Image.open(gt_path)

        # TODO intrinsics if needed
        intrinsics = np.zeros((3, 3))
        focal = 0.

        if dataset == 'kitti':
            height, width = image.height, image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            self.image_height = 352
            self.image_width = 1216
            if USE_SAM:
                sam_feats = self.load_kitti_sam_feats(raw_path)[
                    top_margin:top_margin+self.image_height, left_margin:left_margin+self.image_width
                ]
        else:
            depth_gt = depth_gt.crop((43, 45, 608, 472))
            image = image.crop((43, 45, 608, 472))
            self.image_height = 416
            self.image_width = 544
            if USE_SAM:
                sam_feats = self.load_nyu_sam_feats(raw_path)[45:472, 43:608]


        if self.mode == 'train':
            image = np.asarray(image, dtype=np.float32) / 255.0
            if USE_SAM:
                image = np.concatenate((image, sam_feats), axis=-1)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / depth_norm
            image, depth_gt, intrinsics = self.train_preprocess(image, depth_gt, intrinsics)
            depth_gt_mask = np.logical_and(depth_gt > self.depth_min, depth_gt < self.depth_max)
            sample = {
                'image': image, 
                'depth': depth_gt, 
                'focal': focal, 
                'depth_mask': depth_gt_mask, 
                'intrinsics': intrinsics,
                "dataset": dataset, 
            }
        else:
            image = np.asarray(image, dtype=np.float32) / 255.0
            if USE_SAM:
                image = np.concatenate((image, sam_feats), axis=-1)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / depth_norm
            depth_gt_mask = np.logical_and(depth_gt > self.depth_min, depth_gt < self.depth_max)
            sample = {
                'image': image, 
                'depth': depth_gt, 
                'focal': focal, 
                'image_path': raw_path, 
                'depth_path': gt_path, 
                'depth_mask': depth_gt_mask, 
                'intrinsics': intrinsics,
                "dataset": dataset, 
            }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def train_preprocess(self, image, depth_gt, intrinsics):
        """
        Applies flipping and gamma/brightness/color augmentation. Flipping currently disabled for consistent intrinsics

        Returns:
            image_aug, depth_gt_aug
        """
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image[...,:3] = self.augment_image(image[...,:3])

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


def _collate_key(batches, key):
    tensors = [batch[key] for batch in batches if key in batch]
    if len(tensors) > 0:
        return torch.stack(tensors)

    return None


def collate_both(batches):
    ret = {}
    for key in [
        "image_kitti",
        "depth_kitti",
        "depth_mask_kitti",
        "image_nyu",
        "depth_nyu",
        "depth_mask_nyu",
    ]:
        ret[key] = _collate_key(batches, key)
    
    if ret["image_kitti"] is None and ret["image_nyu"].shape[0] > 1:
        ret["image_kitti"] = ret['image_nyu'][:1]
        ret["depth_kitti"] = ret['depth_nyu'][:1]
        ret["depth_mask_kitti"] = ret['depth_mask_nyu'][:1]
        ret["image_nyu"] = ret['image_nyu'][1:]
        ret["depth_nyu"] = ret['depth_nyu'][1:]
        ret["depth_mask_nyu"] = ret['depth_mask_nyu'][1:]
    
    elif ret["image_nyu"] is None and ret["image_kitti"].shape[0] > 1:
        ret["image_nyu"] = ret['image_kitti'][:1]
        ret["depth_nyu"] = ret['depth_kitti'][:1]
        ret["depth_mask_nyu"] = ret['depth_mask_kitti'][:1]
        ret["image_kitti"] = ret['image_kitti'][1:]
        ret["depth_kitti"] = ret['depth_kitti'][1:]
        ret["depth_mask_kitti"] = ret['depth_mask_kitti'][1:]

    for key in ["image_path", "depth_path"]:
        if key in batches[0]:
            ret[key] = [batch[key] for batch in batches]
    
    return ret