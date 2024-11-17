# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

KITTI_DEPTH_MIN = 1e-3
KITTI_DEPTH_MAX = torch.inf
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
        elif self.args.dataset == 'kitti':
            self.depth_normalizer = 256.0
            self.depth_min = KITTI_DEPTH_MIN
            self.depth_max = KITTI_DEPTH_MAX
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        raw_path = self.raw_paths[idx]
        gt_path = self.gt_paths[idx]
        # This param shouldn't matter
        focal = 518. #float(Path(raw_path).stem)

        image = Image.open(raw_path)
        depth_gt = Image.open(gt_path)

        # Store the image height and width for later use
        if self.image_height is None:
            self.image_height = image.height
            self.image_width = image.width
            # self.image_height, self.image_width = image.shape[:2]
        
        # if self.args.do_kb_crop is True:
        #     self.image_height = 352
        #     self.image_width = 1216
        #     height, width = image.shape[:2]
        #     top_margin = int(height - 352)
        #     left_margin = int((width - 1216) / 2)
        #     image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
        #     depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                
        if self.mode == 'train':
            # TODO: re-enable with intrinsics compensation
            # # To avoid blank boundaries due to pixel registration
            # if self.args.dataset == 'nyu':
            #     depth_gt = depth_gt.crop((43, 45, 608, 472))
            #     image = image.crop((43, 45, 608, 472))
            #     self.image_height = 416
            #     self.image_width = 544

            # if self.args.do_random_rotate is True:
            #     random_angle = (random.random() - 0.5) * 2 * self.args.degree
            #     image = self.rotate_image(image, random_angle)
            #     depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            # image, depth_gt = self.random_crop(image, depth_gt, self.image_height, self.args.image_width)
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / self.depth_normalizer
            depth_gt_mask = np.logical_and(depth_gt > self.depth_min, depth_gt < self.depth_max)
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'depth_mask': depth_gt_mask}
        else:
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / self.depth_normalizer
            depth_gt_mask = np.logical_and(depth_gt > self.depth_min, depth_gt < self.depth_max)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'image_path': raw_path, 'depth_path': gt_path, 'depth_mask': depth_gt_mask}

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

    def train_preprocess(self, image, depth_gt):
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
            image = self.augment_image(image)

        return image, depth_gt

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
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth, depth_mask = sample['depth'], sample['depth_mask']
        if self.mode == 'train':
            # TODO: why only in training?
            depth = self.to_tensor(depth)
            depth_mask = self.to_tensor(depth_mask)
            return {'image': image, 'depth': depth, 'focal': focal, 'depth_mask': depth_mask}
        else:
            return {'image': image, 'depth': depth, 'focal': focal, 'image_path': sample['image_path'], 'depth_path': sample['depth_path'], 'depth_mask': depth_mask}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
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
