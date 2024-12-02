import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import model_io
import utils
from models import UnetAdaptiveBins
from dataloader import KITTI_DEPTH_MAX, KITTI_DEPTH_MIN, NYU_DEPTH_MAX, NYU_DEPTH_MIN


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

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


class InferenceHelper:
    def __init__(self, checkpoint, dataset='nyu', device='cuda:0'):
        self.toTensor = ToTensor()
        self.device = device
        if dataset == 'nyu':
            self.min_depth = NYU_DEPTH_MIN
            self.max_depth = NYU_DEPTH_MAX
            self.saving_factor = 1000  # used to save in 16 bit
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = checkpoint
        elif dataset == 'kitti':
            self.min_depth = KITTI_DEPTH_MIN
            self.max_depth = KITTI_DEPTH_MAX
            self.saving_factor = 256
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = checkpoint
        else:
            raise ValueError("dataset can be either 'nyu' or 'kitti' but got {}".format(dataset))

        model, _, _ = model_io.load_checkpoint(pretrained_path, model)
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict_pil(self, pil_image, visualized=False):
        # pil_image = pil_image.resize((640, 480))
        img = np.asarray(pil_image) / 255.

        img = self.toTensor(img).unsqueeze(0).float().to(self.device)
        bin_centers, pred = self.predict(img)

        if visualized:
            viz = utils.colorize(torch.from_numpy(pred).unsqueeze(0), vmin=None, vmax=None, cmap='magma')
            # pred = np.asarray(pred*1000, dtype='uint16')
            viz = Image.fromarray(viz)
            return bin_centers, pred, viz
        return bin_centers, pred

    @torch.no_grad()
    def predict(self, image):
        bins, pred = self.model(image)
        pred = np.clip(pred.cpu().numpy(), self.min_depth, self.max_depth)

        # Flip
        image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(self.device)
        pred_lr = self.model(image)[-1]
        pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], self.min_depth, self.max_depth)

        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:],
                                          mode='bilinear', align_corners=True).cpu().numpy()

        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[np.isinf(final)] = self.max_depth
        final[np.isnan(final)] = self.min_depth

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.cpu().squeeze().numpy()
        centers = centers[centers > self.min_depth]
        centers = centers[centers < self.max_depth]

        return centers, final

    @torch.no_grad()
    def predict_dir(self, test_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        transform = ToTensor()
        all_files = glob.glob(os.path.join(test_dir, "*"))
        self.model.eval()
        for f in tqdm(all_files):
            image = np.asarray(Image.open(f), dtype='float32') / 255.
            image = transform(image).unsqueeze(0).to(self.device)

            centers, final = self.predict(image)
            # final = final.squeeze().cpu().numpy()

            final = (final * self.saving_factor).astype('uint16')
            basename = os.path.basename(f).split('.')[0]
            save_path = os.path.join(out_dir, basename + ".png")

            Image.fromarray(final.squeeze()).save(save_path)


if __name__ == '__main__':
    import os
    import debugpy
    if os.environ.get("ENABLE_DEBUGPY"):
        print("listening...")
        debugpy.listen(("127.0.0.1", 5678))
        debugpy.wait_for_client()
        
    import matplotlib.pyplot as plt
    from time import time
    # checkpoint = "/home/cfang/AdaBins/checkpoints/kitti_150_lr_aug.py"
    # checkpoint = "/mnt/remote/shared_data/users/cfang/AdaBins/checkpoints/kitti_150_baseline.pt"
    checkpoint = "/mnt/remote/shared_data/users/jtu/adabins/nyu_base/nyu_base_21-Nov_13-58-nodebs8-tep3-lr0.0001-wd0.1-1c17b213-cb62-4595-902e-ca18830d23ae_latest.pt"
    dataset = "nyu"
    filenames_file_eval = "/home/james/AdaBins/nyu/nyu_depth_v2_val.csv"
    # filenames_file_eval = "/home/james/AdaBins/kitti/kitti_val.csv"
    num_samples = 10
    out_dir = "/mnt/remote/shared_data/users/cfang/viz_nyu_baseline"
    os.makedirs(out_dir, exist_ok=True)
    
    inferHelper = InferenceHelper(checkpoint=checkpoint, dataset=dataset)
    
    with open(filenames_file_eval, 'r') as f:
        raw_gt_tuples = f.readlines()
        raw_paths, gt_paths = zip(*[fn.split(',') for fn in raw_gt_tuples])
        raw_paths = [p.replace('\n', '') for p in raw_paths]
        gt_paths = [p.replace('\n', '') for p in gt_paths]
    
    torch.manual_seed(1)
    rand_perm = torch.randperm(len(raw_paths))
    for i in range(num_samples):
        idx = rand_perm[i].item()
        raw_path, gt_path = raw_paths[idx], gt_paths[idx]
        image = Image.open(raw_path)

        centers, pred = inferHelper.predict_pil(image)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(311)
        ax.imshow(image)
        ax.axis('off')
        # ax.set_title("Input")
        
        depth_gt = np.array(Image.open(gt_path)) / 256.0
        ax = fig.add_subplot(312)
        # ax.set_title("Pred")
        ax.imshow(depth_gt, cmap='inferno')
        ax.axis('off')
        
        ax = fig.add_subplot(313)
        ax.imshow(pred.squeeze(), cmap='inferno')
        ax.axis('off')

        plt.savefig(f"{out_dir}/output_{i}.jpg")
        plt.close(fig)
    