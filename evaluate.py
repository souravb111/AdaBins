import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import model_io
from dataloader import DepthDataLoader, KITTI_DEPTH_MAX, KITTI_DEPTH_MIN, NYU_DEPTH_MAX, NYU_DEPTH_MIN
from models import UnetAdaptiveBins
from utils import RunningAverageDict


KITTI_DISTANCE_BUCKETS = [
    [0, 30],
    [30, 60],
    [60, 80],
]

NYU_DISTANCE_BUCKETS = [
    [0, 4],
    [4, 8],
    [8, 10],
]

def compute_errors(gt, pred, eval_range=None):
    if eval_range is not None:
        mask = np.logical_and(gt >= eval_range[0], gt < eval_range[1])
        if mask.sum() == 0:
            return {}
        gt_eval = gt[mask]
        pred_eval = pred[mask]
        postfix = f"_{eval_range[0]:.1f}m_{eval_range[1]:.1f}m"
    else:
        gt_eval = gt.copy()
        pred_eval = pred.copy()
        postfix = ""
    
    thresh = np.maximum((gt_eval / pred_eval), (pred_eval / gt_eval))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt_eval - pred_eval) / gt_eval)
    sq_rel = np.mean(((gt_eval - pred_eval) ** 2) / gt_eval)

    rmse = (gt_eval - pred_eval) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt_eval) - np.log(pred_eval)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred_eval) - np.log(gt_eval)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt_eval) - np.log10(pred_eval))).mean()
    return {
        f"a1{postfix}": a1,
        f"a2{postfix}": a2,
        f"a3{postfix}": a3,
        f"abs_rel{postfix}": abs_rel,
        f"rmse{postfix}": rmse,
        f"log_10{postfix}": log_10,
        f"rmse_log{postfix}": rmse_log,
        f"silog{postfix}": silog,
        f"sq_rel{postfix}": sq_rel,
    }


def predict_tta(model, image, intrinsics, args):
    pred = model(image, intrinsics)[-1]
    pred = np.clip(pred.detach().cpu().numpy(), args.min_depth_eval, args.max_depth_eval)

    final = pred
    final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:], mode='bilinear', align_corners=True)
    return torch.Tensor(final)


def eval(model, test_loader, args, gpus=None, ):
    if gpus is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = gpus[0]

    if args.save_dir is not None and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    distance_buckets = KITTI_DISTANCE_BUCKETS if args.dataset == 'kitti' else NYU_DISTANCE_BUCKETS
    metrics = RunningAverageDict()

    total_invalid = 0
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in tqdm(sequential):

            image = batch['image'].to(device)
            gt = batch['depth'].to(device) 
            intrinsics = batch['intrinsics'].to(device)
            gt_mask = batch['depth_mask'].squeeze()
            sam_feats = batch['sam_feats'].to(device)
            final = predict_tta(model, image, intrinsics, args, sam_feats)
            final = final.squeeze().cpu().numpy()

            final[np.isinf(final)] = args.max_depth_eval
            final[np.isnan(final)] = args.min_depth_eval

            if args.save_dir is not None:
                if args.dataset == 'nyu':
                    impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '').replace('.png', '')}"
                    factor = 1
                else:
                    dpath = batch['image_path'][0].split('/')
                    impath = dpath[1] + "_" + dpath[-1]
                    impath = impath.split('.')[0]
                    factor = 256

                pred_path = os.path.join(args.save_dir, f"{impath}.png")
                pred = (final * factor) # .astype('uint16')
                pred_uint8 = ((pred / args.max_depth_eval) * 255).astype(np.uint8)
                gt_uint8 = ((gt[0,:,:,0].cpu().numpy() / args.max_depth_eval) * 255).astype(np.uint8)
                Image.fromarray(pred_uint8).save(pred_path)
                Image.fromarray(gt_uint8).save(pred_path.replace("eval_out/", "eval_out/gt_"))

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth_eval, gt < args.max_depth_eval)
            eval_mask = np.ones(valid_mask.shape)

            valid_mask = np.logical_and(valid_mask, eval_mask)
            valid_mask = np.logical_and(valid_mask, gt_mask.numpy())
            
            metrics.update(compute_errors(gt[valid_mask], final[valid_mask]))
            for bucket_range in distance_buckets:
                metrics.update(compute_errors(gt[valid_mask], final[valid_mask], bucket_range))

    print(f"Total invalid: {total_invalid}")
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")
    print([k for k in metrics.keys() if k.startswith("abs_rel_") or k.startswith("rmse_")])
    print([v for k,v in metrics.items() if k.startswith("abs_rel_") or k.startswith("rmse_")])


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':
    import os
    import debugpy
    if os.environ.get("ENABLE_DEBUGPY"):
        print("listening...")
        debugpy.listen(("127.0.0.1", 5678))
        debugpy.wait_for_client()
        
    # Arguments
    parser = argparse.ArgumentParser(description='Model evaluator', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--n-bins', '--n_bins', default=256, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument('--save-dir', '--save_dir', default=None, type=str, help='Store predictions in folder')

    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")
    parser.add_argument('--filenames_file_eval',
                        default="./kitti_eval.csv",
                        type=str, help='path to the filenames text file for online evaluation')
    
    parser.add_argument('--checkpoint_path', '--checkpoint-path', type=str, required=True,
                        help="checkpoint file to use for prediction")
    
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.both_data = False
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    args.distributed = False
    device = torch.device('cuda:{}'.format(args.gpu))
    test = DepthDataLoader(args, 'eval').data
    if args.dataset == 'kitti':
        args.min_depth, args.max_depth = KITTI_DEPTH_MIN, KITTI_DEPTH_MAX
        args.min_depth_eval, args.max_depth_eval = KITTI_DEPTH_MIN, KITTI_DEPTH_MAX
    elif args.dataset == 'nyu':
        args.min_depth, args.max_depth = NYU_DEPTH_MIN, NYU_DEPTH_MAX
        args.min_depth_eval, args.max_depth_eval = NYU_DEPTH_MIN, NYU_DEPTH_MAX
    else:
        raise NotImplementedError("Only KITTI and NYU are supported")
    model = UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=KITTI_DEPTH_MIN, max_val=KITTI_DEPTH_MAX, norm='linear').to(device)
    model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    model = model.eval()

    with torch.no_grad():
        eval(model, test, args, gpus=[device])
