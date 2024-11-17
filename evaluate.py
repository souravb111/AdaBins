import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import model_io
from dataloader import DepthDataLoader
from models import UnetAdaptiveBins
from utils import RunningAverageDict


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


# def denormalize(x, device='cpu'):
#     mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
#     std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
#     return x * std + mean
#
def predict_tta(model, image, args):
    pred = model(image)[-1]
    #     pred = utils.depth_norm(pred)
    #     pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred = np.clip(pred.cpu().numpy(), 10, 1000)/100.
    pred = np.clip(pred.detach().cpu().numpy(), args.min_depth_eval, args.max_depth_eval)

    # image = torch.Tensor(np.array(image.detach().cpu().numpy())[..., ::-1].copy()).to(device)
    # pred_lr = model(image)[-1]
    # #     pred_lr = utils.depth_norm(pred_lr)
    # #     pred_lr = nn.functional.interpolate(pred_lr, depth.shape[-2:], mode='bilinear', align_corners=True)
    # #     pred_lr = np.clip(pred_lr.cpu().numpy()[...,::-1], 10, 1000)/100.
    # pred_lr = np.clip(pred_lr.detach().cpu().numpy()[..., ::-1], args.min_depth_eval, args.max_depth_eval)
    # final = 0.5 * (pred + pred_lr)
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

    metrics = RunningAverageDict()
    # crop_size = (471 - 45, 601 - 41)
    # bins = utils.get_bins(100)
    total_invalid = 0
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in tqdm(sequential):

            image = batch['image'].to(device)
            gt = batch['depth'].to(device) 
            final = predict_tta(model, image, args)
            final = final.squeeze().cpu().numpy()

            # final[final < args.min_depth] = args.min_depth
            # final[final > args.max_depth] = args.max_depth
            final[np.isinf(final)] = args.max_depth_eval
            final[np.isnan(final)] = args.min_depth_eval

            if args.dataset == 'nyu':
                gt *= args.max_depth_eval

            if args.save_dir is not None:
                if args.dataset == 'nyu':
                    impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '').replace('.png', '')}"
                    factor = 1
                else:
                    dpath = batch['image_path'][0].split('/')
                    impath = dpath[1] + "_" + dpath[-1]
                    impath = impath.split('.')[0]
                    factor = 256

                # rgb_path = os.path.join(rgb_dir, f"{impath}.png")
                # tf.ToPILImage()(denormalize(image.squeeze().unsqueeze(0).cpu()).squeeze()).save(rgb_path)

                pred_path = os.path.join(args.save_dir, f"{impath}.png")
                pred = (final * factor) # .astype('uint16')
                pred_uint8 = ((pred / args.max_depth_eval) * 255).astype(np.uint8)
                gt_uint8 = ((gt[0,:,:,0].cpu().numpy() / args.max_depth_eval) * 255).astype(np.uint8)
                Image.fromarray(pred_uint8).save(pred_path)
                Image.fromarray(gt_uint8).save(pred_path.replace("eval_out/", "eval_out/gt_"))

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    # print("Invalid ground truth")
                    total_invalid += 1
                    continue

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth_eval, gt < args.max_depth_eval)
            eval_mask = np.ones(valid_mask.shape)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            #             gt = gt[valid_mask]
            #             final = final[valid_mask]

            metrics.update(compute_errors(gt[valid_mask], final[valid_mask]))

    print(f"Total invalid: {total_invalid}")
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")


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

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=100)
    
    parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--do_kb_crop', help='Use kitti benchmark cropping', action='store_true')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # args = parser.parse_args()
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    args.distributed = False
    device = torch.device('cuda:{}'.format(args.gpu))
    test = DepthDataLoader(args, 'eval').data
    model = UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth_eval, max_val=args.max_depth_eval,
                                   norm='linear').to(device)
    model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    model = model.eval()

    with torch.no_grad():
        eval(model, test, args, gpus=[device])
