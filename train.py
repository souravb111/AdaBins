import argparse
import os
import sys
import uuid
from datetime import datetime as dt
import gc

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (99999, rlimit[1]))

import random
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import wandb
from tqdm import tqdm

import model_io
import models
import utils
from dataloader import DepthDataLoader, KITTI_DEPTH_MAX, KITTI_DEPTH_MIN, NYU_DEPTH_MAX, NYU_DEPTH_MIN, augment_long_range_tensors
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage, colorize
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

# os.environ['WANDB_MODE'] = 'dryrun'
PROJECT = "adabins"
logging = True
logging_print_interval = 5
logging_vis_interval = 3000
if os.environ.get("DISABLE_LOGGING"):
    logging = False

def is_rank_zero(args):
    return args.rank == 0


import matplotlib


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################
    model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
    ################################################################################################

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
          experiment_name=args.name, optimizer_state_dict=None)


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}" #_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        # if args.dataset != 'nyu':
        #     PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)
        wandb.watch(model)
    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    for i, _ in enumerate(train_loader):
        print(f"{i} / {train_loader}")
    return
    test_loader = DepthDataLoader(args, 'eval').data 

    ###################################### losses ##############################################
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    ################################################################################################

    model.train()

    ###################################### Optimizer ################################################
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": lr}]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    ###################################### Scheduler ###############################################
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=lr/100., last_epoch=args.last_epoch)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
    #                                           cycle_momentum=True,
    #                                           base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
    #                                           div_factor=args.div_factor,
    #                                           final_div_factor=args.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)
    ################################################################################################

    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):
        ################################# Train loop ##########################################################
        if should_log: wandb.log({"Epoch": epoch}, step=step)
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(
                args) else enumerate(train_loader):

            img = batch['image']
            depth = batch['depth']
            depth_mask = batch['depth_mask']
            intrinsics = batch['intrinsics']

            # # Long range augmentation
            # if random.random() < 0.2:
            #     img, depth, intrinsics = augment_long_range_tensors(img, depth, intrinsics, alpha=1.333)
            #     depth_mask = torch.logical_and(depth > args.min_depth, depth < args.max_depth)
            
            img = img.to(device)
            depth = depth.to(device)
            depth_mask = depth_mask.to(device)
            intrinsics = intrinsics.to(device)
            
            optimizer.zero_grad()
            
            bin_edges, pred = model(img, intrinsics)
            l_dense = criterion_ueff(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True)

            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)

            loss = l_dense + args.w_chamfer * l_chamfer
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            if should_log and step > 0 and step % logging_print_interval == 0:
                wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
                wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)
                for j, lr in enumerate(scheduler.get_last_lr()):
                    wandb.log({f"LR_{j}": lr})

                # if step % logging_vis_interval == 0: # TODO can tune this
                #     log_images(img.clone().cpu(), depth.clone().cpu(), pred.detach().clone().cpu(), args, step)

            step += 1
            scheduler.step()

            ########################################################################################################
            if should_write and step % args.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)

                print("Validated: {}".format(metrics))
                if should_log: 
                    wandb.log({
                        f"Test/{criterion_ueff.name}": val_si.get_value(),
                        # f"Test/{criterion_bins.name}": val_bins.get_value()
                    }, step=step)

                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_latest.pt",
                                             root=os.path.join(root, "checkpoints"))

                if metrics['abs_rel'] < best_loss and should_write:
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_best.pt",
                                             root=os.path.join(root, "checkpoints"))
                    best_loss = metrics['abs_rel']
                model.train()
                #################################################################################################

    return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device) 
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            depth_mask = batch['depth_mask'].to(device).squeeze(-1)
            intrinsics = batch['intrinsics'].to(device)
            bins, pred = model(img, intrinsics)

            l_dense = criterion_ueff(pred, depth, mask=depth_mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
            pred = pred.squeeze().cpu().numpy()
            pred[pred < test_loader.dataset.depth_min] = test_loader.dataset.depth_min
            pred[pred > test_loader.dataset.depth_max] = test_loader.dataset.depth_max
            pred[np.isinf(pred)] = test_loader.dataset.depth_max
            pred[np.isnan(pred)] = test_loader.dataset.depth_min

            gt_depth = depth.squeeze().cpu().numpy()
            gt_depth_mask = depth_mask.squeeze().cpu().numpy()
            metrics.update(utils.compute_errors(gt_depth[gt_depth_mask], pred[gt_depth_mask]))
            
            # valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            # if args.garg_crop or args.eigen_crop:
            #     gt_height, gt_width = gt_depth.shape
            #     eval_mask = np.zeros(valid_mask.shape)

            #     if args.garg_crop:
            #         eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
            #         int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            #     elif args.eigen_crop:
            #         if args.dataset == 'kitti':
            #             eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
            #             int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            #         else:
            #             eval_mask[45:471, 41:601] = 1
            # valid_mask = np.logical_and(valid_mask, eval_mask)
            # metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


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
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--epochs', default=4, type=int, help='number of total epochs to run')
    parser.add_argument('--n-bins', '--n_bins', default=256, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")

    parser.add_argument('--bs', default=4, type=int, help='batch size')
    parser.add_argument('--validate-every', '--validate_every', default=3000, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")

    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument('--filenames_file',
                        default="./process_nyu_data/nyu_depth_v2_train.csv",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)

    parser.add_argument('--do_random_rotate', default=True,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')

    parser.add_argument('--filenames_file_eval',
                        default="./process_nyu_data/nyu_depth_v2_mini_val.csv",
                        type=str, help='path to the filenames text file for online evaluation')

    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.chamfer = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.dataset == 'kitti':
        args.min_depth, args.max_depth = KITTI_DEPTH_MIN, KITTI_DEPTH_MAX
    elif args.dataset == 'nyu':
        args.min_depth, args.max_depth = NYU_DEPTH_MIN, NYU_DEPTH_MAX
    else:
        raise NotImplementedError("Only KITTI and NYU are supported")
    
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
