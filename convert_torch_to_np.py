import numpy as np
import os
import torch
from fast_np import save, load
import sys

with open("nyu/nyu_depth_v2_train.csv") as f:
    pths = [x.split(",")[0].replace("nyu_depth_v2_sync", "nyu_sam_feats") for x in f.readlines()]   
with open("nyu/nyu_depth_v2_val.csv") as f:
    pths2 = [x.split(",")[0].replace("nyu_depth_v2_sync", "nyu_sam_feats") for x in f.readlines()]   

#with open("/mnt/remote/shared_data/users/jtu/kitti_train.csv") as f:
#    pths = [x.split(",")[0].replace("kitti-depth", "kitti-depth-sam-feats") for x in f.readlines()]   
#with open("/mnt/remote/shared_data/users/jtu/kitti_val.csv") as f:
#    pths2 = [x.split(",")[0].replace("kitti-depth", "kitti-depth-sam-feats") for x in f.readlines()]   

pths = pths + pths2

STRIDE = 8
START = 5600 + int(sys.argv[1])

idcs = list(range(len(pths)))

for i in idcs[START::STRIDE]:
    print(i, len(idcs))
    pth = pths[i]
    
    if not os.path.exists(pth):
        continue
    feats = torch.load(pth)
    feats = torch.nn.functional.interpolate(feats, scale_factor=0.25).numpy()
    dest = pth.replace("nyu_sam_feats", "nyu_sam_feats_downsample").replace(".png", ".npy")
    #dest = pth.replace("kitti-depth-sam-feats", "kitti-depth-sam-feats-np").replace(".png", ".npy")
    ddir = os.path.dirname(dest)
    if not os.path.isdir(ddir):
        try:
            os.makedirs(ddir)
        except FileExistsError:
            print("skip makefile, exists")
    print(f"Saving to {dest}")
    try:
        save(dest, feats)
    except Exception:
        save(dest, feats)



