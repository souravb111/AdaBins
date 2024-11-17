import os
from pathlib import Path
import csv
from copy import deepcopy

mini = False
raw_dir = Path("/mnt/remote/shared_data/datasets/kitti-depth/2011_09_26")

## Val
split = 'val'
gt_dir = Path(f"/mnt/remote/shared_data/datasets/avg-kitti/{split}")
raw_gt_pairs = []
val_raw_paths = []

for sub_dir in gt_dir.iterdir():
    if 'sync' in sub_dir.name:
        for gt_path in (sub_dir / 'proj_depth/groundtruth/image_02').iterdir():
            raw_path = str(gt_path).replace(f'avg-kitti/{split}', 'kitti-depth/2011_09_26').replace('proj_depth/groundtruth/image_02', 'image_02/data')
            
            if Path(raw_path).exists() and gt_path.exists():
                print(f"{raw_path} -> {gt_path}")
                raw_gt_pairs.append([raw_path, str(gt_path)])
                val_raw_paths.append(raw_path)
            
            other_gt_path = str(gt_path).replace('image_02', 'image_03')
            other_raw_path = raw_path.replace('image_02', 'image_03')
            if Path(other_raw_path).exists() and Path(other_gt_path).exists():
                raw_gt_pairs.append([other_raw_path, other_gt_path])
                print(f"{other_raw_path} -> {other_gt_path}")
                val_raw_paths.append(raw_path)
            
    if mini and len(raw_gt_pairs) >= 100:
        break

if mini:
    out_path = f'kitti_mini_{split}.csv'
else:
    out_path = f'kitti_{split}.csv'
    
with open(out_path,'w') as out:
    csv_out=csv.writer(out)
    for row in raw_gt_pairs:
        csv_out.writerow(row)
        
## Train
split = 'train'
gt_dir = Path(f"/mnt/remote/shared_data/datasets/avg-kitti/{split}")
raw_gt_pairs = []
train_raw_paths = []

for sub_dir in gt_dir.iterdir():
    if 'sync' in sub_dir.name:
        for gt_path in (sub_dir / 'proj_depth/groundtruth/image_02').iterdir():
            raw_path = str(gt_path).replace(f'avg-kitti/{split}', 'kitti-depth/2011_09_26').replace('proj_depth/groundtruth/image_02', 'image_02/data')
            
            if Path(raw_path).exists() and gt_path.exists() and raw_path not in val_raw_paths:
                print(f"{raw_path} -> {gt_path}")
                raw_gt_pairs.append([raw_path, str(gt_path)])
            
            other_gt_path = str(gt_path).replace('image_02', 'image_03')
            other_raw_path = raw_path.replace('image_02', 'image_03')
            if Path(other_raw_path).exists() and Path(other_gt_path).exists() and raw_path not in val_raw_paths:
                raw_gt_pairs.append([other_raw_path, other_gt_path])
                print(f"{other_raw_path} -> {other_gt_path}")
            
    if mini and len(raw_gt_pairs) >= 100:
        break

if mini:
    out_path = f'kitti_mini_{split}.csv'
else:
    out_path = f'kitti_{split}.csv'
    
with open(out_path,'w') as out:
    csv_out=csv.writer(out)
    for row in raw_gt_pairs:
        csv_out.writerow(row)
        
