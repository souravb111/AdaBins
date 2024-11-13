import os
from pathlib import Path
import csv

split = 'train'
assert split in ['train', 'val']
raw_dir = Path("/mnt/remote/shared_data/datasets/kitti-depth/2011_09_26")
gt_dir = Path(f"/mnt/remote/shared_data/datasets/avg-kitti/{split}")
raw_gt_pairs = []

for sub_dir in gt_dir.iterdir():
    if 'sync' in sub_dir.name:
        for gt_path in (sub_dir / 'proj_depth/groundtruth/image_02').iterdir():
            raw_path = str(gt_path).replace(f'avg-kitti/{split}', 'kitti-depth/2011_09_26').replace('proj_depth/groundtruth/image_02', 'image_02/data')
            
            if Path(raw_path).exists() and gt_path.exists():
                print(f"{raw_path} -> {gt_path}")
                raw_gt_pairs.append([raw_path, str(gt_path)])
            
            other_gt_path = str(gt_path).replace('image_02', 'image_03')
            other_raw_path = raw_path.replace('image_02', 'image_03')
            if Path(other_raw_path).exists() and Path(other_gt_path).exists():
                raw_gt_pairs.append([other_raw_path, other_gt_path])
                print(f"{other_raw_path} -> {other_gt_path}")

with open(f'kitti_{split}.csv','w') as out:
    csv_out=csv.writer(out)
    for row in raw_gt_pairs:
        csv_out.writerow(row)