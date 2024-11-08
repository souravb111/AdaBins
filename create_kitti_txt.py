import os
from pathlib import Path
import csv

split = 'val'
assert split in ['train', 'val']
raw_dir = Path("/mnt/remote/shared_data/datasets/kitti-depth/2011_09_26")
gt_dir = Path("/mnt/remote/shared_data/datasets/avg-kitti")
raw_gt_pairs = []

for sub_dir in gt_dir.iterdir():
    if 'sync' in sub_dir.name:
        for gt_path in (sub_dir / 'proj_depth/groundtruth/image_02').iterdir():
            raw_path = str(gt_path).replace(f'avg-kitti/{split}', 'kitti-depth/2011_09_26').replace('proj_depth/groundtruth/image_02', 'image_02/data')
            
            if Path(raw_path).exists() and gt_path.exists():
                raw_gt_pairs.append((raw_path, gt_path))
                print(f"{raw_path} -> {gt_path}")
                raw_gt_pairs.append([raw_path, str(gt_path)])

with open('kitti_eval.csv','w') as out:
    csv_out=csv.writer(out)
    for row in raw_gt_pairs:
        csv_out.writerow(row)