import numpy as np
import os, glob
from fastnumpyio import save, load
paths = list(glob.iglob("/mnt/remote/shared_data/datasets/kitti-depth-sam-feats-np/**/*.npy", recursive=True))

for i, path in enumerate(paths):
    print(f"Path: {i}")
    if 'fp16' not in path:
        data = load(path)
        new_path = path.replace(".npy", "_fp16.npy")
        with open(new_path, "wb") as out_f:
            save(out_f, data.astype(np.float16))