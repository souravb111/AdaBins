import numpy as np
import os, glob
from fastnumpyio import save, load
import h5py
import hdf5plugin
paths = list(glob.iglob("/mnt/remote/shared_data/datasets/kitti-depth-sam-feats-np/**/*.npy", recursive=True))

for i, path in enumerate(paths):
    print(f"Path: {i}")
    if 'fp16' not in path:
        data = load(path)
        fp16_path = path.replace(".npy", "_fp16.npy")
        if os.path.exists(fp16_path):
            os.remove(fp16_path)
        
        h5_path = path.replace(".npy", ".h5")
        h5f = h5py.File(h5_path, 'w')
        h5f.create_dataset('data', data=data.astype(np.float16), compression=hdf5plugin.Zstd(clevel=11))
        h5f.close()