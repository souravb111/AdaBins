import torch
import sys
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import PIL
import numpy as np
import os


kitti_train = "/mnt/remote/shared_data/users/jtu/kitti_train.csv"
kitti_val = "/mnt/remote/shared_data/users/jtu/kitti_val.csv"
nyu_train = "/mnt/remote/shared_data/users/jtu/nyu_train.csv"
nyu_val = "/mnt/remote/shared_data/users/jtu/nyu_val.csv"

def get_rgb_files(file_path):
    with open(file_path, "r") as f:
        examples = f.readlines()
        rgbs = [e.split(",")[0] for e in examples]
    return rgbs

def get_kitti_crops(kitti_img): 
    h, w, _ = kitti_img.shape
    crops = []
    for start_x in range(0, w - h, h):
        crops.append(kitti_img[:, start_x:start_x + h])
    
    if w % h > 0:
        last_crop = kitti_img[:, w - h:w]
        crops.append(last_crop)
    
    last_crop_w = w % h
    return crops, last_crop_w

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))


to_dump = get_rgb_files(nyu_train) + get_rgb_files(nyu_val)
n_dump = len(to_dump)

STRIDE = 6
OFFSET = 0 + int(sys.argv[1])


for idx in range(OFFSET, n_dump, STRIDE):
    rgb_file = to_dump[idx]
    print(idx, rgb_file, n_dump)
    try:
        image = Image.open(rgb_file)
    except PIL.UnidentifiedImageError: # some NYU images get this (courtesy of janky matlab script)
        continue
    image_array = np.array(image)

    save_path = rgb_file.replace("nyu_depth_v2_sync", "nyu_sam_feats")
    save_dir = os.path.dirname(save_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        crops, last_crop_w = get_kitti_crops(image_array)
        h, w, _ = image_array.shape
        
        features = [
            torch.nn.functional.interpolate(predictor.set_image(crop)[0], (h, h)) 
            for crop in crops
        ]

        if w % h == 0:
            all_features = torch.cat(features, -1)
        else:
            last_crop_w = w % h
            all_features = torch.cat(features[:-1] + [features[-1][..., -last_crop_w:]], -1)

        assert all_features.shape[-2] == h
        assert all_features.shape[-1] == w

        if os.path.exists(save_path):
            os.remove(save_path)
        #assert not os.path.exists(save_path)
        torch.save(all_features.detach().cpu(), save_path)
        print(f"saved to {save_path}")


