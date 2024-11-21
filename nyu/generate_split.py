import os
import math

# FOCAL_LENGTH = 518.8579 # this is a model param for adabins
DATA_DIR = "/mnt/remote/shared_data/datasets/nyu_depth_v2_sync/"

OUT_FILE = "nyu_depth_v2"
MINI_VAL_FACTOR = 10

scenes = os.listdir(DATA_DIR)

synced_rgbd: dict[str, list[tuple[str, str]]] = {}

for scene in scenes:
    scene_dir = os.path.join(DATA_DIR, scene)
    image_files = os.listdir(scene_dir)

    rgb = [scene + "/" + image_file for image_file in image_files if image_file.endswith("rgb.png")]
    depth = [rgb_file.replace("rgb.png", "depth.png") for rgb_file in rgb]

    rgb = [os.path.join(DATA_DIR, rgb_) for rgb_ in rgb]
    depth = [os.path.join(DATA_DIR, depth_) for depth_ in depth]

    synced_rgbd[scene] = [
        [rgb_, depth_] for rgb_, depth_ in zip(rgb, depth)
        if os.path.exists(rgb_) and os.path.exists(depth_)
    ]

val_scenes = scenes[1::5]
train_scenes = [s for s in scenes if s not in val_scenes]

target_train = 50000
target_val = 10000

train = []
val = []

# get even amount of data from all scenes
# sample more uniformly in ordered frames
total = 0
num_from_scene: dict[str, int] = {s: 0 for s in train_scenes}
while total < target_train:
    for scene in train_scenes:
        if num_from_scene[scene] == len(synced_rgbd[scene]):
            continue
        num_from_scene[scene] += 1
        total += 1
        if total >= target_train:
            break

for scene, num in num_from_scene.items():
    if num == 0:
        continue
    stride = len(synced_rgbd[scene]) //  num
    train.extend(synced_rgbd[scene][::stride])


total = 0
num_from_scene: dict[str, int] = {s: 0 for s in val_scenes}
while total < target_val:
    for scene in val_scenes:
        if num_from_scene[scene] == len(synced_rgbd[scene]):
            continue
        num_from_scene[scene] += 1
        total += 1
        if total >= target_val:
            break

for scene, num in num_from_scene.items():
    if num == 0:
        continue
    stride = len(synced_rgbd[scene]) //  num
    val.extend(synced_rgbd[scene][::stride])


print(f"Saving {len(train)} images to train split.")
print(f"Saving {len(val)} images to val split.")
print(f"Saving {len(val) // MINI_VAL_FACTOR} images to mini val split.")

with open(f"{OUT_FILE}_val.csv", "w+") as f:
    for rgbd in val:
        f.write(f"{rgbd[0]},{rgbd[1]}\n")

with open(f"{OUT_FILE}_train.csv", "w+") as f:
    for rgbd in train:
        f.write(f"{rgbd[0]},{rgbd[1]}\n")

with open(f"{OUT_FILE}_mini_val.csv", "w+") as f:
    for rgbd in val[::MINI_VAL_FACTOR]:
        f.write(f"{rgbd[0]},{rgbd[1]}\n")









