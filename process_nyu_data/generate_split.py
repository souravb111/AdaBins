import os


# FOCAL_LENGTH = 518.8579 # this is a model param for adabins
DATA_DIR = "/mnt/remote/shared_data/datasets/nyu_depth_v2_sync/"

OUT_FILE = "nyu_depth_v2"

scenes = os.listdir(DATA_DIR)

synced_rgbd: dict[str, [tuple[str, str]]] = {}

for scene in scenes:
    scene_dir = os.path.join(DATA_DIR, scene)
    image_files = os.listdir(scene_dir)

    rgb = [scene + "/" + image_file for image_file in image_files if image_file.endswith("rgb.png")]
    depth = [rgb_file.replace("rgb.png", "depth.png") for rgb_file in rgb]

    rgb = [os.path.join(DATA_DIR, rgb_) for rgb_ in rgb]
    depth = [os.path.join(DATA_DIR, depth_) for depth_ in depth]

    synced_rgbd[scene] = (list(zip(rgb, depth)))

val_scenes = scenes[1::5]
train_scenes = [s for s in scenes if s not in val_scenes]

val_rgbd = [rgbd for scene in val_scenes for rgbd in synced_rgbd[scene]]
train_rgbd = [rgbd for scene in train_scenes for rgbd in synced_rgbd[scene]]

with open(f"{OUT_FILE}_val.csv", "w+") as f:
    for rgbd in val_rgbd:
        f.write(f"{rgbd[0]},{rgbd[1]}\n")

with open(f"{OUT_FILE}_train.csv", "w+") as f:
    for rgbd in train_rgbd:
        f.write(f"{rgbd[0]},{rgbd[1]}\n")








