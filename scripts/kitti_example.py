#!/usr/bin/python

from PIL import Image
import numpy as np
import cv2


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / 256.
    depth[depth_png == 0] = -1.
    return depth


if __name__ == "__main__":
    img_path = '/mnt/remote/shared_data/datasets/kitti-depth/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000008.png'
    depth_img = depth_read('/mnt/remote/shared_data/datasets/avg-kitti/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000008.png')

    raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(raw_img, 0.7, depth_img, 0.3, 0)
    cv2.imwrite('overlay.png', overlay_img)