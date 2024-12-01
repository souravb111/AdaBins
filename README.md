# Generalisation of Monocular Depth Estimation to OOD Data Regimes
Forked from https://github.com/shariqfarooq123/AdaBins

## Requirements
Experiments were conducted with Python 3.9, CUDA 12.1 and PyTorch 2.5.1
- `conda install pytorch-cuda=12.1 cuda -c pytorch -c nvidia` 
- Install nvcc with apt
- `pip install -r requirements.txt`

## Data Preparation
- NYU Depth V2: download the raw dataset parts for all the scenes here from https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html and preprocess the data with the author's toolkit. Set the dataset root path and run our split generation script `nyu/generate_split.py`
- KITTI Depth: download the [raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) and [depth labels](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) from the KITTI website. Set the dataset root path and run our split generation script `kitti/generate_split.py`
- SAM embeddings: update the paths to split files and run `scripts/dump_sam_embeddings.py` to dump sam feature maps for samples in the training and validation sets of NYU and KITTI

## Training
Training is currently configured to log to Weights and Biases. To log to your own wandb page, update the project name and team name accordingly in `train.py`.

Sample commands:
```
python train.py --dataset=kitti --filenames_file_eval kitti/kitti_val.csv --filenames_file kitti/kitti_train.csv --bs 4 --distributed
python train.py --dataset=nyu --filenames_file_eval nyu/nyu_depth_v2_val.csv --filenames_file nyu/nyu_depth_v2_train.csv --bs 4 --distributed
```

To run without logging, we can use the environment variable: `export DISABLE_LOGGING=1`

## Evaluation
Sample commands:
```
python evaluate.py --dataset kitti --filenames_file_eval kitti/kitti_val.csv --checkpoint_path /mnt/remote/shared_data/users/cfang/AdaBins/checkpoints/kitti_150_aug02.pt
python evaluate.py --dataset nyu --filenames_file_eval nyu/nyu_depth_v2_val.csv --checkpoint_path /mnt/remote/shared_data/users/cfang/AdaBins/checkpoints/kitti_150_aug02.pt

```

To run without logging, we can use the environment variable: `export DISABLE_LOGGING=1`

## Inference
Note that you must update the checkpoint path and dataset parameters in `infer.py`
```
python infer.py
```
