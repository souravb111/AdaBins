
import os
import lightning as L
from torch import optim, nn, utils, Tensor
from refactor.custom_dataset import CustomDataset, ToTensor
from refactor.custom_model import LitModel

filenames_file = "/home/james/AdaBins/kitti/kitti_train.csv"
dataset = CustomDataset("kitti", filenames_file, "train", transform=ToTensor("train"))
dataloader = utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=24, pin_memory=False, persistent_workers=True)
model = LitModel(256, 0, 150, "linear")

trainer = L.Trainer(max_steps=2500, devices=2)
trainer.fit(model=model, train_dataloaders=dataloader)
