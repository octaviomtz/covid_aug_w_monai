# %%
import argparse
import glob
import logging
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar

import monai
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

# %%
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.handlers import MetricsSaver

# %%
def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                    scale_range=(0.1, 0.1, None), 
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16), num_samples=3),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return monai.transforms.Compose(xforms)


def get_net():
    """returns a unet model instance."""

    n_classes = 2
    net = monai.networks.nets.BasicUNet(
        dimensions=3,
        in_channels=1,
        out_channels=n_classes,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
    )
    return net


def get_inferer(_mode=None):
    """returns a sliding window inference instance."""

    patch_size = (192, 192, 16)
    sw_batch_size, overlap = 2, 0.5
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy

# %%
data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20/Train'
images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))[:10]
labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))[:10]
len(images)

# %%
keys = ("image", "label")
train_frac, val_frac = 0.8, 0.2
n_train = int(train_frac * len(images)) + 1
n_val = min(len(images) - n_train, int(val_frac * len(images)))
n_train, n_val

# %%
train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]
type(train_files), type(train_files[0]), len(train_files)

# %%
batch_size = 2
train_transforms = get_xforms("train", keys)
train_transforms

# %%
train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

# %%
aa = next(iter(train_loader))
print(len(aa))
print(np.shape(aa['image'][0]))
IDX=4
SLICE=3
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(aa['image'][IDX][0,...,SLICE])
ax[0].imshow(aa['label'][IDX][0,...,SLICE], alpha=.1)
ax[1].imshow(aa['label'][IDX][0,...,SLICE])

# %%
# create a validation data loader
val_transforms = get_xforms("val", keys)
val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
val_loader = monai.data.DataLoader(
    val_ds,
    batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

# %%
# create BasicUNet, DiceLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = get_net().to(device)
max_epochs, lr, momentum = 1, 1e-4, 0.95
opt = torch.optim.Adam(net.parameters(), lr=lr)

# %%
print(net)

# %%
# create evaluator (to be used to measure model quality during training
model_folder="runs"
amp = True
val_post_transform = monai.transforms.Compose(
    [AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=True, n_classes=2)]
)
val_handlers = [
    ProgressBar(),
    CheckpointSaver(save_dir=model_folder, save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=3),
]
evaluator = monai.engines.SupervisedEvaluator(
    device=device,
    val_data_loader=val_loader,
    network=net,
    inferer=get_inferer(),
    post_transform=val_post_transform,
    key_val_metric={"val_mean_dice": MeanDice(include_background=False, output_transform=lambda x: (x["pred"], x["label"]))},
    val_handlers=val_handlers,
    amp=amp,
)

# %%
# evaluator as an event handler of the trainer
train_handlers = [
    ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
    StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
    MetricsSaver(save_dir=model_folder, metrics='*')
]
trainer = monai.engines.SupervisedTrainer(
    device=device,
    max_epochs=max_epochs,
    train_data_loader=train_loader,
    network=net,
    optimizer=opt,
    loss_function=DiceCELoss(),
    inferer=get_inferer(),
    key_train_metric=None,
    # key_train_metric={
    #         "train_mean_dice": MeanDice(include_background=False, output_transform=lambda x: (x["pred"], x["label"]))
    #     },
    train_handlers=train_handlers,
    amp=amp,
)

# %%
trainer.run()
# %%
print(trainer.state)
print(list(trainer.state.output))
print('oc')
print(trainer.state.output.get('loss'))

# %%
logging.info()
# %%
