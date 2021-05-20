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
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.handlers import MetricsSaver

# %%
from scipy.ndimage import label

# %%
def plot_image_with_lesion(mini_batch, IDX=0, SLICE=50):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(mini_batch['image'][IDX][0,...,SLICE])
    ax[0].imshow(mini_batch['label'][IDX][0,...,SLICE], alpha=.1)
    ax[1].imshow(mini_batch['label'][IDX][0,...,SLICE])

def coords_min_max_2D(array):
    yy, xx = np.where(array==True)
    y_max = np.max(yy); y_min = np.min(yy)
    x_max = np.max(xx); x_min = np.min(xx)
    return y_min, y_max, x_min, x_max

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


# %%
data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20/Train'
images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))[:10]
labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))[:10]
print(f'len(images)={len(images)}')

keys = ("image", "label")
train_frac, val_frac = 0.8, 0.2
n_train = int(train_frac * len(images)) + 1
n_val = min(len(images) - n_train, int(val_frac * len(images)))
print(f'n_train, n_val = {n_train, n_val}')

train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]
print(type(train_files), type(train_files[0]), len(train_files))

# %%
batch_size = 1
transforms = get_xforms("val", keys)
transforms_read

# %%
train_ds = monai.data.CacheDataset(data=train_files, transform=transforms_read)
train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

# %%
mini_batch = next(iter(train_loader))
print(len(mini_batch))
print(np.shape(mini_batch['image'][0]))
plot_image_with_lesion(mini_batch, SLICE=50)

# %%
mini_batch.keys()

# %%
BATCH_IDX=0
lesion = mini_batch['label'][BATCH_IDX][0,...]
print(np.shape(lesion))
lesion_ch0, lesion_ch1, lesion_ch2 = np.where(lesion==1)
lesion_slices = np.unique(lesion_ch2)
print(lesion_slices)
plot_image_with_lesion(mini_batch, SLICE=lesion_slices[0])

# %%
def large_lesion_per_slice(lesion_mask, LESION_SIZE_THRES = 100):
    '''go through all slices of a mask. In each slice label the components.
    Save largest lesion (lesion_slices_max_size).
    Save each lesion larger than LESION_SIZE_THRES (lesion_coords_larger_than)'''
    lesion_ch0, lesion_ch1, lesion_ch2 = np.where(lesion_mask==1)
    lesion_slices = np.unique(lesion_ch2)

    lesion_coords_larger_than = []
    lesion_slices_max_size = []
    for lesion_slice in lesion_slices:
        labelled, nr = label(lesion_mask[...,lesion_slice])
        lesion_size_o = 0
        for i in np.unique(labelled)[1:]: # skip background
            lesion_size_i = np.sum(labelled==i)
            if lesion_size_i > lesion_size_o: # largest lesion
                lesion_size_o = lesion_size_i
            if lesion_size_i > LESION_SIZE_THRES: # save coords all lesion larger than
                y_min, y_max, x_min, x_max = coords_min_max_2D(labelled==i)
                lesion_coords_larger_than.append((lesion_slice,y_min, y_max, x_min, x_max))
                ## CONTINUE HERE. USE the coords to return also the lesion (we need to input also the image)
        lesion_slices_max_size.append(lesion_size_o)
    return lesion_coords_larger_than, lesion_slices_max_size

# %%
lesion_coords_larger_than, lesion_slices_max_size = large_lesion_per_slice(lesion)
print(np.shape(lesion_coords_larger_than), np.shape(lesion_slices_max_size))

# %%
lesion_slices_max_size

# %%
# Get the size of the largest lesion per slice
LESION_SIZE_THRES = 100
lesion_slices_large = []
lesion_slices_max_size = []
for lesion_slice in lesion_slices:
    labelled, nr = label(mini_batch['label'][BATCH_IDX][0,...,lesion_slice])
    lesion_size_o = 0
    for i in np.unique(labelled)[1:]: # skip background
        lesion_size_i = np.sum(labelled==i)
        if lesion_size_i > lesion_size_o: # largest lesion
            lesion_size_o = lesion_size_i
        if lesion_size_i > LESION_SIZE_THRES: # save coords lesion
            y_min, y_max, x_min, x_max = coords_min_max_2D(labelled==i)
            lesion_slices_large.append((lesion_slice,y_min, y_max, x_min, x_max))
    lesion_slices_max_size.append(lesion_size_o)
    

print('======')
for i,j in zip(lesion_slices,lesion_slices_max_size):
    print(i,j)
print(len(lesion_slices), len(lesion_slices_max_size))

# %%
print(lesion_slices)
print(np.unique(labelled), i)

# %%
plot_image_with_lesion(mini_batch, SLICE=6)

# %%
y_min, y_max, x_min, x_max = coords_min_max_2D(labelled==i)

# %%

# %%
lesion_large_slices
# %%
plot_image_with_lesion(mini_batch, SLICE=29)

# %%
lesion_slices_large
# %%
