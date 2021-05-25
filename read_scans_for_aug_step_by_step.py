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
from scipy.ndimage import label
import matplotlib.patches as patches

# %%
def plot_image_with_lesion(mini_batch, IDX=0, SLICE=50):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(mini_batch['image'][IDX][0,...,SLICE])
    ax[0].imshow(mini_batch['label'][IDX][0,...,SLICE], alpha=.1)
    ax[1].imshow(mini_batch['label'][IDX][0,...,SLICE])

def coords_min_max_2D(array):
    '''return the min and max+1 of a mask. We use mask+1 to include the whole mask'''
    yy, xx = np.where(array==True)
    y_max = np.max(yy)+1; y_min = np.min(yy)
    x_max = np.max(xx)+1; x_min = np.min(xx)
    return y_min, y_max, x_min, x_max

def large_lesion_per_slice(scan, lesion_mask, LESION_SIZE_THRES = 100):
    '''go through all slices of a mask. In each slice label the components.
    Save largest lesion (lesion_slices_max_size).
    Save each lesion larger than LESION_SIZE_THRES (lesion_coords_larger_than)'''
    lesion_ch0, lesion_ch1, lesion_ch2 = np.where(lesion_mask==1)
    lesion_slices = np.unique(lesion_ch2)

    lesion_coords_larger_than = []
    lesion_slices_max_size = []
    mini_scan = []
    mini_mask = []
    for lesion_slice in lesion_slices:
        labelled, nr = label(lesion_mask[...,lesion_slice])
        lesion_size_o = 0
        for i in np.unique(labelled)[1:]: # skip background
            lesion_size_i = np.sum(labelled==i)
            if lesion_size_i > lesion_size_o: # largest lesion
                lesion_size_o = lesion_size_i
            if lesion_size_i > LESION_SIZE_THRES: # save coords all lesion larger than
                y_min, y_max, x_min, x_max = coords_min_max_2D(labelled==i)
                lesion_coords_larger_than.append((y_min, y_max, x_min, x_max, lesion_slice))
                mini_scan.append(scan[y_min:y_max, x_min:x_max, lesion_slice].numpy())                
                mini_mask.append(lesion_mask[y_min:y_max, x_min:x_max, lesion_slice].numpy())                
        lesion_slices_max_size.append(lesion_size_o)
    return lesion_coords_larger_than, lesion_slices_max_size, mini_scan, mini_mask

def plot_rect_with_coords(scan, lesion_mask, lesion_coords_larger_than, slice_, pad=10):
    lesions_in_this_slice = [i for i in lesion_coords_larger_than if i[-1]==slice_]
    fig, ax =plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(scan[...,lesions_in_this_slice[0][-1]])
    ax[0].imshow(lesion_mask[...,lesions_in_this_slice[0][-1]], alpha=.3)
    for i in lesions_in_this_slice:
        y_min = i[0]
        y_max = i[1]
        x_min = i[2]
        x_max = i[3]
        print(x_min, y_min, y_max-y_min, x_max-x_min)
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
    ax[1].imshow(scan[y_min-pad:y_max+pad,x_min-pad:x_max+pad,lesions_in_this_slice[0][-1]])
    ax[1].imshow(lesion_mask[y_min-pad:y_max+pad,x_min-pad:x_max+pad,lesions_in_this_slice[0][-1]],alpha=.3)

def set_individual_lesion_name(mini_batch, lesion_coords):
    '''construct name of individual lesion based on its scan name and its coords'''
    name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0]
    name_prefix = name_prefix.split('/')[-1].split('.nii')[0]
    name_prefix
    name_suffix = [f'{i}' for i in lesion_coords]
    name_suffix = '_'.join(name_suffix)
    name_lesion = '_'.join([name_prefix, name_suffix])
    return name_lesion

# %%
# ORIGINAL get_xforms
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

def get_xforms_load(mode="load", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        # ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "load":
        dtype = (np.int16, np.uint8)
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return monai.transforms.Compose(xforms)


# %%
data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20/Train'
folder_dest = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20/individual_lesions/'
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
transforms_load = get_xforms_load("load", keys)

# %%
train_ds = monai.data.CacheDataset(data=train_files, transform=transforms_load)
train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False, #should be true for training
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

# %%
mini_batch = next(iter(train_loader))
print(len(mini_batch))
print(np.shape(mini_batch['image'][0]))
print(mini_batch.keys())
plot_image_with_lesion(mini_batch, SLICE=50)

# %%
BATCH_IDX=0
scan = mini_batch['image'][BATCH_IDX][0,...]
lesion_mask = mini_batch['label'][BATCH_IDX][0,...]
lesion_ch0, lesion_ch1, lesion_ch2 = np.where(lesion_mask==1)
lesion_slices = np.unique(lesion_ch2)
print(np.shape(scan), np.shape(lesion_mask))
print(lesion_slices)
plot_image_with_lesion(mini_batch, SLICE=lesion_slices[10])

# %%
# get individual lesions per slice
lesion_coords_larger_than, lesion_slices_max_size, mini_scan, mini_mask = large_lesion_per_slice(scan, lesion_mask)
print(np.shape(lesion_coords_larger_than), np.shape(lesion_slices_max_size))
for idx, (i,j) in enumerate(zip(lesion_coords_larger_than, mini_scan)):
    if idx==2:break
    print(i, np.shape(j))

# %%
# get the slices that have lesions
slices_with_lesions=[]
for i in lesion_coords_larger_than:
    slices_with_lesions.append(i[-1])
slices_with_lesions = np.unique(slices_with_lesions)
slices_with_lesions

# %%
# plot to check images
SLICE=2
slices_with_lesions[SLICE]
plot_rect_with_coords(scan, lesion_mask, lesion_coords_larger_than, slices_with_lesions[SLICE])
plot_rect_with_coords(scan, lesion_mask, lesion_coords_larger_than, slices_with_lesions[SLICE],pad=0)

# %%
# make sure individual lesion a its mask are the same size and get its name
print(len(lesion_coords_larger_than),len(mini_scan),len(mini_mask))
for lesion_coords, i,j, in zip(lesion_coords_larger_than, mini_scan, mini_mask):
    assert np.shape(i) == np.shape(j)
for idx, (lesion_coords,i,j) in enumerate(zip(lesion_coords_larger_than, mini_scan, mini_mask)):
    if idx ==5:break
    assert np.shape(i) == np.shape(j)
    name_lesion = set_individual_lesion_name(mini_batch, lesion_coords)
    print(name_lesion, lesion_coords, np.shape(i), np.shape(j))


# %%
lesion_coords_larger_than[0]


# %%
# # All together
# for idx_mini_batch, mini_batch in enumerate(train_loader):
#     if idx_mini_batch==1:break
#     BATCH_IDX=0
#     scan = mini_batch['image'][BATCH_IDX][0,...]
#     lesion_mask = mini_batch['label'][BATCH_IDX][0,...]
#     # get individual lesions per slice
#     lesion_coords_larger_than, lesion_slices_max_size, mini_scan, mini_mask = large_lesion_per_slice(scan, lesion_mask)
#     # get the slices that have lesions
#     slices_with_lesions=[i[-1] for i in lesion_coords_larger_than]
#     slices_with_lesions = np.unique(slices_with_lesions)
#     # make sure individual lesion a its mask are the same size, get its name and save the images
#     for idx_individual_lesion, (lesion_coords, m_scan, m_mask) in enumerate(zip(lesion_coords_larger_than, mini_scan, mini_mask)):
#         if idx_individual_lesion ==5:break #OMM
#         assert np.shape(m_scan) == np.shape(m_mask)
#         name_lesion = set_individual_lesion_name(mini_batch, lesion_coords)
#         np.save(f'{folder_dest}{name_lesion}', m_scan)
#         np.savez_compressed(f'{folder_dest}{name_lesion}_mask', m_mask)
