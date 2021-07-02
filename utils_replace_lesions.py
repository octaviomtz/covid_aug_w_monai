#%%
import glob
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
    AddChanneld, AsDiscreted, CastToTyped, LoadImaged,
    Orientationd, RandAffined, RandCropByPosNegLabeld,
    RandFlipd, RandGaussianNoised, ScaleIntensityRanged,
    Spacingd, SpatialPadd, ToTensord, CopyItemsd
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.handlers import MetricsSaver
from scipy.ndimage import label
import matplotlib.patches as patches
from scipy.ndimage import binary_closing, binary_erosion, binary_dilation
from pathlib import Path

import imageio
import os
import moviepy.editor as mvp
from pathlib import Path
from copy import copy
from scipy.ndimage import binary_fill_holes
from skimage.restoration import inpaint
import cv2

from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from typing import Dict, Hashable, Mapping, Union

#%%
def read_cea_aug_slice2(path_parent, SLICE=10):
  '''read all files produced by read_scans_for_aug'''
  all_files = os.listdir(path_parent)
  # get only those files with 'less' then get only those from specific slice
  all_files = [i for i in all_files if 'les' in i if i.split('_')[5] == str(SLICE)]
  lesion_files = [i for i in all_files if 'lesion' in i and 'coords' not in i and 'mask' not in i and 'time' not in i and 'loss' not in i]
  coords_files = [i for i in all_files if 'coords' in i]
  masks_files = [i for i in all_files if 'mask' in i]
  loss_files = [i for i in all_files if 'loss' in i]
  # print(len(lesion_files),len(coords_files),len(masks_files))
  # read lesions and coordinates making sure both files exist
  lesions_all = []
  coords_all = []
  masks_all = []
  names_all = []
  loss_all = []

  # go through all lesions selected
  lesion_root_all = [i.split('_lesion')[0] for i in lesion_files]
  lesion_roots = np.unique(lesion_root_all)
  for root in lesion_roots:
    lesion_cluster = sorted([i for i in lesion_files if root in i])
    coords_cluster = sorted([i for i in coords_files if root in i])
    masks_cluster = sorted([i for i in masks_files if root in i])
    loss_cluster = sorted([i for i in loss_files if root in i])
    for (lesion, coords, mask, loss) in zip(lesion_cluster, coords_cluster, masks_cluster, loss_cluster):
      lesion_ = np.load(f'{path_parent}{lesion}')
      lesions_all.append(lesion_.f.arr_0)
      coords_all.append(np.load(f'{path_parent}{coords}'))
      mask_ = np.load(f'{path_parent}{mask}')
      masks_all.append(mask_.f.arr_0)
      names_all.append(lesion)
      loss_all.append(np.load(f'{path_parent}{loss}'))
  return lesions_all, coords_all, masks_all, names_all, loss_all

def check_index(idx, total_gen, split=1, splits=10):
  '''auxiliary recursive function of get_decreasing_sequence '''
  if idx<total_gen * split//splits:
    if idx%split==0:
      return idx
  else:
    split = split+1
    idx = check_index(idx, total_gen, split, splits)
    return idx

def get_decreasing_sequence(total_gen = 256, splits= 10, plot = False):
  '''create a sequence that takes every n elements where n keeps increasing.
  This functions calls check_index recursively
  Example: 1,2,3, 5,7,9, 12,15,18'''
  seq = []
  for i in range(total_gen):
    bb = check_index(i, total_gen, 1, splits)
    seq.append(bb)
  # remove None
  seq = list(filter(None, seq))
  seq.insert(0,0)
  if plot:
    xs = np.linspace(0,total_gen-1,total_gen)
    ys = np.zeros((total_gen))
    for i in seq:
      ys[i]=1
    plt.figure(figsize=(18,2))
    plt.scatter(xs, ys)
  return seq



def blur_masked_image(image, kernel_blur = (3,3)):
    '''https://answers.opencv.org/question/3031/smoothing-with-a-mask/'''
    mask_for_blur = image >0
    image[mask_for_blur == 0] = 0
    blurred_image = cv2.blur(image,kernel_blur)
    blurred_mask = cv2.blur(mask_for_blur.astype(float),kernel_blur)
    result = blurred_image / blurred_mask
    result = np.nan_to_num(result*mask_for_blur, nan=0)
    return result

def pseudo_healthy_with_texture(scan_slice, lesions_all, coords_all, masks_all, names_all, texture, iter_erosion_dilation = 1, plot= False, Tp=20):
    '''1. Read all clusters' masks in a lesion (a cluster is a part of a lesion obtained with slic) 
    2. Replace them for the synthetic healthy texture. 3. Make a mask ring on the outer perimeter of
    the mask and inpaint to blend the image '''
    slice_healthy = copy(scan_slice)
    mask_for_inpain = np.zeros_like(slice_healthy)
    Tp = 20
    for idx_x, (lesion, coord, mask, name) in enumerate(zip(lesions_all, coords_all, masks_all, names_all)):
        coords_big = [int(i) for i in name.split('_')[1:5]]
        coords_sums = coord + coords_big
        # print('LOADING: ', coord, coords_big, coords_sums[0], coords_sums[2], name)
        new_coords_mask = np.where(mask==1)[0]+coords_sums[0], np.where(mask==1)[1]+coords_sums[2]
        slice_healthy[new_coords_mask] = texture[new_coords_mask]
        # rings to inpaint
        mask_closed = binary_fill_holes(mask,)
        mask_in = (mask_closed).astype('int') - binary_erosion(mask_closed, iterations=iter_erosion_dilation)
        mask_out = binary_dilation(mask_closed, iterations=iter_erosion_dilation) - (mask_closed).astype('int')
        mask_ring = mask_in + mask_out
        new_coords_mask_inpain = np.where(mask_ring==1)[0]+coords_sums[0], np.where(mask_ring==1)[1]+coords_sums[2] # mask outer rings for inpaint
        mask_for_inpain[new_coords_mask_inpain] = 1
    slice_healthy_inpain = inpaint.inpaint_biharmonic(slice_healthy, mask_for_inpain)

    if plot:
        fig, ax = plt.subplots(1,3, figsize=(18,6))
        ax[0].imshow(scan_slice[coords_big[0]-Tp:coords_big[1]+Tp,coords_big[2]-Tp:coords_big[3]+Tp])
        ax[0].imshow(scan_mask_slice[coords_big[0]-Tp:coords_big[1]+Tp,coords_big[2]-Tp:coords_big[3]+Tp], alpha=.3)
        ax[1].imshow(slice_healthy[coords_big[0]-Tp:coords_big[1]+Tp,coords_big[2]-Tp:coords_big[3]+Tp]);
        ax[2].imshow(slice_healthy_inpain[coords_big[0]-Tp:coords_big[1]+Tp,coords_big[2]-Tp:coords_big[3]+Tp]);
    
    return slice_healthy_inpain

def fig_blend_lesion(slice_healthy_inpain2, coords_big, GEN, decreasing_sequence, path_synthesis, file_path="images/image.png", Tp=10, V_MAX=1, close=True, plot_size=6):
    name_synthesis = path_synthesis.split('/')[-3]
    name_synthesis_two_lines = '\n_scale'.join(name_synthesis.split('_seed')) # two lines
    fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [30, 1]}, figsize=(8,8));
    ax[0].imshow(slice_healthy_inpain2[coords_big[0]-Tp:coords_big[1]+Tp,coords_big[2]-Tp:coords_big[3]+Tp], cmap='viridis', vmin=0, vmax=V_MAX);
    ax[0].text(2,8, GEN, fontsize=20, c='r')
    ax[0].text(2,5, name_synthesis_two_lines, fontsize=14, c='r')
    ax[1].vlines(x=0, ymin=0, ymax=len(lesion), color='k');
    ax[1].scatter(0,GEN, c='k', s=decreasing_sequence[-1]);
    ax[1].set_ylim([0,decreasing_sequence[-1]]);
    ax[1].text(0,0, 0, fontsize=20, c='k')
    ax[1].text(0,decreasing_sequence[-1]-5, decreasing_sequence[-1], fontsize=20, c='k')
    ax[1].text(0,0, 0, fontsize=20, c='k')
    ax[1].text(0,decreasing_sequence[-1]-5, decreasing_sequence[-1], fontsize=20, c='k')
    for axx in ax.ravel(): axx.axis('off')
    fig.tight_layout();
    plt.savefig(file_path);
    if close:
        plt.clf();

def make_mask_ring(mask, iter_erosion_dilation=1):
    mask_closed = binary_fill_holes(mask)
    mask_in = (mask_closed).astype('int') - binary_erosion(mask_closed, iterations=iter_erosion_dilation)
    mask_out = binary_dilation(mask_closed, iterations=iter_erosion_dilation) - (mask_closed).astype('int')
    mask_ring = mask_in + mask_out
    return mask_ring

def get_orig_scan_in_lesion_coords(scan_slice, new_coords_mask):
    yy, xx = new_coords_mask
    x_min, x_max = np.min(xx), np.max(xx)+1
    y_min, y_max = np.min(yy), np.max(yy)+1
    return scan_slice[y_min: y_max, x_min: x_max]

def normalize_new_range4(array, orig_distr, scale=1, default_min=-1):
  array_new_range_orig_shape = np.zeros_like(array)
  if default_min == -1:
    OldRange = (np.max(array[array>0]) - np.min(array[array>0]))  
    NewRange = (scale*np.max(orig_distr) - np.min(orig_distr))  
    array_new_range = (((array[array>0] - np.min(array[array>0])) * NewRange) / OldRange) + np.min(orig_distr)
    # print(np.shape(array_new_range), np.shape(array_new_range_orig_shape))
    array_new_range_orig_shape[np.where(array>0)] = array_new_range
    # print('default normalize values')
  else:
    array_new_range = ((array-np.min(array[array>0]))/(np.max(array[array>0])-np.min(array[array>0]))) * ((scale * np.max(orig_distr)- default_min) + default_min)
  return array_new_range_orig_shape

def to_torch_right_shape(scan_slice, cy, cx, pad_to_this_len=192):
  '''Aux function of TransCustom that changes numpy array into the right torch shape.
  1. We crop first only until now because the coords used to insert the lesions were based on
  original coordinates.
  2. After cropping we add pad if needed'''
  pp = pad_to_this_len/2
  sh_y, sh_x = np.shape(scan_slice)

  cy1_cut, cy1_pad = [(pp,0) if cy > pp else (cy,pp-cy)][0]
  cy2_cut, cy2_pad = [(pp,0) if sh_y > cy + pp  else (sh_y-cy,cy+pp-sh_y)][0]
  cx1_cut, cx1_pad = [(pp,0) if cx > pp else (cx,pp-cx)][0]
  cx2_cut, cx2_pad = [(pp,0) if sh_x > cx + pp  else (sh_x-cx,cx+pp-sh_x)][0]
  cy1_cut = int(cy1_cut)
  cy2_cut = int(cy2_cut)
  cx1_cut = int(cx1_cut)
  cx2_cut = int(cx2_cut)
  cy1_pad = int(cy1_pad)
  cy2_pad = int(cy2_pad)
  cx1_pad = int(cx1_pad)
  cx2_pad = int(cx2_pad)
  print(f'cut={cy1_cut,cy2_cut,cx1_cut,cx2_cut}')
  print(f'padd={cy1_pad,cy2_pad,cx1_pad,cx2_pad}')
  scan_slice = scan_slice[cy-cy1_cut:cy+cy2_cut, cx-cx1_cut:cx+cx2_cut]
  print(f'before pad = {np.shape(scan_slice)}')
  # scan_slice = nn.functional.pad(scan_slice,(cx1_pad,cx2_pad,cy1_pad,cy2_pad))
  scan_slice = np.pad(scan_slice, ((cy1_pad,cy2_pad),(cx1_pad,cx2_pad)),mode='reflect')
  print(f'after pad = {np.shape(scan_slice)}')
  
  scan_slice = torch.from_numpy(scan_slice)
  scan_slice = torch.unsqueeze(scan_slice,0)

  # use the centers to get the right position
  return scan_slice

def crop_and_pad(scan_slice, cy, cx, pad_to_this_len=192):
  '''Aux function of TransCustom that changes numpy array into the right shape.
  1. We crop first only until now because the coords used to insert the lesions were based on
  original coordinates.
  2. After cropping we add pad if needed'''
  pp = pad_to_this_len/2
  sh_y, sh_x = np.shape(scan_slice)

  cy1_cut, cy1_pad = [(pp,0) if cy > pp else (cy,pp-cy)][0]
  cy2_cut, cy2_pad = [(pp,0) if sh_y > cy + pp  else (sh_y-cy,cy+pp-sh_y)][0]
  cx1_cut, cx1_pad = [(pp,0) if cx > pp else (cx,pp-cx)][0]
  cx2_cut, cx2_pad = [(pp,0) if sh_x > cx + pp  else (sh_x-cx,cx+pp-sh_x)][0]
  cy1_cut = int(cy1_cut)
  cy2_cut = int(cy2_cut)
  cx1_cut = int(cx1_cut)
  cx2_cut = int(cx2_cut)
  cy1_pad = int(cy1_pad)
  cy2_pad = int(cy2_pad)
  cx1_pad = int(cx1_pad)
  cx2_pad = int(cx2_pad)
#   print(f'cut={cy1_cut,cy2_cut,cx1_cut,cx2_cut}')
#   print(f'padd={cy1_pad,cy2_pad,cx1_pad,cx2_pad}')
  scan_slice = scan_slice[cy-cy1_cut:cy+cy2_cut, cx-cx1_cut:cx+cx2_cut]
#   print(f'before pad = {np.shape(scan_slice)}')
  # scan_slice = nn.functional.pad(scan_slice,(cx1_pad,cx2_pad,cy1_pad,cy2_pad))
  scan_slice = np.pad(scan_slice, ((cy1_pad,cy2_pad),(cx1_pad,cx2_pad)),mode='reflect')
#   print(f'after pad = {np.shape(scan_slice)}')
  
  scan_slice = np.expand_dims(scan_slice,0)
  # print(f'from crop_and_pad = {np.shape(scan_slice)}')
  # use the centers to get the right position
  return scan_slice

def crop_and_pad_multiple_x1(scan_slice, cy, cx, pad_to_this_len=192):
  '''Aux function of TransCustom that changes numpy array into the right shape.
  1. We crop first only until now because the coords used to insert the lesions were based on
  original coordinates.
  2. After cropping we add pad if needed'''
  pp = pad_to_this_len/2
  sh_y, sh_x, _ = np.shape(scan_slice)

  cy1_cut, cy1_pad = [(pp,0) if cy > pp else (cy,pp-cy)][0]
  cy2_cut, cy2_pad = [(pp,0) if sh_y > cy + pp  else (sh_y-cy,cy+pp-sh_y)][0]
  cx1_cut, cx1_pad = [(pp,0) if cx > pp else (cx,pp-cx)][0]
  cx2_cut, cx2_pad = [(pp,0) if sh_x > cx + pp  else (sh_x-cx,cx+pp-sh_x)][0]
  cy1_cut = int(cy1_cut)
  cy2_cut = int(cy2_cut)
  cx1_cut = int(cx1_cut)
  cx2_cut = int(cx2_cut)
  cy1_pad = int(cy1_pad)
  cy2_pad = int(cy2_pad)
  cx1_pad = int(cx1_pad)
  cx2_pad = int(cx2_pad)
#   print(f'cut={cy1_cut,cy2_cut,cx1_cut,cx2_cut}')
#   print(f'padd={cy1_pad,cy2_pad,cx1_pad,cx2_pad}')
  scan_slice = scan_slice[cy-cy1_cut:cy+cy2_cut, cx-cx1_cut:cx+cx2_cut, :]
#   print(f'before pad = {np.shape(scan_slice)}')
  # scan_slice = nn.functional.pad(scan_slice,(cx1_pad,cx2_pad,cy1_pad,cy2_pad))
  scan_slice = np.pad(scan_slice, ((cy1_pad,cy2_pad),(cx1_pad,cx2_pad),(0,0)),mode='reflect')
#   print(f'after pad = {np.shape(scan_slice)}')
  
  scan_slice = np.expand_dims(scan_slice,0)
  # print(f'from crop_and_pad = {np.shape(scan_slice)}')
  # use the centers to get the right position
  return scan_slice