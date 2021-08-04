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
    # ax[1].vlines(x=0, ymin=0, ymax=len(lesion), color='k');
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

## classes for CeA synthesis
class TransCustom(MapTransform): # from Identityd
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Identity`.
    """

    def __init__(self, keys: KeysCollection, path_synthesis, 
                 func_read_cea_aug, func_pseudo_healthy, scans_syns, decreasing_sequence, 
                 GEN, POST_PROCESS, mask_outer_ring, texture, new_value,
                 allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.new_value = new_value
        self.path_synthesis = path_synthesis
        self.func_read_cea_aug = func_read_cea_aug
        self.scans_syns = scans_syns
        self.func_pseudo_healthy = func_pseudo_healthy
        self.BATCH_SCAN = 0
        self.decreasing_sequence = decreasing_sequence
        self.GEN = GEN
        self.POST_PROCESS = POST_PROCESS
        self.mask_outer_ring = mask_outer_ring
        self._half_num_slices = 8
        self.texture = texture
        # self.scan_slices = torch.tensor(()) #[]


    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        
        #===
        # print(d.keys())
        # print(f"scan={d['image'].shape, d.get('label_meta_dict').get('filename_or_obj').split('Train/')[-1].split('_seg')[0]}")

        # print(f'KEYS={d.keys()}')
        # print(f"TRANS:\n{d.get('label_transforms')}")
        SCAN_NAME = d.get('label_meta_dict').get('filename_or_obj').split('Train/')[-1].split('_seg')[0] 
        SLICE = d.get('label_transforms')[3].get('extra_info').get('center')[-1]
        CENTER_Y = d.get('label_transforms')[3].get('extra_info').get('center')[0]
        CENTER_X = d.get('label_transforms')[3].get('extra_info').get('center')[1]
        path_synthesis2 = f'{str(self.path_synthesis)}/{SCAN_NAME}/'
        # print(f'path_synthesis2 = {path_synthesis2}')
        # print(f'SCAN_NAME = {SCAN_NAME}, SLICE = {SLICE}')
        # scan_slices = torch.tensor(())
        scan_slices = np.array([], dtype=np.float32).reshape(0,192,192)
        label_slices = np.array([], dtype=np.uint8).reshape(0,192,192)
        if SCAN_NAME in self.scans_syns:
          # print('the scan selected has augmentions')
          
          for SLICE_IDX, SLICE_I in enumerate(np.arange(SLICE - self._half_num_slices, SLICE + self._half_num_slices,1)):
            
            scan_slice = np.squeeze(d.get('image_1')[self.BATCH_SCAN,...,SLICE_I]) 
            label_slice = np.squeeze(d.get('label_1')[self.BATCH_SCAN,...,SLICE_I]) 
            # print(f'scan_slice = {scan_slice.shape}, forloop idx={SLICE_IDX}') 
            lesions_all, coords_all, masks_all, names_all, loss_all = self.func_read_cea_aug(path_synthesis2, SLICE_I)
            # print(len(lesions_all), len(coords_all), len(masks_all), len(names_all), len(loss_all))
            
            if len(lesions_all) > 0:
              slice_healthy_inpain = pseudo_healthy_with_texture(scan_slice, lesions_all, coords_all, masks_all, names_all, self.texture)
              
              mse_lesions = []
              mask_for_inpain = np.zeros_like(slice_healthy_inpain)
              for idx_x, (lesion, coord, mask, name) in enumerate(zip(lesions_all, coords_all, masks_all, names_all)):
                #get the right coordinates
                coords_big = [int(i) for i in name.split('_')[1:5]]
                coords_sums = coord + coords_big
                new_coords_mask = np.where(mask==1)[0]+coords_sums[0], np.where(mask==1)[1]+coords_sums[2]
                # syn_norm = lesion[GEN] *x_seq2[idx_x]
                if self.GEN<60:
                  if self.POST_PROCESS:
                    syn_norm = normalize_new_range4(lesion[self.GEN], scan_slice[new_coords_mask])#, log_seq_norm2[idx_x])#, 0.19)
                  else:
                    syn_norm = lesion[self.GEN]
                else:
                  syn_norm = lesion[self.GEN]

                # get the MSE between synthetic and original (for metrics)
                orig_lesion = get_orig_scan_in_lesion_coords(scan_slice, new_coords_mask)
                mse_lesions.append(np.mean(mask*(lesion[self.GEN] - orig_lesion)**2))

                syn_norm = syn_norm * mask  

                # add cea syn with absolute coords
                new_coords = np.where(syn_norm>0)[0]+coords_sums[0], np.where(syn_norm>0)[1]+coords_sums[2]
                slice_healthy_inpain[new_coords] = syn_norm[syn_norm>0]
                
                # inpaint the outer ring
                if self.mask_outer_ring:
                  mask_ring = make_mask_ring(syn_norm>0)
                  new_coords_mask_inpain = np.where(mask_ring==1)[0]+coords_sums[0], np.where(mask_ring==1)[1]+coords_sums[2] # mask outer rings for inpaint
                  mask_for_inpain[new_coords_mask_inpain] = 1
                
              if self.mask_outer_ring:
                slice_healthy_inpain = inpaint.inpaint_biharmonic(slice_healthy_inpain, mask_for_inpain)
              
            #   print(f'slice_healthy_inpain = {slice_healthy_inpain.shape, type(slice_healthy_inpain)}') 
            #   print('0000: yes augs yes lesion, adding slice_healthy_inpain')
              scan_slices = np.concatenate((scan_slices, crop_and_pad(slice_healthy_inpain, CENTER_Y, CENTER_X)), 0)
              label_slices = np.concatenate((label_slices, crop_and_pad(label_slice, CENTER_Y, CENTER_X)), 0)
            else:
            #   print('1111: yes augs no lesion, adding scan_slice')
              scan_slices = np.concatenate((scan_slices, crop_and_pad(scan_slice, CENTER_Y, CENTER_X)), 0)
              label_slices = np.concatenate((label_slices, crop_and_pad(label_slice, CENTER_Y, CENTER_X)), 0)
        else:
          for SLICE_I in np.arange(SLICE - self._half_num_slices, SLICE + self._half_num_slices,1):
            scan_slice = np.squeeze(d.get('image_1')[self.BATCH_SCAN,...,SLICE_I])
            label_slice = np.squeeze(d.get('label_1')[self.BATCH_SCAN,...,SLICE_I])
            # print('2222: no augmentations, adding scan_slice')
            scan_slices = np.concatenate((scan_slices, crop_and_pad(scan_slice, CENTER_Y, CENTER_X)), 0) 
            label_slices = np.concatenate((label_slices, crop_and_pad(label_slice, CENTER_Y, CENTER_X)), 0)
        # scan_slices = torch.unsqueeze(torch.swapaxes(scan_slices,0,-1),0) # np.zeros_like(d['image_1'][0,...,0]).shape
        scan_slices = np.expand_dims(np.swapaxes(scan_slices,0,-1),0) # np.zeros_like(d['image_1'][0,...,0]).shape
        label_slices = np.expand_dims(np.swapaxes(label_slices,0,-1),0)
        d['synthetic_lesion'] = scan_slices
        d['synthetic_label'] = label_slices
        return d

class TransCustom2(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.Identity`."""

    def __init__(self, keys, replace_image_for_synthetic:float = 0.333,
    allow_missing_keys: bool = False) -> None:
        """rotate90 to match rotation of 'image', then 
        apply flips previously applied to 'image'."""
        super().__init__(keys, allow_missing_keys)
        self.replace_image_for_synthetic = replace_image_for_synthetic

    def pad_because_monai_transf_is_not_doing_it(self, array):
        _ , shy, shx, ch = np.shape(array)
        size = 192
        if (size - shy) > 0 or (size - shx) > 0:
            print('we need to pad_because_monai_transf_is_not_doing_it')
            array = np.pad(array, ((0,0),(0,size-shy),(0,size-shx),(0,0)),mode='reflect')
        return array

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        flip0 = d.get('label_transforms')[5].get('do_transforms')
        flip1 = d.get('label_transforms')[6].get('do_transforms')
        flip2 = d.get('label_transforms')[7].get('do_transforms')
        affine_matrix = d.get('label_transforms')[4].get('extra_info').get('affine')
        # print(f'FLIPS = {flip0, flip1, flip2}')
        array_trans = d.get('synthetic_lesion')
        array_trans_lab = d.get('synthetic_label')
        aa = [np.shape(array_trans[0,...,i]) for i in range(16)]
        # print(f"affine before =>{array_trans.shape}, {aa}")
        # array_trans = self.pad_because_monai_transf_is_not_doing_it(array_trans)
        array_trans = np.rot90(array_trans,1,axes=[1,2])
        array_trans_lab = np.rot90(array_trans_lab,1,axes=[1,2])
        # print(f"affine after =>{array_trans.shape}")
        array_trans = np.squeeze(array_trans)
        array_trans_lab = np.squeeze(array_trans_lab)
        if flip0:
            array_trans = np.flip(array_trans,[0]).copy()
            array_trans_lab = np.flip(array_trans_lab,[0]).copy()
        if flip1:
            array_trans = np.flip(array_trans,[1]).copy()
            array_trans_lab = np.flip(array_trans_lab,[1]).copy()
        if flip2:
            array_trans = np.flip(array_trans,[2]).copy()
            array_trans_lab = np.flip(array_trans_lab,[2]).copy()
        d['synthetic_lesion'] = np.expand_dims(array_trans.copy(),0)
        d['synthetic_label'] = np.expand_dims(array_trans_lab.copy(),0)
        if np.random.rand() > self.replace_image_for_synthetic:
            # print('SWITCHED image & synthesis')
            # temp_image = d['synthetic_lesion']
            # temp_label = d['synthetic_label']
            # d['synthetic_lesion'] = d['image']
            # d['synthetic_label'] = d['label']
            d['image'] = d['synthetic_lesion']
            d['label'] = d['synthetic_label']       
        return d

class PrintTypesShapes(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Identity`.
    """

    def __init__(self, keys, text, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.text = text
        # self.identity = Identity()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.key_iterator(d):
            print(f"{self.text}={key, type(d[key]), d[key].shape, d[key].dtype}")
        return d