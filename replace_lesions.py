# ##############################################
# INTERACTIVE CODE USED TO ADD THE SYNTHETIC LESIONS
# TO THE MONAI DATASETS
# ##############################################
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

#%%
from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from typing import Dict, Hashable, Mapping, Union
from utils_replace_lesions import get_decreasing_sequence, crop_and_pad 
from utils_replace_lesions import read_cea_aug_slice2, pseudo_healthy_with_texture, to_torch_right_shape, normalize_new_range4, get_orig_scan_in_lesion_coords, make_mask_ring
from scipy.ndimage import affine_transform

#%%
from monai.transforms.croppad.dictionary import RandCropByPosNegLabeld as RandCropByPosNegLabeld2

#%%
%reload_ext autoreload
%autoreload 2

#%%
path_parent = Path('/content/drive/My Drive/Datasets/covid19/COVID-19-20_augs_cea/')
path_synthesis = Path(path_parent / 'CeA_BASE_grow=1_bg=-1.00_step=-1.0_scale=-1.0_seed=1.0_ch0_1=-1_ch1_16=-1_ali_thr=0.1')

#%%
data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20/Train'
folder_dest = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20/individual_lesions/'
images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))[:10] #OMM
labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))[:10] #OMM
# =====
keys = ("image", "label")
train_frac, val_frac = 0.8, 0.2
n_train = int(train_frac * len(images)) + 1
n_val = min(len(images) - n_train, int(val_frac * len(images)))
# =====
train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]
print(f'train_files={len(train_files)}, val_files={len(val_files)}')

# READ THE SYTHETIC HEALTHY TEXTURE
path_synthesis_old = '/content/drive/My Drive/Datasets/covid19/results/cea_synthesis/patient0/'
texture_orig = np.load(f'{path_synthesis_old}texture.npy.npz')
texture_orig = texture_orig.f.arr_0
texture = texture_orig + np.abs(np.min(texture_orig)) + .07

# %%
scans_syns = os.listdir(path_synthesis)
decreasing_sequence = get_decreasing_sequence(255, splits= 20)
keys2=("image", "label", "synthetic_lesion")

#%%
class PrintTypesShapes(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Identity`.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        # self.identity = Identity()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.key_iterator(d):
            print(f"PPPPPPP={key, type(d[key]), d[key].shape, d[key].dtype}")
            # d[key] = self.identity(d[key])
        return d

# %%
# WITH FOR LOOP
class TransCustom(MapTransform): # from Identityd
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Identity`.
    """

    def __init__(self, keys: KeysCollection, path_synthesis, 
                 func_read_cea_aug, func_pseudo_healthy, scans_syns, decreasing_sequence, 
                 GEN, POST_PROCESS, mask_outer_ring, new_value,
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
        # self.scan_slices = torch.tensor(()) #[]


    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        
        #===
        # print(d.keys())
        # print(f"scan={d['image'].shape, d.get('label_meta_dict').get('filename_or_obj').split('Train/')[-1].split('_seg')[0]}")

        
        SCAN_NAME = d.get('label_meta_dict').get('filename_or_obj').split('Train/')[-1].split('_seg')[0] 
        SLICE = d.get('label_transforms')[3].get('extra_info').get('center')[-1]
        CENTER_Y = d.get('label_transforms')[3].get('extra_info').get('center')[0]
        CENTER_X = d.get('label_transforms')[3].get('extra_info').get('center')[1]
        path_synthesis2 = f'{str(path_synthesis)}/{SCAN_NAME}/'
        # print(f'path_synthesis2 = {path_synthesis2}')
        print(f'SCAN_NAME = {SCAN_NAME}, SLICE = {SLICE}')
        # scan_slices = torch.tensor(())
        scan_slices = np.array([], dtype=np.float32).reshape(0,192,192)
        label_slices = np.array([], dtype=np.uint8).reshape(0,192,192)
        if SCAN_NAME in self.scans_syns:
          print('the scan selected has augmentions')
          
          for SLICE_IDX, SLICE_I in enumerate(np.arange(SLICE - self._half_num_slices, SLICE + self._half_num_slices,1)):
            scan_slice = np.squeeze(d.get('image_1')[self.BATCH_SCAN,...,SLICE_I])
            label_slice = np.squeeze(d.get('label_1')[self.BATCH_SCAN,...,SLICE_I])
            # print(f'scan_slice = {scan_slice.shape}, forloop idx={SLICE_IDX}') 
            lesions_all, coords_all, masks_all, names_all, loss_all = self.func_read_cea_aug(path_synthesis2, SLICE_I)
            # print(len(lesions_all), len(coords_all), len(masks_all), len(names_all), len(loss_all))
            
            if len(lesions_all) > 0:
              slice_healthy_inpain = pseudo_healthy_with_texture(scan_slice, lesions_all, coords_all, masks_all, names_all, texture)
              
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

# %%
class TransCustom2(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.Identity`."""

    def __init__(self, replace_image_for_synthetic:float = 0.333) -> None:
        """Args:
            keys: keys of the corresponding items to be transformed.
            allow_missing_keys: don't raise exception if key is missing."""
        super().__init__(replace_image_for_synthetic)
        self.replace_image_for_synthetic = replace_image_for_synthetic

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        flip0 = d.get('label_transforms')[5].get('do_transforms')
        flip1 = d.get('label_transforms')[6].get('do_transforms')
        flip2 = d.get('label_transforms')[7].get('do_transforms')
        affine_matrix = d.get('label_transforms')[4].get('extra_info').get('affine')
        print(f'FLIPS = {flip0, flip1, flip2}')
        array_trans = d.get('synthetic_lesion')
        array_trans_lab = d.get('synthetic_label')
        # print(f"affine before =>{array_trans.shape}")
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
            print('SWITCHED image & synthesis')
            temp_image = d['synthetic_lesion']
            temp_label = d['synthetic_label']
            d['synthetic_lesion'] = d['image']
            d['synthetic_label'] = d['label']
            d['image'] = temp_image
            d['label'] = temp_label       
        return d

# %%
def get_xforms_with_synthesis(mode="synthesis", keys=("image", "label"), keys2=("image", "label", "synthetic_lesion")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        CopyItemsd(keys,1, names=['image_1', 'label_1']),
    ]
    if mode == "synthesis":
        xforms.extend([
                  SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                  RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16), num_samples=3),
                  TransCustom(keys, path_synthesis, read_cea_aug_slice2, 
                              pseudo_healthy_with_texture, scans_syns, decreasing_sequence, GEN=15,
                              POST_PROCESS=True, mask_outer_ring=True, new_value=.5),
                  RandAffined(
                      # keys,
                      keys2,
                      prob=0.15,
                      rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                      scale_range=(0.1, 0.1, None), 
                      mode=("bilinear", "nearest", "bilinear"),
                      # mode=("bilinear", "nearest"),
                      as_tensor_output=False
                  ),
                  RandGaussianNoised((keys2[0],keys2[2]), prob=0.15, std=0.01),
                  # RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                  RandFlipd(keys, spatial_axis=0, prob=0.5),
                  RandFlipd(keys, spatial_axis=1, prob=0.5),
                  RandFlipd(keys, spatial_axis=2, prob=0.5),
                  TransCustom2(0.333)
              ])
    dtype = (np.float32, np.uint8)
    # dtype = (np.float32, np.uint8, np.float32)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)

# %%
# create a training data loader
# LOAD the images with the FIRST SET of TRAIN transforms
batch_size = 1
# logging.info(f"batch size {batch_size}")
train_transforms_syn = get_xforms_with_synthesis("synthesis", keys, keys2)
train_ds_syn = monai.data.CacheDataset(data=train_files, transform=train_transforms_syn)
train_loader_syn = monai.data.DataLoader(
    train_ds_syn,
    batch_size=batch_size,
    shuffle=False,  # XX WARNING THIS SHOULD BE TRUE
    num_workers=1, # XX WARNING THIS SHOULD BE 2
    pin_memory=torch.cuda.is_available(), 
)

# %%
sample_syn = next(iter(train_loader_syn))
print(sample_syn.keys())
print(f"shape = {sample_syn['image'].shape}")
print(sample_syn.get('label_meta_dict').get('filename_or_obj'))
fig, ax = plt.subplots(3,4, figsize=(12,8))
for idx_batch in range(sample_syn['image'].shape[0]):
  for i in range(4):
    ax[idx_batch, i].imshow(sample_syn['image'][idx_batch,0,...,i+8])
    ax[idx_batch, i].imshow(sample_syn['label'][idx_batch,0,...,i+8], alpha=.3)
    ax[idx_batch, i].axis('off')
fig.tight_layout()

#%% CHECK THE ORDER OF THESE TRANSFORMS 
arr_aff = sample_syn.get('label_transforms')
arr_aff

# %%
# PLOT CUSTOM FIGS
fig, ax = plt.subplots(3,4, figsize=(12,8))
for idx_batch in range(sample_syn['synthetic_lesion'].shape[0]):
  for i in range(4):
    ax[idx_batch, i].imshow(sample_syn['synthetic_lesion'][idx_batch,0,...,i+8])
    ax[idx_batch, i].imshow(sample_syn['synthetic_label'][idx_batch,0,...,i+8], alpha=.3)
    ax[idx_batch, i].axis('off')
fig.tight_layout()

#%%
# PLOT IMG FIGS
fig, ax = plt.subplots(3,4, figsize=(12,8))
for idx_batch in range(sample_syn['image'].shape[0]):
  for i in range(4):
    ax[idx_batch, i].imshow(sample_syn['image'][idx_batch,0,...,i+8])
    ax[idx_batch, i].imshow(sample_syn['label'][idx_batch,0,...,i+8], alpha=.3)
    ax[idx_batch, i].axis('off')
fig.tight_layout()

# %%
print('with CUSTOM AUGS')
print(sample_syn.keys())
print(f"image = {sample_syn.get('image').shape}")
print(f"label = {sample_syn.get('label').shape}")
print(f"image_1 = {sample_syn.get('image_1').shape}")
print(f"label_1 = {sample_syn.get('label_1').shape}")
print(f"synthetic_lesion = {sample_syn.get('synthetic_lesion').shape}")

# %%
# print(sample_syn.get('label_transforms')[4])
arr_aff = sample_syn.get('label_transforms')[4].get('extra_info').get('affine')
print(arr_aff.shape)
print(arr_aff)
# %%
aa = sample_syn['synthetic_lesion'][0]
print(f'aa={aa.shape}')
print(f'aff={arr_aff[0].shape}')
print(aa.shape, arr_aff[0].shape)
# bb = affine_transform(aa[0], arr_aff[0][:3,:3])
# print(f'bb={bb.shape}')

#%%
sample_syn.keys()

#%%
# True, False, True   | 2
# True, True, True,   | 1
# True, False, False  | 2
# False, False, False | 
IDX=1
orig = sample_syn.get('image')[IDX].numpy()
label = sample_syn.get('label')[IDX].numpy()
cust = sample_syn.get('synthetic_lesion')[IDX].numpy()
cust1 = np.flip(cust,3)
cust1 = np.flip(cust1,1)
# cust1 = np.flip(cust1,3)
# cust1 = np.flip(cust1,2)
print(orig.shape, cust.shape)
fig, ax = plt.subplots(1,3,figsize=(9,3))
ax[0].imshow(orig[0,...,8])
ax[0].imshow(label[0,...,8], alpha=.3)
ax[1].imshow(cust[0,...,8])
ax[2].imshow(cust1[0,...,8])
ax[2].imshow(label[0,...,8], alpha=.3)
fig.tight_layout()

#%%
# CHECK THE ORDER OF THESE TRANSFORMS 
arr_aff = sample_syn.get('label_transforms')
arr_aff



#%%
fig, ax = plt.subplots(2,5, figsize=(18,9))
for i in range(10):
    ax.flat[i].imshow(cust1[0,...,i+4])
fig.tight_layout()

#%%
from utils_replace_lesions import crop_and_pad_multiple_x1

#%%
aa = sample_syn.get('image_1')[0].numpy()
print(aa.shape)
bb = crop_and_pad(aa[0,...,10], 100, 100)
print(bb.shape)
cc = crop_and_pad_multiple_x1(aa[0,...,10:26], 100, 100)
print(cc.shape)

#%%
from monai.transforms import Affine

#%%
affine = Affine(
    rotate_params=np.pi / 4,
    scale_params=(1.2, 1.2),
    translate_params=(200, 40),
    padding_mode="zeros",
)
a0 = aa[...,8]
new_img, _ = affine(a0, (192, 192), mode="bilinear")
print(new_img.shape)
plt.imshow(new_img[0])

#%%
aa = sample_syn.get('image').numpy()
print(aa.shape)
print(np.min(aa), np.max(aa))
aa = np.clip(aa,0,1)
print(np.min(aa), np.max(aa))

#%%
arr_aff = sample_syn.get('label_transforms')[4].get('extra_info').get('affine')[0]
arr_aff

#%%
import cv2

#%%
bb = cv2.cvtColor(aa, cv2.COLOR_GRAY2BGR)
bb.shape

#%%
fig, ax = plt.subplots(1,2)
ax[0].imshow(aa[0,...,8])
ax[1].imshow(bb[...,8])
# %%
arr_aff[0][:3,:3].shape
# %%
sample_syn.get('label_transforms')[4]



# %%
from scipy.ndimage import interpolation
a0 = aa[0,...,7:9].numpy()
print(a0.shape, arr_aff[0].shape)
plt.imshow(a0[...,0])

#%%
a0 = np.swapaxes(a0,-1,0)
a0.shape

#%%
a0 = aa.numpy()[0,...,:2]
aff = arr_aff.numpy()[:3,:3]
print(a0.shape, aff.shape)
s1 = 192.0/2
s2 = int(s1*2) 
c_in=0.5*np.array(a0.shape)
c_out=np.array((s1,s1,2))
print(c_in, c_out)
transform = aff
offset = c_in-c_out.dot(transform)
dst=interpolation.affine_transform(
  a0,transform.T,order=2,offset=offset,output_shape=(s2,s2),cval=0.0,output=np.float32)
np.shape(dst)

#%%
aff
# %%
fig, ax = plt.subplots(1,7,figsize=(18,3))
s1 = 192.0/2
s2 = int(s1*2) 
c_in=0.5*np.array(a0.shape)
c_out=np.array((s1,s1))
for i in np.arange(0,7):
    a=i*15.0*np.pi/180.0
    transform=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    offset=c_in-c_out.dot(transform)
    dst=interpolation.affine_transform(
        a0,transform.T,order=2,offset=offset,output_shape=(s2,s2),cval=0.0,output=np.float32)
    
    # plt.subplot(1,7,i+1);
    ax[i].axis('off');
    ax[i].imshow(dst)
plt.show()
# %%
a0.shape
# %%
plt.imshow(dst)
# %%
transform

# %%
arr_aff[0]
# %%
