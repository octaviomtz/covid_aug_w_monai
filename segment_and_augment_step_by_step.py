#%% IMPORTS
import argparse
import glob
import logging
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import math
from copy import copy
from scipy.ndimage import binary_closing
from scipy.ndimage import distance_transform_bf
from torch.utils.data import Dataset, DataLoader
import imageio
import os
from pathlib import Path
from tqdm.notebook import tqdm

import monai
from monai.transforms import (
    LoadImaged,
    ScaleIntensityRanged,
    CastToTyped,
    AddChanneld,
    Orientationd,
    Spacingd,
)

from utils import (
    select_lesions_match_conditions2, 
    superpixels, 
    coords_min_max_2D, 
    make_list_of_targets_and_seeds,
    fig_superpixels_only_lesions, fig_superpixels_ICCV)
from utils_cell_auto import (correct_label_in_plot, 
    create_sobel_and_identity, 
    prepare_seed, 
    epochs_in_inner_loop, 
    plot_loss_and_lesion_synthesis,
    to_rgb,
    CeA_BASE, CeA_BASE_1CNN 
    )
# from utils import coords_min_max_2D

from time import time
import torch.nn.functional as F
from skimage.morphology import remove_small_holes, remove_small_objects
import argparse

#%% AUTORELOAD  
%reload_ext autoreload
%autoreload 2

#%% FUNCTIONS
def get_xforms_load(mode="load", keys=("image", "label")):
    """returns a composed transform."""
    xforms = [
        LoadImaged(keys),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "load":
        dtype = (np.float32, np.uint8)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)

def get_xforms_load_scans(mode="load", keys=("image", "label")):
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
    # xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)

#%% HYPERPARAMETERS
SCAN_NAME = 'volume-covid19-A-0014'
SKIP_LESIONS = -1
ONLY_ONE_SLICE = 34
SEED_VALUE = 1
GROW_ON_K_ITER = 1
BACKGROUND_INTENSITY = 0.11
STEP_SIZE = 1
SCALE_MASK = 0.19 
CH0_1 = 1
CH1_16 = 15
ALIVE_THRESH = 0.1
ITER_INNER = 1
#%% READ INDIVIDUAL LESIONS
folder_source = f'/content/drive/MyDrive/Datasets/covid19/COVID-19-20/individual_lesions/{SCAN_NAME}_ct/'
files_scan = sorted(glob.glob(os.path.join(folder_source,"*.npy")))
files_mask = sorted(glob.glob(os.path.join(folder_source,"*.npz")))
keys = ("image", "label")
files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(files_scan, files_mask)]
print(len(files_scan), len(files_mask), len(files))

# %% DATASET DATALOADER
batch_size = 1
transforms_load = get_xforms_load("load", keys)
ds_lesions = monai.data.CacheDataset(data=files, transform=transforms_load)
loader_lesions = monai.data.DataLoader(
        ds_lesions,
        batch_size=batch_size,
        shuffle=False, #should be true for training
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
#%%
mask_sizes=[]
cluster_sizes = []
targets_all = []
for idx_mini_batch,mini_batch in enumerate(loader_lesions):
    if idx_mini_batch < SKIP_LESIONS:continue #resume incomplete reconstructions

    img = mini_batch['image'].numpy()
    mask = mini_batch['label'].numpy()
    mask = remove_small_objects(mask, 20)
    mask_sizes.append([idx_mini_batch, np.sum(mask)])
    name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.npy')[0].split('19-')[-1]
    img_lesion = img*mask
    
    # # if 2nd argument is provided then only analyze that slice
    if ONLY_ONE_SLICE != -1: 
        slice_used = int(name_prefix.split('_')[-1])
        if slice_used != int(ONLY_ONE_SLICE): continue

    #%%
    # mini_batch = next(iter(train_loader))
    print(f'mini_batch = {idx_mini_batch} {name_prefix}')
    print(mini_batch.keys())
    print(len(mini_batch))
    print(np.shape(mini_batch['image'][0]))
    print(name_prefix)

    #%%
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img[0])
    ax[0].imshow(mask[0],alpha=.3)
    ax[1].imshow(img_lesion[0])

    # SLIC
    # numSegments = 300 # run slic with large segments to eliminate background & vessels
    SCALAR_LIMIT_CLUSTER_SIZE = 200 #200 was used in the ICCV try \340
    numSegments = np.max([np.sum(mask[0]>0)//SCALAR_LIMIT_CLUSTER_SIZE, 1]) # run slic with large segments to eliminate background & vessels
    TRESH_BACK = 0.10 #orig=0.15
    THRES_VESSEL = 0.3 #orig=.5
    print(f'numSegments={numSegments}')
    if numSegments>1: # if mask is large then superpixels
        SCALAR_SIZE2 = 100
        numSegments = np.max([np.sum(mask[0]>0)//SCALAR_SIZE2, 4])
        segments = slic((img[0]).astype('double'), n_segments = numSegments, mask=mask[0], sigma = .2, multichannel=False, compactness=.1)
        background, lesion_area, vessels = superpixels((img[0]).astype('double'), segments, background_threshold=TRESH_BACK, vessel_threshold=THRES_VESSEL)
        mask_slic = lesion_area>0
        boundaries = mark_boundaries(mask_slic*img[0], segments)[...,0]
        # label_seg, nr_seg = label(segments)
    else: # small lesion (use the original mask)
        numSegments=-1
        background, lesion_area, vessels = superpixels((img[0]).astype('double'), mask[0], background_threshold=TRESH_BACK, vessel_threshold=THRES_VESSEL)
        mask_slic = mask[0]
        boundaries = np.zeros_like(mask_slic)
        segments = mask[0]
    segments_sizes = [np.sum(segments==i_segments) for i_segments in np.unique(segments)[1:]]
    cluster_sizes.append(segments_sizes)
    segments_sizes = [str(f'{i_segments}') for i_segments in segments_sizes]
    segments_sizes = '\n'.join(segments_sizes)

    # save vars for fig_slic
    background_plot = background;  lesion_area_plot = lesion_area
    vessels_plot = vessels; boundaries_plot = boundaries
    
    # remove small holes
    labelled, nr = label(mask_slic)
    mask_dil = remove_small_holes(remove_small_objects(mask_slic, 50))
    labelled2, nr2 = label(mask_dil)

    # separate each small lesion, make each seed and save them in a list
    tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(segments, img[0], skip_index=0)
    targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=SEED_VALUE, seed_method='max')
    targets_all.append(len(targets))

    ###
    coords_big = name_prefix.split('_')
    coords_big = [int(i) for i in coords_big[1:]]
    TRESH_P=10
    scan =  np.random.rand(400,400,60) # XX these variables exist in segment_and_augment
    scan_mask = np.random.randint(0,2,(400,400,60)) # XX these variables exist in segment_and_augment
    # if len(np.unique(background)) == 1 and len(np.unique(vessels)) ==1:
    #     fig_slic = fig_superpixels_only_lesions(scan, scan_mask, img, mask_slic, 
    #                                             boundaries_plot, segments, segments_sizes, coords_big, 
    #                                             TRESH_P, idx_mini_batch, numSegments)
    # else:
    #     fig_slic = fig_superpixels_ICCV(scan, scan_mask, img, mask_slic, 
    #                                     boundaries_plot, segments, segments_sizes, coords_big, 
    #                                     TRESH_P, idx_mini_batch, numSegments)


    # ======= CELLULAR AUTOMATA
    device = 'cuda'
    num_channels = 16
    epochs = 2500
    sample_size = 8
    path_parent = 'interactive/CeA_BASE1'
    path_save_synthesis = f'{path_parent}_={GROW_ON_K_ITER}_bg={BACKGROUND_INTENSITY:.02f}_step={STEP_SIZE}_scale={SCALE_MASK}_seed={SEED_VALUE}_ch0_1={CH0_1}_ch1_16={CH1_16}_ali_thr={ALIVE_THRESH}_iter={ITER_INNER}/'
    path_save_synthesis = f'{path_save_synthesis}{SCAN_NAME}/'
    Path(path_save_synthesis).mkdir(parents=True, exist_ok=True)#OMM
    path_synthesis_figs = f'{path_save_synthesis}fig_slic/'
    Path(f'{path_synthesis_figs}').mkdir(parents=True, exist_ok=True)
    fig_superpixels_only_lesions(path_synthesis_figs, name_prefix, scan, scan_mask, img, mask_slic, boundaries_plot, segments, segments_sizes, coords_big, TRESH_P, idx_mini_batch, numSegments)
    # fig_superpixels_only_lesions(path_synthesis_figs, name_prefix, scan, scan_mask, img, mask_slic, 
    #                             boundaries_plot, segments, segments_sizes, coords_big, 
    #                             TRESH_P, idx_mini_batch, numSegments)
    # fig_slic.savefig(f'{path_synthesis_figs}{name_prefix}_slic.png')
    

    if slice_used == int(ONLY_ONE_SLICE): break

#%%
background, lesion_area, vessels = superpixels((img[0]).astype('double'), segments, background_threshold=TRESH_BACK, vessel_threshold=THRES_VESSEL)
len(np.unique(background)), np.unique(lesion_area), len(np.unique(vessels))

#%%
from utils import fig_superpixels_only_lesions, fig_superpixels_only_lesions2

#%%
plt.imshow(vessels)
#%%
len(targets), len(coords), len(masks), len(seeds)

#%%
for (i, segVal) in enumerate(np.unique(segments)):
    im2 = (img[0]).astype('double')
    mask = np.zeros_like(im2)
    mask[segments == segVal] = 1
    clus = im2*mask
    # plt.figure()
    # plt.imshow(clus)
    median_intensity = np.median(clus[clus>0])
    print(median_intensity)

#%%
plt.imshow(im2)
######

#%%
train_ds = monai.data.CacheDataset(data=files, transform=transforms_load)
train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False, #should be true for training
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

#%%
mask_sizes=[]
for idx,mini_batch in enumerate(train_loader):
    img = mini_batch['image'].numpy()
    mask = mini_batch['label'].numpy()
    mask_sizes.append([idx, np.sum(mask)])
    img_lesion = img*mask
    if idx==0: break # to get a large mask

# %%
# mini_batch = next(iter(train_loader))
print(mini_batch.keys())
print(len(mini_batch))
print(np.shape(mini_batch['image'][0]))

#%%
fig, ax = plt.subplots(1,2)
ax[0].imshow(img[0])
ax[0].imshow(mask[0],alpha=.3)
ax[1].imshow(img_lesion[0])

#%%
# First use of SLIC, if the lesion is small only need to use SLIC once
numSegments = 300 # run slic with large segments to eliminate background & vessels
TRESH_BACK = 0.15
THRES_VESSEL = 0.5
segments = slic((img[0]).astype('double'), n_segments = numSegments, sigma = .2, multichannel=False, compactness=.1)
background, lesion_area, vessels = superpixels((img[0]).astype('double'), segments, background_threshold=TRESH_BACK, vessel_threshold=THRES_VESSEL)

# %%
fig, ax = plt.subplots(1,4, figsize=(16,6))
ax[0].imshow((background>0)*img[0], vmax=1)
ax[1].imshow((vessels>0)*img[0], vmax=1)
ax[2].imshow((lesion_area>0)*img[0], vmax=1)
boundaries = mark_boundaries((lesion_area>0)*img[0], segments)[...,0]
ax[3].imshow(boundaries*(lesion_area>0), vmax=1)
for axx in ax.ravel(): axx.axis('off')
fig.tight_layout()

# %%
lesion_area_only = (lesion_area>0)*img[0]
plt.imshow(lesion_area_only)
np.shape(lesion_area_only)

#%%
# IF LESION is too large us SLICMASK
lesion_area_only = (lesion_area>0)*img[0]
SCALAR_LIMIT_CLUSTER_SIZE = 340 # to make clusters size approx 40x40
y_min, y_max, x_min, x_max = coords_min_max_2D(lesion_area_only>0)
if np.sum(lesion_area_only>0) > SCALAR_LIMIT_CLUSTER_SIZE * 2 and (y_max-y_min) > 35 and (x_max-x_min) > 35:
    numSegments = np.sum(lesion_area_only>0)//SCALAR_LIMIT_CLUSTER_SIZE # run slic with large segments to eliminate background & vessels
    segments = slic((img[0]).astype('double'), n_segments = numSegments, mask=lesion_area_only>0, sigma = .2, multichannel=False, compactness=.1)
    background, lesion_area, vessels = superpixels((img[0]).astype('double'), segments, background_threshold=TRESH_BACK, vessel_threshold=THRES_VESSEL)
boundaries = mark_boundaries(lesion_area_only, segments)[...,0]
plt.imshow(boundaries)

#%%
tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(lesion_area, img[0], skip_index=0)
SEED_VALUE = .19
targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=SEED_VALUE, seed_method='max')

# %%
for (a,b,c,d) in zip(targets, coords, masks, seeds):
    print(np.shape(a), b, np.shape(c), np.shape(d))
# %%
rows=3; cols=3
fig, ax = plt.subplots(3,3,figsize=(rows*4,cols*4))
for idx, (i,j) in enumerate(zip(targets,seeds)):
  if idx == rows*cols: break
  ax.flat[idx].imshow(i[...,0])
  ax.flat[idx].imshow(j, alpha=.3)
  ax.flat[idx].text(3,3,idx,c='r')
fig.tight_layout()

# ======= CELLULAR AUTOMATA

#%%
device = 'cuda'
num_channels = 16
epochs = 2500
sample_size = 8
GROW_ON_K_ITER = 1
BACKGROUND_INTENSITY = 0.11 
STEP_SIZE = 1 
SCALE_MASK = .19
path_parent = '/content/drive/My Drive/Datasets/covid19/COVID-19-20_augs_cea/CeA_00'
path_save_synthesis = f'{path_parent}_grow={GROW_ON_K_ITER}_bg={BACKGROUND_INTENSITY:.02f}_step={STEP_SIZE}_scale_mask={SCALE_MASK}_seed_value={SEED_VALUE}/'
Path(path_save_synthesis).mkdir(parents=True, exist_ok=True)#OMM

#%%
for idx_lesion, (target, coord, mask, this_seed) in enumerate(zip(targets, coords, masks, seeds)):
  # if idx_lesion==3:break #OMM
  # prepare seed
  seed, seed_tensor, seed_pool = prepare_seed(target, this_seed, device, num_channels = num_channels, pool_size = 1024)

  # initialize model
  model = CeA_00(device = device, grow_on_k_iter=GROW_ON_K_ITER, background_intensity=BACKGROUND_INTENSITY, step_size=STEP_SIZE, scale_mask=SCALE_MASK)
  optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1500,2500], gamma=0.1) ## keep 1e-4 longer
  model_str = correct_label_in_plot(model)
  target = torch.tensor(target.transpose(-1,0,1)).unsqueeze(0).to(device)
  target_batch = torch.repeat_interleave(target, repeats = sample_size, dim = 0)

  losses = []
  alive_masks = []
  others=[]
  # train automata
  
  start = time()
  
  inner_iter_aux = 0
  inner_iter = 100
  inner_iters=[]
  for i in range(epochs):
    inner_iter, inner_iter_aux = epochs_in_inner_loop(i, inner_iter_aux, inner_iter)
    inner_iters.append(inner_iter)

    scheduler.step()
    batch_idx = np.random.choice(len(seed_pool), sample_size, replace = False)
    seed_batch = seed_pool[batch_idx].to(device)
    seed_batch[:1] = seed_tensor.to(device)
    
    loss, out, alive_mask, other = model.train_step(
        seed = seed_batch,
        target = target_batch, 
        target_loss_func = F.mse_loss, 
        epochs_inside = inner_iter,
        epoch_outside = i,
        masked_loss = False
        )
    
    alive_masks.append(alive_mask)
    others.append(other)

    seed_pool[batch_idx] = out.detach().to(device)
    loss.backward() # calculate gradients
    model.normalize_grads() # normalize them
    optimizer.step() # update weights and biases 
    optimizer.zero_grad() # prevent accumulation of gradients
    losses.append(loss.item())
    #early-stopping
    if loss.item() < 1e-5: break

    if i % 50==0 or i  == epochs-1:
      model_str_final = plot_loss_and_lesion_synthesis(losses, optimizer, model_str, i, loss, sample_size, out)

  stop = time(); time_total = f'{(stop-start)/60:.1f} mins'; print(time_total)
  model_str_final = model_str_final + f'\nep={i}, {time_total}' # for reconstruction figure
  #save model
  torch.save(model.model.state_dict(), f'{path_save_synthesis}weights_{idx_lesion}.pt')

  #lesion synthesis
  x = torch.tensor(seed).permute(0,-1,1,2).to(device)
  outs = []
  with torch.no_grad():
    for i,special_sequence in zip(range(256),[1,1,1,3]*64):
      # x = model(x,special_sequence,101)
      x, alive_mask_, others_ = model(x,i,101)
      out = np.clip(to_rgb(x[0].permute(-2, -1,0).cpu().detach().numpy()), 0,1)
      outs.append(out)
  
  #save results    
  outs_masked = []
  for out_ in outs:
      out_masked = np.squeeze(out_) * target[0,1,...].detach().cpu().numpy()
      out_masked[out_masked==1]=0
      outs_masked.append(out_masked)
  outs_float = np.asarray(outs_masked)
  print(np.shape(outs_float))
  outs_float = np.clip(outs_float, 0 ,1)
  # outs_int = (outs_int*255).astype('int16')
  print(idx_lesion)
  
  np.savez_compressed(f'{path_save_synthesis}lesion_{idx_lesion}.npz', outs_float)
  np.save(f'{path_save_synthesis}coords_lesion_{idx_lesion}.npy', coord)
  np.savez_compressed(f'{path_save_synthesis}mask_lesion_{idx_lesion}.npz', mask)
  np.save(f'{path_save_synthesis}loss_lesion_{idx_lesion}.npy', losses)
  np.save(f'{path_save_synthesis}total_time_lesion_{idx_lesion}_{time_total}.npy', time_total)


# %%
