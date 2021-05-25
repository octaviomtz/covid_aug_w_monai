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
    make_list_of_targets_and_seeds)
from utils_cell_auto import (correct_label_in_plot, 
    create_sobel_and_identity, 
    prepare_seed, 
    epochs_in_inner_loop, 
    plot_loss_and_lesion_synthesis,
    to_rgb,
    CeA_00
    )

#%%
from time import time
import torch.nn.functional as F
from skimage.morphology import remove_small_holes, remove_small_objects
#%%
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

#%%
SCAN_NAME = 'volume-covid19-A-0011_ct/'

# LOAD ORIGINAL SCAN /=== FOR fig_slic, later can be removed
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

batch_size = 1
transforms_load = get_xforms_load_scans("load", keys)
train_ds = monai.data.CacheDataset(data=train_files, transform=transforms_load)
train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False, #should be true for training
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

for idx_mini_batch, mini_batch in enumerate(train_loader):
    # if idx_mini_batch==1:break #OMM
    BATCH_IDX=0
    scan = mini_batch['image'][BATCH_IDX][0,...]
    scan_mask = mini_batch['label'][BATCH_IDX][0,...]
    scan_name = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii')[0]
    if scan_name == SCAN_NAME[:-1]: break
scan = scan.numpy()
scan_mask = scan_mask.numpy()

# LOAD ORIGINAL SCAN ===/

# LOAD INDIVIDUAL LESIONS
folder_source = f'/content/drive/MyDrive/Datasets/covid19/COVID-19-20/individual_lesions/{SCAN_NAME}'
files_scan = sorted(glob.glob(os.path.join(folder_source,"*.npy")))
files_mask = sorted(glob.glob(os.path.join(folder_source,"*.npz")))
keys = ("image", "label")
files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(files_scan, files_mask)]
print(len(files_scan), len(files_mask), len(files))

# %%
batch_size = 1
transforms_load = get_xforms_load("load", keys)

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
for idx_mini_batch,mini_batch in enumerate(train_loader):
    img = mini_batch['image'].numpy()
    mask = mini_batch['label'].numpy()
    mask = remove_small_objects(mask, 20)
    mask_sizes.append([idx_mini_batch, np.sum(mask)])
    name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.npy')[0].split('19-')[-1]
    print(f'mini_batch = {idx_mini_batch} {name_prefix}')
    img_lesion = img*mask
    # if idx==6: break #OMM to get a large mask

    # %%
    ## mini_batch = next(iter(train_loader))
    # print(mini_batch.keys())
    # print(len(mini_batch))
    # print(np.shape(mini_batch['image'][0]))
    # print(name_prefix)


    #%%
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(img[0])
    # ax[0].imshow(mask[0],alpha=.3)
    # ax[1].imshow(img_lesion[0])

    #%%
    # First use of SLIC, if the lesion is small only need to use SLIC once
    # numSegments = 300 # run slic with large segments to eliminate background & vessels
    SCALAR_LIMIT_CLUSTER_SIZE = 200 #340
    numSegments = np.max([np.sum(mask[0]>0)//SCALAR_LIMIT_CLUSTER_SIZE, 1]) # run slic with large segments to eliminate background & vessels
    TRESH_BACK = 0.10 #orig=0.15
    THRES_VESSEL = 0.7 #orig=.5
    if numSegments>1: # if mask is large then superpixels
        SCALAR_SIZE2 = 300
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
    # save vars for fig_slic
    
    background_plot = background;  lesion_area_plot = lesion_area
    vessels_plot = vessels; boundaries_plot = boundaries
    
    labelled, nr = label(mask_slic)
    mask_dil = remove_small_holes(remove_small_objects(mask_slic, 50))
    labelled2, nr2 = label(mask_dil)
    
    # %%
    # fig, ax = plt.subplots(1,4, figsize=(16,6))
    # ax[0].imshow((background>0)*img[0], vmax=1)
    # ax[1].imshow((vessels>0)*img[0], vmax=1)
    # ax[2].imshow((lesion_area>0)*img[0], vmax=1)
    # boundaries = mark_boundaries((lesion_area>0)*img[0], segments)[...,0]
    # ax[3].imshow(boundaries*(lesion_area>0), vmax=1)
    # for axx in ax.ravel(): axx.axis('off')
    # fig.tight_layout()

    # %%
    # lesion_area_only = (lesion_area>0)*img[0]
    # plt.imshow(lesion_area_only)
    # np.shape(lesion_area_only)

    #%%
    # IF LESION is too large us SLICMASK
    # lesion_area_only = (lesion_area>0)*img[0]
    # SCALAR_LIMIT_CLUSTER_SIZE = 340 # to make clusters size approx 40x40
    # y_min, y_max, x_min, x_max = coords_min_max_2D(lesion_area_only>0)
    # if np.sum(lesion_area_only>0) > SCALAR_LIMIT_CLUSTER_SIZE * 1 and (y_max-y_min) > 35 and (x_max-x_min) > 35:
    #     numSegments = np.max([np.sum(lesion_area_only>0)//SCALAR_LIMIT_CLUSTER_SIZE,2]) # run slic with large segments to eliminate background & vessels
    #     segments = slic((img[0]).astype('double'), n_segments = numSegments, mask=lesion_area_only>0, sigma = .2, multichannel=False, compactness=.1)
    #     background, lesion_area, vessels = superpixels((img[0]).astype('double'), segments, background_threshold=TRESH_BACK, vessel_threshold=THRES_VESSEL)
    # boundaries = mark_boundaries(lesion_area_only, segments)[...,0]
    # plt.imshow(boundaries)

    coords_big = name_prefix.split('_')
    coords_big = [int(i) for i in coords_big[1:]]
    TRESH_P=10
    fig_slic, ax = plt.subplots(3,3, figsize=(12,12))
    ax[0,0].imshow(scan[...,coords_big[-1]])
    ax[0,0].imshow(scan_mask[...,coords_big[-1]], alpha=.3)
    ax[0,1].imshow(scan[coords_big[0]-TRESH_P:coords_big[1]+TRESH_P,coords_big[2]-TRESH_P:coords_big[3]+TRESH_P,coords_big[-1]])
    ax[0,1].imshow(scan_mask[coords_big[0]-TRESH_P:coords_big[1]+TRESH_P,coords_big[2]-TRESH_P:coords_big[3]+TRESH_P,coords_big[-1]], alpha=.3)
    ax[0,2].imshow(img[0])
    ax[0,2].imshow(mask[0],alpha=.3)
    ax[1,0].imshow((background_plot>0)*img[0], vmax=1)
    ax[1,0].text(5,5,'bckg',c='r', fontsize=12)
    ax[1,1].imshow((vessels_plot>0)*img[0], vmax=1)
    ax[1,1].text(5,5,'vessel',c='r', fontsize=12)
    ax[1,2].imshow(mask_slic*img[0], vmax=1)
    ax[1,2].text(5,10,f'lesion\nnSegm={numSegments}',c='r', fontsize=12)
    ax[2,0].imshow(boundaries_plot*mask_slic, vmax=1)
    ax[2,0].text(5,5,f'seg={len(np.unique(segments))}',c='r', fontsize=12)
    ax[2,1].imshow(segments)
    # ax[2,1].text(5,5,nr2,c='r', fontsize=12)
    # ax[2,2].imshow(boundaries, vmax=1)
    # ax[2,2].text(5,5,len(np.unique(segments)),c='r', fontsize=12)
    # for axx in ax.ravel(): axx.axis('off')
    fig_slic.tight_layout()

    #%%
    tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(lesion_area, img[0], skip_index=0)
    SEED_VALUE = .19
    targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=SEED_VALUE, seed_method='max')

    # %%
    # for (a,b,c,d) in zip(targets, coords, masks, seeds):
    #     print(np.shape(a), b, np.shape(c), np.shape(d))
    # %%
    # rows=3; cols=3
    # fig, ax = plt.subplots(3,3,figsize=(rows*4,cols*4))
    # for idx_fig, (i,j) in enumerate(zip(targets,seeds)):
    # if idx_fig == rows*cols: break
    #     ax.flat[idx_fig].imshow(i[...,0])
    #     ax.flat[idx_fig].imshow(j, alpha=.3)
    #     ax.flat[idx_fig].text(3,3,idx_fig,c='r')
    # fig.tight_layout()

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
    path_save_synthesis_parent = f'{path_parent}_grow={GROW_ON_K_ITER}_bg={BACKGROUND_INTENSITY:.02f}_step={STEP_SIZE}_scale_mask={SCALE_MASK}_seed_value={SEED_VALUE}/'
    path_save_synthesis = f'{path_save_synthesis_parent}{SCAN_NAME}'
    Path(path_save_synthesis).mkdir(parents=True, exist_ok=True)#OMM
    fig_slic.savefig(f'{path_save_synthesis}{name_prefix}_slic.png')
    plt.close()

    continue
    #%%
    for idx_lesion, (target, coord, mask, this_seed) in tqdm(enumerate(zip(targets, coords, masks, seeds)),total=len(targets)):
        #   if idx_lesion<1:continue #OMM
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
        for i in tqdm(range(epochs), total=epochs, leave=False):
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
                model_str_final = plot_loss_and_lesion_synthesis(losses, optimizer, model_str, i, loss, sample_size, out, no_plot=True)

        stop = time(); time_total = f'{(stop-start)/60:.1f} mins'; print(time_total)
        model_str_final = model_str_final + f'\nep={i}, {time_total}' # for reconstruction figure
        #save model
        torch.save(model.model.state_dict(), f'{path_save_synthesis}{name_prefix}_weights_{idx_lesion:02d}.pt')

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
        print(f'idx_lesion done = {idx_lesion}')
        
        np.savez_compressed(f'{path_save_synthesis}{name_prefix}_lesion_{idx_lesion:02d}.npz', outs_float)
        np.save(f'{path_save_synthesis}{name_prefix}_coords_{idx_lesion:02d}.npy', coord)
        np.savez_compressed(f'{path_save_synthesis}{name_prefix}_mask_{idx_lesion:02d}.npz', mask)
        np.save(f'{path_save_synthesis}{name_prefix}_loss_{idx_lesion:02d}.npy', losses)
        np.save(f'{path_save_synthesis}{name_prefix}_time_{idx_lesion:02d}_{time_total}.npy', time_total)


        # %%
