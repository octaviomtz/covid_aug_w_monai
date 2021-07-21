#%% IMPORTS
import os
import numpy as np
import monai
import torch
from copy import copy
from tqdm import tqdm
from skimage.restoration import inpaint
from pathlib import Path
import matplotlib.pyplot as plt
from utils_replace_lesions import (
    read_cea_aug_slice2,
    pseudo_healthy_with_texture,
    get_decreasing_sequence,
    normalize_new_range4,
    get_orig_scan_in_lesion_coords,
    blur_masked_image,
    make_mask_ring
)
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandFlipd,
    RandFlipd,
    RandFlipd,
    CastToTyped,
)
#%% FUNCTIONS
def get_xforms_load(mode="load", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend([
                  SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                  RandAffined(
                      keys,
                      prob=0.15,
                      rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                      scale_range=(0.1, 0.1, None), 
                      mode=("bilinear", "nearest"),
                      as_tensor_output=False,
                  ),
                  RandCropByPosNegLabeld(keys, label_key=keys[1], 
                  spatial_size=(192, 192, 16), num_samples=3),
                  RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                  RandFlipd(keys, spatial_axis=0, prob=0.5),
                  RandFlipd(keys, spatial_axis=1, prob=0.5),
                  RandFlipd(keys, spatial_axis=2, prob=0.5),
              ])
    dtype = (np.float32, np.uint8)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)

#%% LOAD ORIGINAL SCANS
# data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20/Train'
# # folder_dest = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20/individual_lesions/'
# images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))[:10] #OMM
# labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))[:10] #OMM
# # =====
# keys = ("image", "label")
# train_frac, val_frac = 0.8, 0.2
# n_train = int(train_frac * len(images)) + 1
# n_val = min(len(images) - n_train, int(val_frac * len(images)))
# # =====
# train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
# val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]
# print(f'train_files={len(train_files)}, val_files={len(val_files)}')

SCAN_NAME = 'volume-covid19-A-0014'
data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20_v2/Train'
images= [f'{data_folder}/{SCAN_NAME}_ct.nii.gz']
labels= [f'{data_folder}/{SCAN_NAME}_seg.nii.gz']
keys = ("image", "label")
files_scans = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]

batch_size = 1
transforms_load = get_xforms_load("load", keys)
train_ds = monai.data.CacheDataset(data=files_scans, transform=transforms_load)
train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False, #should be true for training
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

# main loop
for idx_mini_batch, mini_batch in enumerate(train_loader):
    # if idx_mini_batch==6:break #OMM
    BATCH_IDX=0
    scan = mini_batch['image'][BATCH_IDX][0,...].numpy()
    scan_mask = mini_batch['label'][BATCH_IDX][0,...].numpy()
    name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('Train/')[-1].split('.nii')[0]
    sum_TEMP_DELETE = np.sum(scan_mask)
    print(name_prefix, sum_TEMP_DELETE)
    if 'A-0014' in name_prefix:
      break #OMM

# READ THE SYTHETIC HEALTHY TEXTURE
path_synthesis_old = '/content/drive/My Drive/Datasets/covid19/results/cea_synthesis/patient0/'
texture_orig = np.load(f'{path_synthesis_old}texture.npy.npz')
texture_orig = texture_orig.f.arr_0
texture = texture_orig + np.abs(np.min(texture_orig))# + .07

#%% READ THE FILES FROM THE CORRESPONDING SCAN AND GET THE SLICE CHOSEN
# READ ORIGINAL SCAN
mask_outer_ring = False # slow
POST_PROCESS = True 
blur_lesion = False
SLICE= 34 #int(names_all[0].split('_lesion')[0].split('_')[-1]) #slices_lesion[len(slices_lesion)//2] # pick one slice with a lesion
SCAN_NAME = name_prefix.split('_ct')[0]
scan_slice = scan[...,SLICE]
# READ LESION SYNTHESIS
path_parent = '/content/drive/My Drive/Datasets/covid19/COVID-19-20_augs_cea/'
path_synthesis_ = f'{path_parent}CeA_BASE_grow=1_bg=-1.00_step=-1.0_scale=-1.0_seed=1.0_ch0_1=-1_ch1_16=-1_ali_thr=0.1/'
path_synthesis = f'{path_synthesis_}{SCAN_NAME}/'
files_from_scan = os.listdir(path_synthesis)
slices_lesion = [int(i.split('_coords')[0].split('_')[-1]) for i in files_from_scan if 'coords' in i]
slices_lesion = np.unique(slices_lesion)
lesions_all, coords_all, masks_all, names_all, loss_all = read_cea_aug_slice2(path_synthesis, SLICE=SLICE)
# add the synthesized lesions onto the texture inpainted image
Path('images').mkdir(parents=True, exist_ok=True) 
file_path = "images/image.png"

V_MAX = np.max(scan_slice)
slice_healthy_inpain = pseudo_healthy_with_texture(scan_slice, lesions_all, coords_all, masks_all, names_all, texture)
slice_healthy_inpain2 = copy(slice_healthy_inpain)
decreasing_sequence = get_decreasing_sequence(128, splits= 14)
decreasing_sequence = get_decreasing_sequence(255, splits= 20) #MORE IMAGES
images=[]
arrays_sequence = []; Tp=10
synthetic_intensities_together = []
syn_norms_TEMP = []
mse_gen = []
for GEN in tqdm(decreasing_sequence):
  
  slice_healthy_inpain2 = copy(slice_healthy_inpain)
  synthetic_intensities=[]
  mask_for_inpain = np.zeros_like(slice_healthy_inpain2)
  syn_norms=[]
  mse_lesions = []
  for idx_x, (lesion, coord, mask, name) in enumerate(zip(lesions_all, coords_all, masks_all, names_all)):
    #get the right coordinates
    coords_big = [int(i) for i in name.split('_')[1:5]]
    coords_sums = coord + coords_big
    new_coords_mask = np.where(mask==1)[0]+coords_sums[0], np.where(mask==1)[1]+coords_sums[2]
    # syn_norm = lesion[GEN] *x_seq2[idx_x]
    if GEN<60:
      if POST_PROCESS:
        syn_norm = normalize_new_range4(lesion[GEN], scan_slice[new_coords_mask], scale=.5)#, log_seq_norm2[idx_x])#, 0.19)
        # syn_norm = lesion[GEN]
      else:
        syn_norm = lesion[GEN]
    else:
      syn_norm = lesion[GEN]
    # if GEN==10:break
    
    # get the MSE between synthetic and original
    orig_lesion = get_orig_scan_in_lesion_coords(scan_slice, new_coords_mask)
    mse_lesions.append(np.mean(mask*(syn_norm - orig_lesion)**2))

    syn_norm = syn_norm * mask  
    syn_norms.append(syn_norm)
    # add background texture with absolute coordinates
    
    if blur_lesion:
      syn_norm = blur_masked_image(syn_norm, kernel_blur=(2,2))
    # add cea syn with absolute coords
    new_coords = np.where(syn_norm>0)[0]+coords_sums[0], np.where(syn_norm>0)[1]+coords_sums[2]
    slice_healthy_inpain2[new_coords] = syn_norm[syn_norm>0]
    
    synthetic_intensities.extend(syn_norm[syn_norm>0])
    
    # inpaint the outer ring
    if mask_outer_ring:
      mask_ring = make_mask_ring(syn_norm>0)
      new_coords_mask_inpain = np.where(mask_ring==1)[0]+coords_sums[0], np.where(mask_ring==1)[1]+coords_sums[2] # mask outer rings for inpaint
      mask_for_inpain[new_coords_mask_inpain] = 1
  
  mse_gen.append(mse_lesions)
  syn_norms_TEMP.append(syn_norms)
  if mask_outer_ring:
    slice_healthy_inpain2 = inpaint.inpaint_biharmonic(slice_healthy_inpain2, mask_for_inpain)

  synthetic_intensities_together.append(synthetic_intensities)
  arrays_sequence.append(slice_healthy_inpain2[coords_big[0]-Tp:coords_big[1]+Tp,coords_big[2]-Tp:coords_big[3]+Tp])

#%% PLOT RESULTS
i = 18 
fig, ax = plt.subplots(1,3,figsize=(12,6));
ax[0].imshow(arrays_sequence[i-1], vmin=0, vmax=1)
ax[0].text(2,4,decreasing_sequence[i-1],c='r', fontsize=16)
ax[1].imshow(arrays_sequence[i], vmin=0, vmax=1)
ax[1].text(2,4,decreasing_sequence[i],c='r', fontsize=16)
ax[2].imshow(arrays_sequence[i+1], vmin=0, vmax=1)
ax[2].text(2,4,decreasing_sequence[i+1],c='r', fontsize=16)
fig.tight_layout()

plt.figure(figsize=(15,2))
mse_gen2 = np.asarray(mse_gen).T
np.shape(mse_gen2)
for i in mse_gen2:
  plt.semilogy(decreasing_sequence,i, c='k', alpha=.5)
plt.grid('on')
# %%
