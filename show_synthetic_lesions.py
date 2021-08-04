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
import argparse
import glob
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy.ndimage import label
import imageio
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
from utils import (
    superpixels,
    make_list_of_targets_and_seeds,
    fig_superpixels_only_lesions,
    select_lesions_match_conditions2,
)
from utils_replace_lesions import (
    fig_blend_lesion,
)
#%% FUNCTIONS
def get_xforms_scans_or_synthetic_lesions(mode="scans", keys=("image", "label")):
    """returns a composed transform for scans or synthetic lesions."""
    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
    ]
    dtype = (np.int16, np.uint8)
    if mode == "synthetic":
        xforms.extend([
          ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        ])
        dtype = (np.float32, np.uint8)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)

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

#%% ARGUMENTS
# parser = argparse.ArgumentParser()
# parser.add_argument('--SCAN_NAME', nargs='?', type=str, const='volume-covid19-A-0014', default='volume-covid19-A-0014')
# args = parser.parse_args()
# SCAN_NAME = args.SCAN_NAME

#%% LOAD SYNTHETIC LESIONS AND ORIGINAL SCANS

SCAN_NAME = 'volume-covid19-A-0014'
SLICE= 34 #int(names_all[0].split('_lesion')[0].split('_')[-1]) #slices_lesion[len(slices_lesion)//2] # pick one slice with a lesion
SKIP_LESIONS = 0
ONLY_ONE_SLICE = 34
GROW_ON_K_ITER = 1
BACKGROUND_INTENSITY = 0.11
STEP_SIZE = 1
SCALE_MASK = 0.19
SEED_VALUE = 0.19
CH0_1 = 1
CH1_16 = 15
ALIVE_THRESH = 0.1
ITER_INNER = 60
data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20_v2/Train'
images= [f'{data_folder}/{SCAN_NAME}_ct.nii.gz']
labels= [f'{data_folder}/{SCAN_NAME}_seg.nii.gz']
keys = ("image", "label")
files_scans = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]
batch_size = 1

# LOAD SYNTHETIC LESIONS
transforms_load = get_xforms_scans_or_synthetic_lesions("synthetic", keys)
ds_synthetic = monai.data.CacheDataset(data=files_scans, transform=transforms_load)
loader_synthetic = monai.data.DataLoader(
        ds_synthetic,
        batch_size=batch_size,
        shuffle=False, #should be true for training
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
for idx_mini_batch, mini_batch in enumerate(loader_synthetic):
    # if idx_mini_batch==6:break #OMM
    BATCH_IDX=0
    scan_synthetic = mini_batch['image'][BATCH_IDX][0,...].numpy()
    scan_mask = mini_batch['label'][BATCH_IDX][0,...].numpy()
    name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('Train/')[-1].split('.nii')[0]
    sum_TEMP_DELETE = np.sum(scan_mask)
    print(name_prefix)

# LOAD SCANS
transforms_load = get_xforms_scans_or_synthetic_lesions("scans", keys)
ds_scans = monai.data.CacheDataset(data=files_scans, transform=transforms_load)
loader_scans = monai.data.DataLoader(
        ds_scans,
        batch_size=batch_size,
        shuffle=False, #should be true for training
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

for idx_mini_batch, mini_batch in enumerate(loader_scans):
    # if idx_mini_batch==1:break #OMM
    BATCH_IDX=0
    scan = mini_batch['image'][BATCH_IDX][0,...]
    scan_mask = mini_batch['label'][BATCH_IDX][0,...]
    scan_name = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii')[0][:-3]
print(f'working on scan= {scan_name}')
assert scan_name == SCAN_NAME, 'cannot load that scan'
scan = scan.numpy()   #ONLY READ ONE SCAN (WITH PREVIOUS BREAK)
scan_mask = scan_mask.numpy()

# LOAD INDIVIDUAL LESIONS
folder_source = f'/content/drive/MyDrive/Datasets/covid19/COVID-19-20/individual_lesions/{SCAN_NAME}_ct/'
files_scan = sorted(glob.glob(os.path.join(folder_source,"*.npy")))
files_mask = sorted(glob.glob(os.path.join(folder_source,"*.npz")))
keys = ("image", "label")
files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(files_scan, files_mask)]
print(len(files_scan), len(files_mask), len(files))
transforms_load = get_xforms_load("load", keys)
ds_lesions = monai.data.CacheDataset(data=files, transform=transforms_load)
loader_lesions = monai.data.DataLoader(
        ds_lesions,
        batch_size=batch_size,
        shuffle=False, #should be true for training
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

# LOAD SYTHETIC INPAINTDE PSEUDO-HEALTHY TEXTURE
path_synthesis_old = '/content/drive/My Drive/Datasets/covid19/results/cea_synthesis/patient0/'
texture_orig = np.load(f'{path_synthesis_old}texture.npy.npz')
texture_orig = texture_orig.f.arr_0
texture = texture_orig + np.abs(np.min(texture_orig))# + .07

#%% SUPERPIXELS
mask_sizes=[]
cluster_sizes = []
targets_all = []
flag_only_one_slice = False
for idx_mini_batch,mini_batch in enumerate(loader_lesions):
    if idx_mini_batch < SKIP_LESIONS:continue #resume incomplete reconstructions

    img = mini_batch['image'].numpy()
    mask = mini_batch['label'].numpy()
    mask = remove_small_objects(mask, 20)
    mask_sizes.append([idx_mini_batch, np.sum(mask)])
    name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.npy')[0].split('19-')[-1]
    img_lesion = img*mask

    # if 2nd argument is provided then only analyze that slice
    if ONLY_ONE_SLICE != -1: 
        slice_used = int(name_prefix.split('_')[-1])
        if slice_used != int(ONLY_ONE_SLICE): continue
        else: flag_only_one_slice = True

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
    segments_sizes = [np.sum(segments==i_segments) for i_segments in np.unique(segments)[1:]]
    cluster_sizes.append(segments_sizes)
    segments_sizes = [str(f'{i_segments}') for i_segments in segments_sizes]
    segments_sizes = '\n'.join(segments_sizes)

    # save vars for fig_slic
    background_plot = background;  lesion_area_plot = lesion_area
    vessels_plot = vessels; boundaries_plot = boundaries
    labelled, nr = label(mask_slic)
    mask_dil = remove_small_holes(remove_small_objects(mask_slic, 50))
    labelled2, nr2 = label(mask_dil)

    tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(segments, img[0], skip_index=0)
    targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=SEED_VALUE, seed_method='max')
    targets_all.append(len(targets))

    coords_big = name_prefix.split('_')
    coords_big = [int(i) for i in coords_big[1:]]
    TRESH_PLOT=20
    device = 'cuda'
    num_channels = 16
    epochs = 2500
    sample_size = 8
    path_parent = 'interactive/'
    path_fig = f'{path_parent}CA={GROW_ON_K_ITER}_bg={BACKGROUND_INTENSITY:.02f}_step={STEP_SIZE}_scale={SCALE_MASK}_seed={SEED_VALUE}_ch0_1={CH0_1}_ch1_16={CH1_16}_ali_thr={ALIVE_THRESH}_iter={ITER_INNER}/'
    path_fig = f'{path_fig}{SCAN_NAME}/'
    Path(path_fig).mkdir(parents=True, exist_ok=True)
    fig_superpixels_only_lesions(path_fig, name_prefix, scan, scan_mask, img, mask_slic, boundaries_plot, segments, segments_sizes, coords_big, TRESH_PLOT, idx_mini_batch, numSegments)
    if flag_only_one_slice: break

#%% MAIN REPLACE LESIONS
# READ ORIGINAL SCAN
mask_outer_ring = False # slow
POST_PROCESS = True 
blur_lesion = False
# SCAN_NAME = name_prefix.split('_ct')[0]
scan_slice = scan_synthetic[...,SLICE]
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


V_MAX = np.max(scan_slice)
slice_healthy_inpain = pseudo_healthy_with_texture(scan_slice, lesions_all, coords_all, masks_all, names_all, texture)
slice_healthy_inpain2 = copy(slice_healthy_inpain)
decreasing_sequence = get_decreasing_sequence(128, splits= 14)
decreasing_sequence = get_decreasing_sequence(255, splits= 20) #MORE IMAGES
images=[]
arrays_sequence = []; 
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
    coords_big2 = [int(i) for i in name.split('_')[1:5]]
    coords_sums = coord + coords_big2
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
  arrays_sequence.append(slice_healthy_inpain2[coords_big2[0]-TRESH_PLOT:coords_big2[1]+TRESH_PLOT,coords_big2[2]-TRESH_PLOT:coords_big2[3]+TRESH_PLOT])
  
  images.append(slice_healthy_inpain2[coords_big2[0]-TRESH_PLOT:coords_big2[1]+TRESH_PLOT,coords_big2[2]-TRESH_PLOT:coords_big2[3]+TRESH_PLOT])
#   path_fig_growth = f"{path_synthesis}/image.png"
#   fig_blend_lesion(slice_healthy_inpain2, coords_big2, GEN, decreasing_sequence, path_synthesis, path_fig_growth, Tp=TRESH_PLOT, V_MAX=1, close=False, plot_size=6)
#   images.append(imageio.imread(path_fig_growth))

#%% PLOT RESULTS
i = 18 
fig, ax = plt.subplots(1,3,figsize=(12,6))
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
np.shape(images)
#%%
plt.imshow(images[4])
# %%

# %%
fig_superpixels_only_lesions(path_fig, name_prefix, scan, scan_mask, img, mask_slic, boundaries_plot, segments, segments_sizes, coords_big, TRESH_PLOT, idx_mini_batch, numSegments)
# %%
plt.hist(scan.flatten());
# %%
from utils import fig_superpixels_only_lesions
# %%
# %%
np.shape(scan), np.shape(scan_mask)

#%%
#=====================================
# SEAM CARVING ALGORITHM
# https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
from utils import coords_min_max_2D
from scipy.ndimage.filters import convolve
# %%
idx_seam = 34
plt.imshow(scan[...,idx_seam], vmin=-1000, vmax=500)
plt.imshow(scan_mask[...,idx_seam])
# %%
y_min, y_max, x_min, x_max = coords_min_max_2D(scan_mask[...,idx_seam])
img_seam = scan[...,idx_seam][y_min: y_max, x_min: x_max]
plt.imshow(img_seam)
# %%
def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    # filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map
# %%
energy_map = calc_energy(np.expand_dims(img_seam,-1))
np.shape(energy_map)
# %%
