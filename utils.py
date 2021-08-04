import numpy as np
import math
from scipy.ndimage import binary_fill_holes, distance_transform_bf
from matplotlib import pyplot as plt 

def coords_min_max_2D(array):
  '''return the min and max+1 of a mask. We use mask+1 to include the whole mask'''
  yy, xx = np.where(array==True)
  y_max = np.max(yy)+1; y_min = np.min(yy)
  x_max = np.max(xx)+1; x_min = np.min(xx)
  return y_min, y_max, x_min, x_max

def superpixels(im2, segments, background_threshold=.2, vessel_threshold=.4):
  '''1) segment all image using superpixels. 
  2) Then, classify each superpixel into background, vessel or lession according
  to its median intensity'''
  background = np.zeros_like(im2)
  vessels = np.zeros_like(im2)
  lesion_area = np.zeros_like(im2)
  label_background, label_vessels, label_lession = 1, 1, 1,
  for (i, segVal) in enumerate(np.unique(segments)):
    mask = np.zeros_like(im2)
    mask[segments == segVal] = 1
    clus = im2*mask
    median_intensity = np.median(clus[clus>0])
    yy,xx = np.where(mask==1)
    if median_intensity < background_threshold or math.isnan(median_intensity):
      background[yy,xx] = label_background
      label_background += 1
    elif median_intensity > vessel_threshold:
      vessels[yy,xx] = label_vessels
      label_vessels += 1
    else:
      lesion_area[yy,xx] = label_lession
      label_lession += 1
  return background, lesion_area, vessels

def select_lesions_match_conditions2(small_lesions, img, skip_index=1, max_size=np.inf):
  '''
  small_lesions: mask with every element selected by superpixels
  img: original slice corresponding to the mask small_lesions 
  For each element of the mask 'small_lesions' AND IF
  the area of the element is smaller than max_size
  then return save the corresponding slice area in target_minis.
  And its mask and its coords in target_minis_coords, target_minis_masks
  '''
  target_minis = []
  target_minis_coords = []
  target_minis_masks = []
  target_minis_big = []
  target_minis_coords_big = []
  target_minis_masks_big = []
  for i in np.unique(small_lesions):
    mm = small_lesions==i
    y_min, y_max, x_min, x_max = coords_min_max_2D(mm)
    mask_mini = (small_lesions==i)[y_min:y_max,x_min:x_max]
    target_mini = img[y_min:y_max,x_min:x_max]
    if i > skip_index:
      if np.sum(mm) < max_size:
        target_minis.append(mask_mini*target_mini)
        target_minis_coords.append((y_min, y_max, x_min, x_max))
        target_minis_masks.append(mask_mini)
      else:
        target_minis_big.append(mask_mini*target_mini)
        target_minis_coords_big.append((y_min, y_max, x_min, x_max))
        target_minis_masks_big.append(mask_mini)
  return target_minis, target_minis_coords, target_minis_masks, target_minis_big, target_minis_coords_big, target_minis_masks_big 

#export
def make_list_of_targets_and_seeds(tgt_small, tgt_coords_small, tgt_masks_small, init_lists=True, seed_value=1, targets=[], seeds=[], masks=[], coords=[], seed_method='center'):
  '''if no list is sent create lists of targets and their seeds, 
  if a list is sent, append the new values '''
  if init_lists: 
    targets = []
    seeds = []
    masks = []
    coords = []
  for i_tgt, i_coords, i_mask in zip(tgt_small, tgt_coords_small, tgt_masks_small):
    target_temp = np.zeros((np.shape(i_tgt)[0],np.shape(i_tgt)[1],2))
    target_temp[...,0] = i_tgt
    mask_mini_closed = binary_fill_holes(i_tgt>0)
    target_temp[...,1] = mask_mini_closed
    targets.append((target_temp).astype('float32'))
    if seed_method=='center': 
      #set the seed in the center of the mask
      mask_mini_dt = distance_transform_bf(mask_mini_closed)
      seed = mask_mini_dt==np.max(mask_mini_dt)
      if np.sum(seed)>1: 
        yy, xx = np.where(seed==1)
        seed = np.zeros_like(mask_mini_dt)
        seed[yy[0], xx[0]] = seed_value
    else:
      #set the seed in the pixel with largest intensity
      yy, xx = np.where(i_tgt==np.max(i_tgt))
      seed = np.zeros_like(i_tgt)
      seed[yy, xx] = seed_value
      
    seeds.append(seed)
    coords.append(i_coords)
    masks.append(i_mask)
  return targets, coords, masks, seeds

def fig_superpixels_ICCV(path_synthesis_figs, name_prefix, scan, scan_mask, img, background_plot, vessels_plot,
mask_slic, boundaries_plot, segments, segments_sizes, lesion_area,
targets, coords_big, TRESH_P, idx_mini_batch, numSegments):
    '''version used in early experiments'''
    fig_slic, ax = plt.subplots(3,3, figsize=(12,12))
    ax[0,0].imshow(scan[...,coords_big[-1]])
    ax[0,0].text(25,25,idx_mini_batch,c='r', fontsize=12)
    ax[0,0].imshow(scan_mask[...,coords_big[-1]], alpha=.3)
    ax[0,1].imshow(scan[coords_big[0]-TRESH_P:coords_big[1]+TRESH_P,coords_big[2]-TRESH_P:coords_big[3]+TRESH_P,coords_big[-1]])
    ax[0,1].imshow(scan_mask[coords_big[0]-TRESH_P:coords_big[1]+TRESH_P,coords_big[2]-TRESH_P:coords_big[3]+TRESH_P,coords_big[-1]], alpha=.3)
    ax[0,2].imshow(img[0])
    # ax[0,2].imshow(mask[0],alpha=.3)
    ax[1,0].imshow((background_plot>0)*img[0], vmax=1)
    ax[1,0].text(5,5,'bckg',c='r', fontsize=12)
    ax[1,1].imshow((vessels_plot>0)*img[0], vmax=1)
    ax[1,1].text(5,5,'vessel',c='r', fontsize=12)
    ax[1,2].imshow(mask_slic*img[0], vmax=1)
    ax[1,2].text(5,10,f'lesion\nnSegm={numSegments}',c='r', fontsize=12)
    ax[2,0].imshow(boundaries_plot*mask_slic, vmax=1)
    ax[2,0].text(5,5,f'seg={len(np.unique(segments))}',c='r', fontsize=12)
    ax[2,1].imshow(segments)
    ax[2,1].text(5,np.shape(segments)[0]//2,segments_sizes,c='r', fontsize=12)
    ax[2,2].imshow(lesion_area, vmax=1)
    ax[2,2].text(5,np.shape(mask_slic)[0]//2,f'targets={len(targets)}',c='r', fontsize=12)
    # for axx in ax.ravel(): axx.axis('off')
    fig_slic.tight_layout()
    fig_slic.savefig(f'{path_synthesis_figs}{name_prefix}_slic.png') 

def fig_superpixels_only_lesions(path_synthesis_figs, name_prefix, scan, scan_mask, img, mask_slic, boundaries_plot, segments, segments_sizes, coords_big, TRESH_P, idx_mini_batch, numSegments):
    '''plot 2 rows'''
    fig_slic, ax = plt.subplots(2,3, figsize=(12,8))
    ax[0,0].imshow(scan[...,coords_big[-1]], vmin=-1000, vmax=500)
    ax[0,0].text(25,25,idx_mini_batch,c='r', fontsize=12)
    ax[0,0].imshow(scan_mask[...,coords_big[-1]], alpha=.3)
    ax[0,1].imshow(scan[coords_big[0]-TRESH_P:coords_big[1]+TRESH_P,coords_big[2]-TRESH_P:coords_big[3]+TRESH_P,coords_big[-1]])
    ax[0,1].imshow(scan_mask[coords_big[0]-TRESH_P:coords_big[1]+TRESH_P,coords_big[2]-TRESH_P:coords_big[3]+TRESH_P,coords_big[-1]], alpha=.3)
    ax[0,2].imshow(img[0])
    # ax[0,2].imshow(mask[0],alpha=.3)
    ax[1,0].imshow(boundaries_plot*mask_slic, vmax=1)
    ax[1,0].text(5,5,f'seg={len(np.unique(segments))}',c='r', fontsize=12)
    ax[1,1].imshow(segments)
    ax[1,1].text(5,np.shape(segments)[0]//2,segments_sizes,c='r', fontsize=12)
    ax[1,2].imshow(mask_slic*img[0], vmax=1)
    ax[1,2].text(5,10,f'lesion\nnSegm={numSegments}',c='r', fontsize=12)
    fig_slic.tight_layout()
    fig_slic.savefig(f'{path_synthesis_figs}{name_prefix}_slic.png')    