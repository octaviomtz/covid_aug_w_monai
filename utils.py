import numpy as np
import math
from scipy.ndimage import binary_fill_holes

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
  '''if no list is sent create lists of targets and their seeds, if a list is sent, append the new values '''
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