# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#%%
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
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler, MetricsSaver
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
    CopyItemsd,
)

#%%
from monai.transforms.transform import MapTransform, Randomizable
from monai.config import KeysCollection
from typing import Dict, Hashable, Mapping, Union
from utils_replace_lesions import get_decreasing_sequence, crop_and_pad 
from utils_replace_lesions import read_cea_aug_slice2, pseudo_healthy_with_texture, to_torch_right_shape, normalize_new_range4, get_orig_scan_in_lesion_coords, make_mask_ring
from scipy.ndimage import affine_transform
import imageio
import os
import moviepy.editor as mvp
from pathlib import Path
from copy import copy
from scipy.ndimage import binary_fill_holes
from skimage.restoration import inpaint
import cv2

# from monai.data.utils import pad_list_data_collate

#%% CLASSES ADDED 
# from typing import Sequence, Optional, List, Tuple
# from monai.utils import ensure_tuple_rep, fall_back_tuple
# from monai.transforms.utils import map_binary_to_indices, generate_pos_neg_label_crop_centers
# from copy import deepcopy
# from monai.transforms.croppad.array import SpatialCrop
# from monai.transforms.transform import RandomizableTransform, Transform
# from monai.utils.enums import InverseKeys
# from monai.utils import ImageMetaKey as Key
# class InvertibleTransform(Transform):
#     """
#     """

#     def push_transform(
#         self,
#         data: dict,
#         key: Hashable,
#         extra_info: Optional[dict] = None,
#         orig_size: Optional[Tuple] = None,
#     ) -> None:
#         """Append to list of applied transforms for that key."""
#         key_transform = str(key) + InverseKeys.KEY_SUFFIX
#         info = {
#             InverseKeys.CLASS_NAME: self.__class__.__name__,
#             InverseKeys.ID: id(self),
#         }
#         if orig_size is not None:
#             info[InverseKeys.ORIG_SIZE] = orig_size
#         elif hasattr(data[key], "shape"):
#             info[InverseKeys.ORIG_SIZE] = data[key].shape[1:]
#         if extra_info is not None:
#             info[InverseKeys.EXTRA_INFO] = extra_info
#         # If class is randomizable transform, store whether the transform was actually performed (based on `prob`)
#         if isinstance(self, RandomizableTransform):
#             info[InverseKeys.DO_TRANSFORM] = self._do_transform
#         # If this is the first, create list
#         if key_transform not in data:
#             data[key_transform] = []
#         data[key_transform].append(info)


#     def check_transforms_match(self, transform: dict) -> None:
#         """Check transforms are of same instance."""
#         if transform[InverseKeys.ID] == id(self):
#             return
#         # basic check if multiprocessing uses 'spawn' (objects get recreated so don't have same ID)
#         if (
#             torch.multiprocessing.get_start_method(allow_none=False) == "spawn"
#             and transform[InverseKeys.CLASS_NAME] == self.__class__.__name__
#         ):
#             return
#         raise RuntimeError("Should inverse most recently applied invertible transform first")


#     def get_most_recent_transform(self, data: dict, key: Hashable) -> dict:
#         """Get most recent transform."""
#         transform = dict(data[str(key) + InverseKeys.KEY_SUFFIX][-1])
#         self.check_transforms_match(transform)
#         return transform


#     def pop_transform(self, data: dict, key: Hashable) -> None:
#         """Remove most recent transform."""
#         data[str(key) + InverseKeys.KEY_SUFFIX].pop()


#     def inverse(self, data: dict) -> Dict[Hashable, np.ndarray]:
#         """
#         Inverse of ``__call__``.

#         Raises:
#             NotImplementedError: When the subclass does not override this method.

#         """
#         raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

# class RandCropByPosNegLabeld(Randomizable, MapTransform, InvertibleTransform):
#     """
#     """

#     def __init__(
#         self,
#         keys: KeysCollection,
#         label_key: str,
#         spatial_size: Union[Sequence[int], int],
#         pos: float = 1.0,
#         neg: float = 1.0,
#         num_samples: int = 1,
#         image_key: Optional[str] = None,
#         image_threshold: float = 0.0,
#         fg_indices_key: Optional[str] = None,
#         bg_indices_key: Optional[str] = None,
#         meta_keys: Optional[KeysCollection] = None,
#         meta_key_postfix: str = "meta_dict",
#         allow_missing_keys: bool = False,
#     ) -> None:
#         MapTransform.__init__(self, keys, allow_missing_keys)
#         self.label_key = label_key
#         self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
#         if pos < 0 or neg < 0:
#             raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
#         if pos + neg == 0:
#             raise ValueError("Incompatible values: pos=0 and neg=0.")
#         self.pos_ratio = pos / (pos + neg)
#         self.num_samples = num_samples
#         self.image_key = image_key
#         self.image_threshold = image_threshold
#         self.fg_indices_key = fg_indices_key
#         self.bg_indices_key = bg_indices_key
#         self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
#         if len(self.keys) != len(self.meta_keys):
#             raise ValueError("meta_keys should have the same length as keys.")
#         self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
#         self.centers: Optional[List[List[np.ndarray]]] = None

#     def randomize(
#         self,
#         label: np.ndarray,
#         fg_indices: Optional[np.ndarray] = None,
#         bg_indices: Optional[np.ndarray] = None,
#         image: Optional[np.ndarray] = None,
#     ) -> None:
#         self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
#         if fg_indices is None or bg_indices is None:
#             fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
#         else:
#             fg_indices_ = fg_indices
#             bg_indices_ = bg_indices
#         self.centers = generate_pos_neg_label_crop_centers(
#             self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_, bg_indices_, self.R
#         )


#     def __call__(self, data: Mapping[Hashable, np.ndarray]) -> List[Dict[Hashable, np.ndarray]]:
#         d = dict(data)
#         label = d[self.label_key]
#         image = d[self.image_key] if self.image_key else None
#         fg_indices = d.get(self.fg_indices_key) if self.fg_indices_key is not None else None
#         bg_indices = d.get(self.bg_indices_key) if self.bg_indices_key is not None else None

#         self.randomize(label, fg_indices, bg_indices, image)
#         if not isinstance(self.spatial_size, tuple):
#             raise AssertionError
#         if self.centers is None:
#             raise AssertionError

#         # initialize returned list with shallow copy to preserve key ordering
#         results: List[Dict[Hashable, np.ndarray]] = [dict(data) for _ in range(self.num_samples)]

#         for i, center in enumerate(self.centers):
#             # fill in the extra keys with unmodified data
#             for key in set(data.keys()).difference(set(self.keys)):
#                 results[i][key] = deepcopy(data[key])
#             for key in self.key_iterator(d):
#                 img = d[key]
#                 cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
#                 orig_size = img.shape[1:]
#                 results[i][key] = cropper(img)
#                 self.push_transform(results[i], key, extra_info={"center": center}, orig_size=orig_size)
#             # add `patch_index` to the meta data
#             for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
#                 meta_key = meta_key or f"{key}_{meta_key_postfix}"
#                 if meta_key not in results[i]:
#                     results[i][meta_key] = {}  # type: ignore
#                 results[i][meta_key][Key.PATCH_INDEX] = i

#         return results


#     def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
#         d = deepcopy(dict(data))
#         for key in self.key_iterator(d):
#             transform = self.get_most_recent_transform(d, key)
#             # Create inverse transform
#             orig_size = np.asarray(transform[InverseKeys.ORIG_SIZE])
#             current_size = np.asarray(d[key].shape[1:])
#             center = transform[InverseKeys.EXTRA_INFO]["center"]
#             cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
#             # get required pad to start and end
#             pad_to_start = np.array([s.indices(o)[0] for s, o in zip(cropper.slices, orig_size)])
#             pad_to_end = orig_size - current_size - pad_to_start
#             # interleave mins and maxes
#             pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
#             inverse_transform = BorderPad(pad)
#             # Apply inverse transform
#             d[key] = inverse_transform(d[key])
#             # Remove the applied transform
#             self.pop_transform(d, key)

#         return d

#==========================

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
            # d[key] = self.identity(d[key])
        return d
#%%
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
        print(f"scan={d['image'].shape, d.get('label_meta_dict').get('filename_or_obj').split('Train/')[-1].split('_seg')[0]}")

        print(f'KEYS={d.keys()}')
        # print(f"TRANS:\n{d.get('label_transforms')}")
        SCAN_NAME = d.get('label_meta_dict').get('filename_or_obj').split('Train/')[-1].split('_seg')[0] 
        SLICE = d.get('label_transforms')[3].get('extra_info').get('center')[-1]
        CENTER_Y = d.get('label_transforms')[3].get('extra_info').get('center')[0]
        CENTER_X = d.get('label_transforms')[3].get('extra_info').get('center')[1]
        path_synthesis2 = f'{str(self.path_synthesis)}/{SCAN_NAME}/'
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
        print(f'FLIPS = {flip0, flip1, flip2}')
        array_trans = d.get('synthetic_lesion')
        array_trans_lab = d.get('synthetic_label')
        aa = [np.shape(array_trans[0,...,i]) for i in range(16)]
        print(f"affine before =>{array_trans.shape}, {aa}")
        # array_trans = self.pad_because_monai_transf_is_not_doing_it(array_trans)
        array_trans = np.rot90(array_trans,1,axes=[1,2])
        array_trans_lab = np.rot90(array_trans_lab,1,axes=[1,2])
        print(f"affine after =>{array_trans.shape}")
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
            # temp_image = d['synthetic_lesion']
            # temp_label = d['synthetic_label']
            # d['synthetic_lesion'] = d['image']
            # d['synthetic_label'] = d['label']
            d['image'] = d['synthetic_lesion']
            d['label'] = d['synthetic_label']       
        return d

#%%
def get_xforms(mode="train", keys=("image", "label"), keys2=("image", "label", "synthesis"), path_synthesis='', decreasing_sequence='', scans_syns=[], texture=[]):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        CopyItemsd(keys,1, names=['image_1', 'label_1']),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                RandCropByPosNegLabeld(keys, label_key=keys[1], 
                spatial_size=(192, 192, 16), num_samples=3),
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                    scale_range=(0.1, 0.1, None), 
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "synthesis":
        print('DOING SYNTHESIS')
        xforms.extend(
            [       
                  PrintTypesShapes(keys, '======SHAPE LOAD'),
                  SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                  RandCropByPosNegLabeld(keys, label_key=keys[1], 
                  spatial_size=(192, 192, 16), num_samples=3), 
                  TransCustom(keys, path_synthesis, read_cea_aug_slice2, 
                              pseudo_healthy_with_texture, scans_syns, decreasing_sequence, GEN=15,
                              POST_PROCESS=True, mask_outer_ring=True, texture=np.empty(shape=(456,456)), new_value=.5),
                  RandAffined(
                      keys2,
                      prob=0.15,
                      rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                      scale_range=(0.1, 0.1, None), 
                      mode=("bilinear", "nearest", "bilinear"),
                    #   mode=("bilinear", "nearest"),
                      as_tensor_output=False
                  ),
                  
                  RandGaussianNoised((keys2[0],keys2[2]), prob=0.15, std=0.01),
                #   RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                  RandFlipd(keys, spatial_axis=0, prob=0.5),
                  RandFlipd(keys, spatial_axis=1, prob=0.5),
                  RandFlipd(keys, spatial_axis=2, prob=0.5),
                  TransCustom2(keys2),
                  SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),
                  PrintTypesShapes(keys, 'LAST'),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        PrintTypesShapes(keys, 'VAAAAAL'),
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return monai.transforms.Compose(xforms)


def get_net():
    """returns a unet model instance."""

    n_classes = 2
    net = monai.networks.nets.BasicUNet(
        dimensions=3,
        in_channels=1,
        out_channels=n_classes,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
    )
    return net


def get_inferer(_mode=None):
    """returns a sliding window inference instance."""

    patch_size = (192, 192, 16)
    sw_batch_size, overlap = 2, 0.5
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy


def train(data_folder=".", model_folder="runs", continue_training=False):
    """run a training pipeline."""

    #/== files for synthesis
    path_parent = Path('/content/drive/My Drive/Datasets/covid19/COVID-19-20_augs_cea/')
    path_synthesis = Path(path_parent / 'CeA_BASE_grow=1_bg=-1.00_step=-1.0_scale=-1.0_seed=1.0_ch0_1=-1_ch1_16=-1_ali_thr=0.1')
    scans_syns = os.listdir(path_synthesis)
    decreasing_sequence = get_decreasing_sequence(255, splits= 20)
    keys2=("image", "label", "synthetic_lesion")
    # READ THE SYTHETIC HEALTHY TEXTURE
    path_synthesis_old = '/content/drive/My Drive/Datasets/covid19/results/cea_synthesis/patient0/'
    texture_orig = np.load(f'{path_synthesis_old}texture.npy.npz')
    texture_orig = texture_orig.f.arr_0
    texture = texture_orig + np.abs(np.min(texture_orig)) + .07
    texture = np.pad(texture,((100,100),(100,100)),mode='reflect')
    print(f'type(texture) = {type(texture)}, {np.shape(texture)}')
    #==/

    images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz"))[:10]) #OMM
    labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz"))[:10]) #OMM
    logging.info(f"training: image/label ({len(images)}) folder: {data_folder}")

    amp = True  # auto. mixed precision
    keys = ("image", "label")
    train_frac, val_frac = 0.8, 0.2
    n_train = int(train_frac * len(images)) + 1
    n_val = min(len(images) - n_train, int(val_frac * len(images)))
    logging.info(f"training: train {n_train} val {n_val}, folder: {data_folder}")

    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]

    # create a training data loader
    batch_size = 1 # XX was 2
    logging.info(f"batch size {batch_size}")
    train_transforms = get_xforms("synthesis", keys, keys2, path_synthesis, decreasing_sequence, scans_syns, texture)
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        # collate_fn=pad_list_data_collate,
    )

    # create a validation data loader
    val_transforms = get_xforms("val", keys)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # create BasicUNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)

    # if continue training
    if continue_training:
        ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
        ckpt = ckpts[-1]
        logging.info(f"continue training using {ckpt}.")
        net.load_state_dict(torch.load(ckpt, map_location=device))

    # max_epochs, lr, momentum = 500, 1e-4, 0.95
    max_epochs, lr, momentum = 20, 1e-4, 0.95 #OMM
    logging.info(f"epochs {max_epochs}, lr {lr}, momentum {momentum}")
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # create evaluator (to be used to measure model quality during training
    val_post_transform = monai.transforms.Compose(
        [AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=True, n_classes=2)]
    )
    val_handlers = [
        ProgressBar(),
        MetricsSaver(save_dir="./metrics_val", metrics="*"),
        CheckpointSaver(save_dir=model_folder, save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=6),
    ]
    evaluator = monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=get_inferer(),
        post_transform=val_post_transform,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=False, output_transform=lambda x: (x["pred"], x["label"]))
        },
        val_handlers=val_handlers,
        amp=amp,
    )

    # evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        # MetricsSaver(save_dir="./metrics_train", metrics="*"),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
    ]
    trainer = monai.engines.SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=DiceCELoss(),
        inferer=get_inferer(),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp,
    )
    trainer.run()


def infer(data_folder=".", model_folder="runs", prediction_folder="output"):
    """
    run inference, the output folder will be "./output"
    """
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "*_ct.nii.gz"))[:10]) #OMM
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer()
    saver = monai.data.NiftiSaver(output_dir=prediction_folder, mode="nearest")
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
            for _ in range(4):
                # test time augmentations
                _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                pred = inferer(_img.to(device), net)
                preds = preds + pred
                n = n + 1.0
                for dims in [[2], [3]]:
                    flip_pred = inferer(torch.flip(_img.to(device), dims=dims), net)
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0
            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            saver.save_batch(preds, infer_data["image_meta_dict"])

    # copy the saved segmentations into the required folder structure for submission
    submission_dir = os.path.join(prediction_folder, "to_submit")
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    files = glob.glob(os.path.join(prediction_folder, "volume*", "*.nii.gz"))
    for f in files:
        new_name = os.path.basename(f)
        new_name = new_name[len("volume-covid19-A-0"):]
        new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
        to_name = os.path.join(submission_dir, new_name)
        shutil.copy(f, to_name)
    logging.info(f"predictions copied to {submission_dir}.")


if __name__ == "__main__":
    """
    Usage:
        python run_net.py train --data_folder "COVID-19-20_v2/Train" # run the training pipeline
        python run_net.py infer --data_folder "COVID-19-20_v2/Validation" # run the inference pipeline
    """
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument(
        "mode", metavar="mode", default="train", choices=("train", "infer", "continue_train"), type=str, help="mode of workflow"
    )
    parser.add_argument("--data_folder", default="", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="runs", type=str, help="model folder")
    args = parser.parse_args()

    monai.config.print_config()
    monai.utils.set_determinism(seed=0)
    logging.basicConfig(handlers=[
        logging.FileHandler("./train_and_val.log"),
        logging.StreamHandler()
    ],
    level=logging.INFO)

    if args.mode == "train":
        data_folder = args.data_folder or os.path.join("COVID-19-20_v2", "Train")
        train(data_folder=data_folder, model_folder=args.model_folder)
    elif args.mode == "infer":
        data_folder = args.data_folder or os.path.join("COVID-19-20_v2", "Validation")
        infer(data_folder=data_folder, model_folder=args.model_folder)
    elif args.mode == "continue_train":
        data_folder = args.data_folder or os.path.join("COVID-19-20_v2", "Train")
        train(data_folder=data_folder, model_folder=args.model_folder, continue_training=True)
    else:
        raise ValueError("Unknown mode.")
# %%
