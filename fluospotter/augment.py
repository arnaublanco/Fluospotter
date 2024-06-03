"""Model utility functions for augmentation."""
import pdb
from typing import Tuple
import warnings

import numpy as np
import torch
import monai.transforms as t
from tifffile import imread


def permute_depth(x):
    return torch.permute(x, [0, 2, 3, 1])


def get_transforms_patches(n_samples, neg_samples, patch_size, n_classes=2, depth_last=False, p_app=0.1, pr_geom=0.1):
    def apply_depth_permutation(transforms_list):
        if depth_last:
            transforms_list.insert(1, t.Lambda(lambda d: {'img': permute_depth(d['img']), 'seg': permute_depth(d['seg'])}))

    tr_transforms = [
        t.Lambda(lambda d: {'img': torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0),
                            'seg': torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)}),
        t.CropForegroundd(keys=('img', 'seg'), source_key='seg'),
        t.ScaleIntensityd(keys=('img',)),
        t.RandCropByPosNegLabeld(keys=('img', 'seg'), label_key='seg', spatial_size=patch_size,
                                 num_samples=n_samples, pos=1, neg=neg_samples),
        t.RandScaleIntensityd(keys=('img',), factors=0.05, prob=p_app),
        t.RandShiftIntensityd(keys=('img',), offsets=0.05, prob=p_app),
        t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=0),
        t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=1),
        t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=2),
        t.AsDiscreted(keys=('seg'), to_onehot=n_classes)
    ]

    vl_transforms = [
        t.Lambda(lambda d: {'img': torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0),
                            'seg': torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)}),
        t.CropForegroundd(keys=('img', 'seg'), source_key='img'),
        t.ScaleIntensityd(keys=('img',)),
        t.AsDiscreted(keys=('seg'), to_onehot=3)
    ]

    # Apply depth permutation if needed
    apply_depth_permutation(tr_transforms)
    apply_depth_permutation(vl_transforms)

    return t.Compose(tr_transforms), t.Compose(vl_transforms)


def get_transforms_fullres(im_size, multiclass=False, p_app=0.1, pr_geom=0.1, depth_last=True):
    def apply_depth_permutation(transforms_list):
        if depth_last:
            transforms_list.insert(1, t.Lambda(lambda d: {'img': permute_depth(d['img']), 'seg': permute_depth(d['seg'])}))

    if multiclass:
        tr_transforms = [
            t.Lambda(lambda d: {'img': torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0),
                                'seg': torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)}),
            t.CropForegroundd(keys=('img', 'seg'), source_key='img'),  # 0 is the default threshold
            t.ScaleIntensityd(keys=('img',)),
            t.Resized(spatial_size=im_size, keys=('img', 'seg'), mode=('bilinear', 'nearest')),
            t.RandScaleIntensityd(keys=('img',), factors=0.05, prob=p_app),
            t.RandShiftIntensityd(keys=('img',), offsets=0.05, prob=p_app),
            t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=0),
            t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=1),
            t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=2),
            t.RandRotate90d(keys=('img', 'seg'), prob=pr_geom, max_k=3),
            t.AsDiscreted(keys=('seg'), to_onehot=3)
        ]

        vl_transforms = [
            t.Lambda(lambda d: {'img': torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0),
                                'seg': torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)}),
            t.CropForegroundd(keys=('img', 'seg'), source_key='img'),  # 0 is the default threshold
            t.ScaleIntensityd(keys=('img',)),
            t.Resized(spatial_size=im_size, keys=('img', 'seg'), mode=('bilinear', 'nearest')),
            t.NormalizeIntensityd(keys=('img',), nonzero=True),
            t.AsDiscreted(keys=('seg'), to_onehot=3)
        ]
    else:
        tr_transforms = t.Compose([
            t.Lambda(lambda d: {'img': torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0),
                                'seg': torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)}),
            t.CropForegroundd(keys=('img', 'seg'), source_key='seg'),
            t.ScaleIntensityd(keys=('img','seg')),
            t.RandCropByPosNegLabeld(keys=('img', 'seg'), label_key='seg', spatial_size=patch_size, num_samples=n_samples, pos=1, neg=neg_samples),
            t.RandScaleIntensityd(keys=('img', ), factors=0.05, prob=p_app),
            t.RandShiftIntensityd(keys=('img', ), offsets=0.05, prob=p_app),
            t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=0),
            t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=1),
            t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=2)
        ])

        vl_transforms = [
            t.Lambda(lambda d: {'img': torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0),
                                'seg': torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)}),
            t.CropForegroundd(keys=('img', 'seg'), source_key='img'),  # 0 is the default threshold
            t.ScaleIntensityd(keys=('img',)),
            t.Resized(spatial_size=im_size, keys=('img', 'seg'), mode=('bilinear', 'nearest')),
            t.NormalizeIntensityd(keys=('img',), nonzero=True),
            t.ScaleIntensityd(keys=('seg',))
        ]

    # Apply depth permutation if needed
    apply_depth_permutation(tr_transforms)
    apply_depth_permutation(vl_transforms)

    return t.Compose(tr_transforms), t.Compose(vl_transforms)
