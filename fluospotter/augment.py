"""Model utility functions for augmentation."""

import numpy as np
import torch
import monai.transforms as t
from tifffile import imread


def permute_depth(x):
    return torch.permute(x, [0, 2, 3, 1])


def custom_transform(d):
    img = torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0)
    if 'seg' in d and d['seg']:
        seg = torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)
        return {'img': img, 'seg': seg}
    else:
        return {'img': img}


def get_transforms_patches(n_samples, neg_samples, patch_size, im_size, n_classes=2, depth_last=False, p_app=0.1, pr_geom=0.1, instance_seg=False):
    def add_custom_layers(transforms_list):
        if not instance_seg:
            transforms_list.append(t.AsDiscreted(keys=('seg'), to_onehot=n_classes, allow_missing_keys=True))
        if depth_last:
            transforms_list.insert(1, t.Lambda(lambda d: {'img': permute_depth(d['img']), 'seg': permute_depth(d['seg'])}))

    tr_transforms = [
        t.Lambda(lambda d: {'img': torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0),
                            'seg': torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)}),
        t.ScaleIntensityd(keys=('img',)),
        t.Resized(spatial_size=im_size, keys=('img', 'seg'), mode=('bilinear', 'nearest')),
        t.RandCropByPosNegLabeld(keys=('img', 'seg'), label_key='seg', spatial_size=patch_size,
                                 num_samples=n_samples, pos=1, neg=neg_samples),
        t.RandScaleIntensityd(keys=('img',), factors=0.05, prob=p_app),
        t.RandShiftIntensityd(keys=('img',), offsets=0.05, prob=p_app),
        t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=0),
        t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=1),
        t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=2)
    ]

    vl_transforms = [
        t.Lambda(custom_transform),
        t.ScaleIntensityd(keys=('img',)),
        t.Resized(spatial_size=im_size, keys=('img', 'seg'), mode=('bilinear', 'nearest'), allow_missing_keys=True)
    ]

    # Apply depth permutation if needed
    add_custom_layers(tr_transforms)
    add_custom_layers(vl_transforms)

    return t.Compose(tr_transforms), t.Compose(vl_transforms)


def get_transforms_fullres(n_samples, neg_samples, patch_size, n_classes=2, depth_last=False, p_app=0.1, pr_geom=0.1, instance_seg=False):
    def add_custom_layers(transforms_list):
        if not instance_seg:
            transforms_list.append(t.AsDiscreted(keys=('seg'), to_onehot=n_classes))
        if depth_last:
            transforms_list.insert(1,
                                   t.Lambda(lambda d: {'img': permute_depth(d['img']), 'seg': permute_depth(d['seg'])}))
    tr_transforms = [
        t.Lambda(lambda d: {'img': torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0),
                            'seg': torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)}),
        t.CropForegroundd(keys=('img', 'seg'), source_key='seg', allow_smaller=False),
        t.ScaleIntensityd(keys=('img',)),
        t.RandCropByPosNegLabeld(keys=('img', 'seg'), label_key='seg', spatial_size=patch_size,
                                 num_samples=n_samples, pos=1, neg=neg_samples),
        t.RandScaleIntensityd(keys=('img',), factors=0.05, prob=p_app),
        t.RandShiftIntensityd(keys=('img',), offsets=0.05, prob=p_app),
        t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=0),
        t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=1),
        t.RandFlipd(keys=('img', 'seg'), prob=pr_geom, spatial_axis=2)
    ]

    vl_transforms = [
        t.Lambda(lambda d: {'img': torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0),
                            'seg': torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)}),
        t.CropForegroundd(keys=('img', 'seg'), source_key='img', allow_smaller=False),
        t.ScaleIntensityd(keys=('img',)),
    ]

    # Apply depth permutation if needed
    add_custom_layers(tr_transforms)
    add_custom_layers(vl_transforms)

    return t.Compose(tr_transforms), t.Compose(vl_transforms)
