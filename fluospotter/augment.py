"""Model utility functions for augmentation."""

import numpy as np
import torch
import monai.transforms as t
from tifffile import imread, TiffFile


def permute_depth(x):
    return torch.permute(x, [0, 2, 3, 1])


'''def custom_transform(d):
    img = torch.as_tensor(imread(d['img']).astype(np.float32)).unsqueeze(0)
    if 'seg' in d and d['seg']:
        seg = torch.as_tensor(imread(d['seg']).astype(np.int8)).unsqueeze(0)
        return {'img': img, 'seg': seg}
    else:
        return {'img': img}'''


def custom_transform(d, chunk_size=(105, 256, 256), overlap=0):
    # Initialize lists to store chunks
    img_chunks = []
    seg_chunks = []

    # Open the image file with TiffFile
    with TiffFile(d['img']) as tif:
        # Get the shape of the entire image
        img_shape = tif.pages[0].shape  # (depth, height, width)

        # Calculate step size (accounting for overlap)
        step_z, step_y, step_x = chunk_size[0], chunk_size[1] - overlap, chunk_size[2] - overlap

        # Loop through the volume in steps
        for y in range(0, img_shape[1], step_y):
            for x in range(0, img_shape[2], step_x):
                # Define the chunk bounds, ensuring they don't exceed the image boundaries
                y_start, y_end = y, min(y + chunk_size[1], img_shape[1])
                x_start, x_end = x, min(x + chunk_size[2], img_shape[2])

                # Read the chunk from the image
                img_chunk = tif.asarray(key=slice(0, img_shape[0]))[:, y_start:y_end, x_start:x_end]

                # Convert to tensor and add batch dimension
                img_tensor = torch.as_tensor(img_chunk.astype(np.float32)).unsqueeze(0)

                # Append to list of image chunks
                img_chunks.append(img_tensor)

                # If segmentation is provided, process segmentation chunks similarly
                if 'seg' in d and d['seg']:
                    with TiffFile(d['seg']) as tif_seg:
                        seg_chunk = tif_seg.asarray(key=slice(0, img_shape[0]))[:, y_start:y_end, x_start:x_end]
                        seg_tensor = torch.as_tensor(seg_chunk.astype(np.int8)).unsqueeze(0)
                        seg_chunks.append(seg_tensor)

    # Concatenate image chunks along batch dimension
    img = torch.cat(img_chunks, dim=0)

    # Concatenate segmentation chunks if available
    if seg_chunks:
        seg = torch.cat(seg_chunks, dim=0)
        return {'img': img, 'seg': seg}
    else:
        return {'img': img}


def custom_transform_numpy(d):
    img = torch.as_tensor(d['img'].astype(np.float32)).unsqueeze(0)
    if 'seg' in d and d['seg']:
        seg = torch.as_tensor(d['seg'].astype(np.int8)).unsqueeze(0)
        return {'img': img, 'seg': seg}
    else:
        return {'img': img}


def get_transforms_patches(n_samples, neg_samples, patch_size, im_size, n_classes=2, depth_last=False, p_app=0.1, pr_geom=0.1, instance_seg=False, is_numpy=False, chunk_size=(105, 256, 256)):
    """
    Generate transformation pipelines for training and validation.

    Parameters:
    - n_samples: int, number of samples to generate per image.
    - neg_samples: int, number of negative samples.
    - patch_size: tuple, size of the patch for cropping.
    - im_size: tuple, size to resize the images to.
    - n_classes: int, number of segmentation classes.
    - depth_last: bool, whether to permute depth to the last dimension.
    - p_app: float, probability of applying appearance transformations.
    - pr_geom: float, probability of applying geometric transformations.
    - instance_seg: bool, whether instance segmentation is enabled.
    - is_numpy: bool, whether the input is in NumPy format.
    - chunk_size: tuple, size of the chunk to be used in custom_transform.

    Returns:
    - Two composed transforms for training and validation.
    """
    def add_custom_layers(transforms_list):
        if not instance_seg:
            transforms_list.append(t.AsDiscreted(keys=('seg'), to_onehot=n_classes, allow_missing_keys=True))
        if depth_last:
            transforms_list.insert(1, t.Lambda(lambda d: {'img': permute_depth(d['img']), 'seg': permute_depth(d['seg'])}))

    # Define the lambda transform with the specified chunk size
    if is_numpy:
        lambda_transform = t.Lambda(lambda d: custom_transform_numpy(d))
    else:
        lambda_transform = t.Lambda(lambda d: custom_transform(d, chunk_size=chunk_size))

    # Define training transforms
    tr_transforms = [
        lambda_transform,
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

    # Define validation transforms
    vl_transforms = [
        lambda_transform,
        t.ScaleIntensityd(keys=('img',)),
        t.Resized(spatial_size=im_size, keys=('img', 'seg'), mode=('bilinear', 'nearest'), allow_missing_keys=True)
    ]

    # Apply custom layers if needed
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
