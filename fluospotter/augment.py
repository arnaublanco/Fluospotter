"""Model utility functions for augmentation."""

import numpy as np
import torch
import torch.nn.functional as F
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


def custom_transform(d, chunk_size=(105, 256, 256), overlap=0.2, im_size=(105, 1024, 1024)):
    """
    Custom transform function to resize, scale intensity, and split a 3D volume into chunks.

    Parameters:
    - d: dict, contains paths to the image ('img') and segmentation ('seg').
    - chunk_size: tuple, size of each chunk (depth, height, width).
    - overlap: float, overlap between chunks in pixels (default: 0).
    - im_size: tuple, desired size to resize the volume to (depth, height, width).

    Returns:
    - A dictionary containing the processed image and optionally segmentation mask.
    """
    # Initialize lists to store chunks
    img_chunks = []
    seg_chunks = []

    # Extract file paths from the tuples
    img_path = d['img'][0] if isinstance(d['img'], tuple) else d['img']
    seg_path = d['seg'][0] if isinstance(d['seg'], tuple) and d['seg'] else None

    # Open the image file and load the entire 3D volume
    with TiffFile(img_path) as tif:
        # Load the full 3D image volume (shape: (depth, original_height, original_width))
        img_volume = tif.asarray()

    # Convert the image volume to a tensor and add batch and channel dimensions
    img_tensor = torch.as_tensor(img_volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Resize the volume to the target size (im_size)
    img_resized = F.interpolate(img_tensor, size=im_size, mode='trilinear', align_corners=False)
    img_resized = img_resized.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

    # Scale intensity based on the entire resized volume
    img_min = img_resized.min()
    img_max = img_resized.max()
    img_resized = (img_resized - img_min) / (img_max - img_min + 1e-8)  # Normalize between 0 and 1

    img_shape = img_resized.shape  # Get the shape of the resized image

    # Calculate step size (accounting for overlap)
    step_z = chunk_size[0]
    step_y = chunk_size[1] - int(overlap * chunk_size[1])
    step_x = chunk_size[2] - int(overlap * chunk_size[2])

    # Loop through the resized volume in steps
    for y in range(0, img_shape[1], step_y):
        for x in range(0, img_shape[2], step_x):
            # Define the chunk bounds, ensuring they don't exceed the image boundaries
            y_start, y_end = y, min(y + chunk_size[1], img_shape[1])
            x_start, x_end = x, min(x + chunk_size[2], img_shape[2])

            # Read the chunk from the resized 3D volume
            img_chunk = img_resized[:, y_start:y_end, x_start:x_end]

            # Pad the chunk if its size is smaller than chunk_size
            padding = (
                0, max(0, chunk_size[2] - img_chunk.shape[2]),  # Padding for x dimension
                0, max(0, chunk_size[1] - img_chunk.shape[1]),  # Padding for y dimension
                0, 0  # No padding needed for z dimension as it should match
            )
            img_chunk = F.pad(img_chunk, padding, mode='constant', value=0)  # Pad with zeros

            # Convert to tensor and add batch dimension
            img_tensor_chunk = img_chunk.unsqueeze(0)  # Add batch dimension

            # Append to list of image chunks
            img_chunks.append(img_tensor_chunk)

            # If segmentation is provided, process segmentation chunks similarly
            if seg_path:  # Check if seg_path is valid
                with TiffFile(seg_path) as tif_seg:
                    # Load the full 3D segmentation volume
                    seg_volume = tif_seg.asarray()

                    # Convert the segmentation volume to a tensor and resize it
                    seg_tensor = torch.as_tensor(seg_volume.astype(np.int8)).unsqueeze(0).unsqueeze(0)
                    seg_resized = F.interpolate(seg_tensor, size=im_size, mode='nearest')  # Use nearest for segmentation
                    seg_resized = seg_resized.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

                    # Read the chunk from the resized segmentation volume
                    seg_chunk = seg_resized[:, y_start:y_end, x_start:x_end]

                    # Pad the chunk if its size is smaller than chunk_size
                    seg_chunk = F.pad(seg_chunk, padding, mode='constant', value=0)  # Pad with zeros

                    # Convert to tensor and add batch dimension
                    seg_tensor_chunk = seg_chunk.unsqueeze(0)  # Add batch dimension

                    # Append to list of segmentation chunks
                    seg_chunks.append(seg_tensor_chunk)

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
        lambda_transform = t.Lambda(lambda d: custom_transform(d, chunk_size=chunk_size, im_size=im_size))

    # Define training transforms
    tr_transforms = [
        lambda_transform,
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
        lambda_transform
    ]

    # Apply custom layers if needed
    add_custom_layers(tr_transforms)
    add_custom_layers(vl_transforms)

    return t.Compose(tr_transforms), t.Compose(vl_transforms)
