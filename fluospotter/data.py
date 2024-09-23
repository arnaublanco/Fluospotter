"""List of functions to handle data including converting matrices <-> coordinates."""
import pdb
from typing import Tuple, Dict
import os
import numpy as np
import torch
import monai.data as md
from torch.utils.data.dataset import Subset
from .augment import get_transforms_fullres, get_transforms_patches
from .metrics import iou


def get_test_split(data_path='data', labels_path=''):
    data_test = os.path.join(data_path, 'test')
    vol_list_test = os.listdir(data_test)
    vol_list_test = [os.path.join(data_test, n) for n in vol_list_test]

    if len(labels_path) > 0:
        annotations_test = os.path.join(labels_path, 'test')
        seg_list_test = os.listdir(annotations_test)
        seg_list_test = [os.path.join(annotations_test, n) for n in seg_list_test]
        test_files = [{'img': img, 'seg': seg} for img, seg in zip(vol_list_test, seg_list_test)]
    else:
        test_files = [{'img': img, 'seg': ''} for img in zip(vol_list_test)]

    return test_files


def get_train_val_test_splits(data_path='data', labels_path='labels'):
    data_train, data_valid = os.path.join(data_path, 'train'), os.path.join(data_path, 'valid')
    annotations_train, annotations_valid = os.path.join(labels_path, 'train'), os.path.join(labels_path, 'valid')

    vol_list_train, vol_list_valid = os.listdir(data_train), os.listdir(data_valid)
    seg_list_train, seg_list_valid = os.listdir(annotations_train), os.listdir(annotations_valid)
    vol_list_train, vol_list_valid = [os.path.join(data_train, n) for n in vol_list_train], [os.path.join(data_valid, n) for n in vol_list_valid]
    seg_list_train, seg_list_valid = [os.path.join(annotations_train, n) for n in seg_list_train], [os.path.join(annotations_valid, n) for n in seg_list_valid]
    tr_files, vl_files = [{'img': img, 'seg': seg} for img, seg in zip(vol_list_train, seg_list_train)], [{'img': img, 'seg': seg} for img, seg in zip(vol_list_valid, seg_list_valid)]
    print(40*'=')
    print('* Total Number of Volumes {}'.format(len(tr_files)+len(vl_files)))
    print(40 * '=')

    print('* Training samples = {}, Validation samples = {}'.format(len(tr_files), len(vl_files)))

    return tr_files, vl_files


def get_loaders_fullres(data_path, batch_size=1, im_size=(96, 96, 64), num_workers=0, tr_percentage=1., ovft_check=0):

    tr_files, vl_files = get_train_val_test_splits(data_path)

    if tr_percentage < 1.:
        print(40 * '-')
        n_tr_examples = len(tr_files)
        random_indexes = np.random.permutation(n_tr_examples)
        kept_indexes = int(n_tr_examples * tr_percentage)
        tr_files = [tr_files[i] for i in random_indexes[:kept_indexes]]
        print('Reducing training data from {} items to {}'.format(n_tr_examples, len(tr_files)))
        print(40 * '-')

    tr_transforms, vl_transforms = get_transforms_fullres(im_size=im_size)

    test_batch_size = 2*batch_size
    gpu = torch.cuda.is_available()
    tr_ds = md.Dataset(data=tr_files, transform=tr_transforms)

    vl_ds = md.Dataset(data=vl_files, transform=vl_transforms)
    tr_loader = md.DataLoader(tr_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=gpu)
    vl_loader = md.DataLoader(vl_ds, batch_size=test_batch_size, num_workers=num_workers, pin_memory=gpu)
    if ovft_check > 0:
        ovft_ds = md.Dataset(data=tr_files, transform=vl_transforms)
        subset_size = len(vl_ds)
        subset_idxs = torch.randperm(len(ovft_ds))[:subset_size]
        ovft_ds = Subset(ovft_ds, subset_idxs)
    else: ovft_ds = md.Dataset(data=tr_files, transform=vl_transforms)
    ovft_loader = md.DataLoader(ovft_ds, batch_size=test_batch_size, num_workers=num_workers, pin_memory=gpu)

    return tr_loader, ovft_loader, vl_loader


def get_loaders(data_path, labels_path, n_samples=1, neg_samples=1, patch_size=(48, 256, 256), num_workers=0, ovft_check=0, depth_last=False, n_classes=2, im_size=(48,512,512)):

    tr_files, vl_files = get_train_val_test_splits(data_path, labels_path)
    tr_transforms, vl_transforms = get_transforms_patches(n_samples, neg_samples, patch_size=patch_size, depth_last=depth_last, n_classes=n_classes, im_size=im_size)
    batch_size = 1
    test_batch_size = 1
    gpu = torch.cuda.is_available()

    tr_ds = md.Dataset(data=tr_files, transform=tr_transforms)
    vl_ds = md.Dataset(data=vl_files, transform=vl_transforms)
    tr_loader = md.DataLoader(tr_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=gpu)
    vl_loader = md.DataLoader(vl_ds, batch_size=test_batch_size, num_workers=0, pin_memory=gpu)
    
    if ovft_check > 0:
        ovft_ds = md.Dataset(data=tr_files, transform=vl_transforms)
        subset_size = len(vl_ds)
        subset_idxs = torch.randperm(len(ovft_ds))[:subset_size]
        ovft_ds = Subset(ovft_ds, subset_idxs)
    else: ovft_ds = md.Dataset(data=tr_files, transform=vl_transforms)
    ovft_loader = md.DataLoader(ovft_ds, batch_size=test_batch_size, num_workers=0, pin_memory=gpu)
    return tr_loader, ovft_loader, vl_loader


'''def get_data_from_mask(volume, mask):
    labels = np.unique(mask)[1:]
    sizes = [np.where(volume == l) for l in labels]
    for l in labels:
        np.where(volume == l)
        volume[volume == l]
    return data'''


def get_loaders_test(data_path, labels_path, n_samples=1, neg_samples=1, patch_size=(48, 256, 256), num_workers=0, depth_last=False, n_classes=2, im_size=(48, 512, 512), instance_seg=False, is_numpy=False):
    if is_numpy:
        if len(data_path.shape) == 4:
            test_files = { 'img': [] }
            for n in range(data_path.shape[0]): test_files['img'].append(data_path[n])
        else:
            test_files = {'img': [data_path]}
    else:
        test_files = get_test_split(data_path, labels_path)
    _, test_transforms = get_transforms_patches(n_samples, neg_samples, patch_size=patch_size,
                                                          depth_last=depth_last, n_classes=n_classes, im_size=im_size, instance_seg=instance_seg, is_numpy=is_numpy, chunk_size=(im_size.shape[0], 256, 256))
    batch_size = 1
    gpu = torch.cuda.is_available()
    test_ds = md.Dataset(data=test_files, transform=test_transforms)
    test_loader = md.DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=gpu)
    return test_loader


def join_connected_puncta(data, mask):
    out = np.zeros_like(mask, dtype=np.int8)
    labels = np.unique(mask)
    for l in labels[1:]:
        mask_label = mask == l
        if not np.any(mask_label):
            continue

        max_idx = np.argmax(data[mask_label])

        max_pos_mask = np.zeros_like(mask_label)
        max_pos_mask[np.nonzero(mask_label)[0][max_idx],
                     np.nonzero(mask_label)[1][max_idx],
                     np.nonzero(mask_label)[2][max_idx]] = True
        out[max_pos_mask] = 1
    return out


def match_labeling(actual, predicted):
    """
    Relabels the predicted segmentation labels to match the ground truth labels
    based on the highest Intersection over Union (IoU) overlap.

    Args:
        actual (np.array): Ground truth labels.
        predicted (np.array): Predicted labels.

    Returns:
        np.array: Relabeled predicted labels.
    """
    new_labels = np.zeros_like(predicted)
    counter = 0

    # Iterate over each unique predicted label (ignoring the background label 0)
    print("Total predicted labels:",len(np.unique(predicted))-1, "Total real labels:", len(np.unique(actual))-1)
    for pred_label in np.unique(predicted):
        print("Label:",pred_label)
        if pred_label == 0:
            continue
        current = (predicted == pred_label)  # Binary mask of current predicted label
        best_label, best_overlap = None, 0  # Initialize with no best label and best overlap

        # Iterate over each unique actual label (ignoring the background label 0)
        for actual_label in np.unique(actual):
            if actual_label == 0:
                continue
            overlap = iou(actual == actual_label, current)  # Compute IoU
            if overlap > best_overlap:  # If better overlap is found
                best_overlap = overlap
                best_label = actual_label

        # If a matching label is found, use it; otherwise, assign a new unique label
        if best_label is not None:
            new_labels[predicted == pred_label] = best_label
        else:
            # Assign a new unique label that is not in used_labels
            new_label = max(np.unique(actual)) + 1 + counter
            counter += 1
            new_labels[predicted == pred_label] = new_label

    return new_labels