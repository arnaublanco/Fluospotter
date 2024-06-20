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
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

def get_test_split(data_path='data', labels_path='labels'):
    data_test = os.path.join(data_path, 'test')
    annotations_test = os.path.join(labels_path, 'test')

    vol_list_test = os.listdir(data_test)
    seg_list_test = os.listdir(annotations_test)
    vol_list_test = [os.path.join(data_test, n) for n in vol_list_test]
    seg_list_test = [os.path.join(annotations_test, n) for n in seg_list_test]
    test_files = [{'img': img, 'seg': seg} for img, seg in zip(vol_list_test, seg_list_test)]

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


def get_loaders(data_path, labels_path, n_samples=1, neg_samples=1, patch_size=(48, 256, 256), num_workers=0, ovft_check=0, depth_last=False, n_classes=2):

    tr_files, vl_files = get_train_val_test_splits(data_path, labels_path)
    tr_transforms, vl_transforms = get_transforms_patches(n_samples, neg_samples, patch_size=patch_size, depth_last=depth_last, n_classes=n_classes)
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


def get_loaders_test(data_path, labels_path, n_samples=1, neg_samples=1, patch_size=(48, 256, 256), num_workers=0, depth_last=False, n_classes=2, im_size=(49, 512, 512), instance_seg=False):
    test_files = get_test_split(data_path, labels_path)
    _, test_transforms = get_transforms_patches(n_samples, neg_samples, patch_size=patch_size,
                                                          depth_last=depth_last, n_classes=n_classes, im_size=im_size, instance_seg=instance_seg)
    batch_size = 1
    gpu = torch.cuda.is_available()
    test_ds = md.Dataset(data=test_files, transform=test_transforms)
    test_loader = md.DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=gpu)
    return test_loader


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


def display_segmentation_metrics(metrics: Dict):
    pixel_wise_metrics = {k: v for k, v in metrics['pixel-wise'].items() if k not in ['roc_auc', 'tpr', 'fpr']}
    object_wise_metrics = {k: v for k, v in metrics['object-wise'].items() if k not in ['roc_auc', 'tpr', 'fpr']}

    pixel_wise_df = pd.DataFrame(data=pixel_wise_metrics)
    object_wise_df = pd.DataFrame(data=object_wise_metrics)

    print("Pixel-wise Metrics:")
    display(pixel_wise_df)

    plot_roc_auc(metrics['pixel-wise']['tpr'], metrics['pixel-wise']['fpr'], metrics['pixel-wise']['roc_auc'],
                 title='Pixel-wise ROC AUC')

    print("\nObject-wise Metrics:")
    display(object_wise_df)

    plot_roc_auc(metrics['object-wise']['tpr'], metrics['object-wise']['fpr'], metrics['object-wise']['roc_auc'],
                 title='Object-wise ROC AUC')


def plot_roc_auc(tpr_list, fpr_list, roc_auc_list, title):
    plt.figure()
    for i in range(len(tpr_list)):
        plt.plot(fpr_list[i], tpr_list[i], lw=2, label=f'ROC curve (area = {roc_auc_list[i]:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
