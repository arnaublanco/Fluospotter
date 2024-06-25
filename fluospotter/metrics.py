"""Functions to calculate training loss on single image."""
import pdb
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from typing import Dict

EPS = 1e-12


def compute_puncta_metrics(predicted: np.array, actual: np.array, metrics: Dict) -> Dict:
    if len(metrics) == 0:
        metrics = {
            'precision': [],
            'recall': [],
            'f1-score': [],
            'number_peaks': [],
            'ground_truth': []
        }
    predicted, actual = mask_to_coords_conversion(predicted), mask_to_coords_conversion(actual)
    precision, recall, f1_score_metric = precision_recall_f1_score_coordinates(actual, predicted)
    metrics['precision'].append(precision), metrics['recall'].append(recall), metrics['f1-score'].append(f1_score_metric)
    metrics['number_peaks'].append(len(predicted))
    metrics['ground_truth'].append(len(actual))

    return metrics


def compute_segmentation_metrics(predicted: np.array, actual: np.array, metrics: Dict) -> Dict:
    if len(metrics) == 0:
        metrics = {'pixel-wise': {
            'iou': [],
            'dice': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            "f1-score": [],
            'roc_auc': [],
            'tpr': [],
            'fpr': [],
        }, 'object-wise': {
            'detection_accuracy': [],
            'segmentation_quality': [],
            'panoptic_quality': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            "f1-score": [],
            'roc_auc': [],
            'tpr': [],
            'fpr': []
        }}
    matches = matched_segments(actual, predicted)
    pq, da, sq = panoptic_quality(actual, predicted, matches)
    metrics['object-wise']['detection_accuracy'].append(da)
    metrics['object-wise']['segmentation_quality'].append(sq)
    metrics['object-wise']['panoptic_quality'].append(pq)
    precision, recall, f1_score_metric = precision_recall_f1_score(matches)
    metrics['object-wise']['precision'].append(precision), metrics['object-wise']['recall'].append(recall)
    metrics['object-wise']['f1-score'].append(f1_score_metric)

    actual_bin, predicted_bin = (actual > 0).astype(np.int8), (predicted > 0).astype(np.int8)
    metrics['pixel-wise']['iou'].append(iou(actual_bin, predicted_bin))
    metrics['pixel-wise']['dice'].append(dice_coefficient(actual_bin, predicted_bin))
    precision, recall, specif, f1_score_metric = precision_recall_specificity_f1_score_pixel_score(actual_bin, predicted_bin)
    metrics['pixel-wise']['precision'].append(precision), metrics['pixel-wise']['recall'].append(recall)
    metrics['pixel-wise']['specificity'].append(specif), metrics['pixel-wise']['f1-score'].append(f1_score_metric)

    return metrics


def iou(actual, predicted) -> float:
    """Calculates the Intersection over Union (IoU) score."""
    tp = np.sum((actual == 1) & (predicted == 1))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    return tp / (tp + fp + fn + EPS)


def dice_coefficient(actual, predicted) -> float:
    """Calculates the Dice Coefficient."""
    tp = np.sum((actual == 1) & (predicted == 1))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    return 2 * tp / (2 * tp + fp + fn + EPS)


def matched_segments(actual, predicted, thr=0.5) -> Dict:
    matches = {}
    for label in np.unique(actual)[1:]:
        actual_mask = (actual == label)
        iou_scores = {pred_label: iou(actual_mask, (predicted == pred_label)) for pred_label in
                      np.unique(predicted)[1:]}
        matches[label] = [(pred_label, score) for pred_label, score in iou_scores.items() if score > thr]
    return matches


def panoptic_quality(actual, predicted, matches) -> (float, float, float):
    tp, fp, fn = 0, 0, 0

    matched_actual = set()
    matched_predicted = set()

    for actual_label, preds in matches.items():
        if preds:
            best_match = max(preds, key=lambda x: x[1])  # Choose the best IoU match
            matched_actual.add(actual_label)
            matched_predicted.add(best_match[0])
            tp += 1

    fp = len(np.unique(predicted)[1:]) - len(matched_predicted)
    fn = len(np.unique(actual)[1:]) - len(matched_actual)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + EPS)

    sq = 0
    for actual_label, preds in matches.items():
        if preds:
            best_match = max(preds, key=lambda x: x[1])
            sq += best_match[1]

    sq /= (tp + EPS)

    return dq * sq, dq, sq


def precision_recall_specificity_f1_score_pixel_score(actual, predicted) -> (float, float, float, float):
    tp = np.sum((actual == 1) & (predicted == 1))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    tn = np.sum((actual == 0) & (predicted == 0))

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)
    f1_score = 2 * precision * recall / (precision + recall + EPS)

    return precision, recall, specificity, f1_score


def mask_to_coords_conversion(mask):
    z, y, x = np.where(mask > 0)
    coords = list(zip(z, y, x))
    return np.array(coords)


def precision_recall_f1_score_coordinates(actual, predicted, threshold=3) -> (float, float, float):
    differences = np.abs(actual[:, None] - predicted)
    below_threshold = np.all(differences < threshold, axis=-1)

    tp = np.sum(np.any(below_threshold, axis=1))
    fp = np.sum(np.logical_not(np.any(below_threshold, axis=0)))
    fn = np.sum(np.logical_not(np.any(below_threshold, axis=1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def precision_recall_f1_score(matches: Dict[int, List[Tuple[int, float]]]) -> Tuple[float, float, float]:
    tp, fp, fn = 0, 0, 0

    for match in matches.values():
        if match:  # If match is not empty
            for actual_match, iou_score in match:
                if iou_score >= 0.5:  # Assuming IoU threshold of 0.5 for a positive match
                    tp += 1
                else:
                    fp += 1
        else:
            fn += 1  # If there are no matches, it means a false negative

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1_score = 2 * precision * recall / (precision + recall + EPS)

    return precision, recall, f1_score


def fast_bin_auc(actual, predicted, partial=False):
    actual, predicted = actual.flatten(), predicted.flatten()
    if partial:
        n_nonzeros = np.count_nonzero(actual)
        n_zeros = len(actual) - n_nonzeros
        k = min(n_zeros, n_nonzeros)
        predicted = np.concatenate([
            np.sort(predicted[actual == 0])[::-1][:k],
            np.sort(predicted[actual == 1])[::-1][:k]
        ])
        actual = np.concatenate([np.zeros(k), np.ones(k)])

    r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    if n_pos == 0 or n_neg == 0: return 0
    return (np.sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fast_bin_dice(actual, predicted):
    actual = np.asarray(actual).astype(bool) # bools are way faster to deal with by numpy
    predicted = np.asarray(predicted).astype(bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0: return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum