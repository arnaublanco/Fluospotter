"""Functions to calculate training loss on single image."""

from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import rankdata
from typing import Dict, Any
from datasets import Dataset
import pdb
from sklearn.metrics import auc

EPS = 1e-12


def compute_segmentation_metrics(predicted: np.array, actual: np.array, metrics: Dict) -> Dict:
    if len(metrics) == 0:
        metrics = {'pixel-wise': {
            'iou': [],
            'dice': [],
            'precision': [],
            'recall': [],
            'roc_auc': [],
            'tpr': [],
            'fpr': []
        }, 'object-wise': {
            'detection_accuracy': [],
            'segmentation_quality': [],
            'panoptic_quality': [],
            'precision': [],
            'recall': [],
            'roc_auc': [],
            'tpr': [],
            'fpr': []
        }}

    matches = matched_segments(actual, predicted)
    pq, da, sq = panoptic_quality(actual, predicted)
    metrics['object-wise']['detection_accuracy'].append(da)
    metrics['object-wise']['segmentation_quality'].append(sq)
    metrics['object-wise']['panoptic_quality'].append(pq)
    precision, recall = precision_recall_score(matches)
    metrics['object-wise']['precision'].append(precision), metrics['object-wise']['recall'].append(recall)
    roc_auc_score, tpr_list, fpr_list = roc_auc(predicted, actual)
    metrics['object-wise']['roc_auc'].append(roc_auc_score)
    metrics['object-wise']['tpr'].append(tpr_list)
    metrics['object-wise']['fpr'].append(fpr_list)

    actual_bin, predicted_bin = (actual > 0).astype(np.int8), (predicted > 0).astype(np.int8)
    metrics['pixel-wise']['iou'].append(iou(actual_bin, predicted_bin))
    metrics['pixel-wise']['dice'].append(dice_coefficient(actual_bin, predicted_bin))
    precision, recall = precision_recall_pixel_score(actual_bin, predicted_bin)
    metrics['pixel-wise']['precision'].append(precision), metrics['pixel-wise']['recall'].append(recall)
    roc_auc_score, tpr_list, fpr_list = roc_auc_pixel(predicted_bin, actual_bin)
    metrics['pixel-wise']['roc_auc'].append(roc_auc_score)
    metrics['pixel-wise']['tpr'].append(tpr_list)
    metrics['pixel-wise']['fpr'].append(fpr_list)

    return metrics


def roc_auc(predicted: np.array, actual: np.array) -> (list, list, list):
    ious = []
    for label in np.unique(actual)[1:]:
        actual_mask = (actual == label)
        ious.extend([iou(actual_mask, (predicted == pred_label)) for pred_label in np.unique(predicted)[1:]])

    iou_thresholds = np.linspace(0, 1, 101)
    tpr_list = []
    fpr_list = []

    for thr in iou_thresholds:
        tp = np.sum(np.array(ious) >= thr)
        fp = np.sum(np.array(ious) < thr)
        fn = len(np.unique(actual)[1:]) - tp
        tpr = tp / (tp + fn + EPS)
        fpr = fp / (fp + len(np.unique(predicted)[1:]) - tp + EPS)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return auc(fpr_list, tpr_list), tpr_list, fpr_list


def roc_auc_pixel(predicted: np.array, actual: np.array) -> (float, list, list):
    """
    Compute ROC AUC for semantic segmentation (pixel-wise).

    Args:
        predicted (np.array): Predicted binary mask (0's and 1's).
        actual (np.array): Ground truth binary mask (0's and 1's).

    Returns:
        (float, list, list): AUC, TPR list, FPR list.
    """

    iou_thresholds = np.linspace(0, 1, 101)
    tpr_list = []
    fpr_list = []

    for thr in iou_thresholds:
        tp_thr = np.sum((actual == 1) & (predicted >= thr))
        fp_thr = np.sum((actual == 0) & (predicted >= thr))
        fn_thr = np.sum((actual == 1) & (predicted < thr))
        tn_thr = np.sum((actual == 0) & (predicted < thr))

        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
        tpr = tp_thr / (tp_thr + fn_thr + EPS)
        fpr = fp_thr / (fp_thr + tn_thr + EPS)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Compute AUC
    roc_auc_score = auc(fpr_list, tpr_list)

    return roc_auc_score, tpr_list, fpr_list

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


def panoptic_quality(actual, predicted) -> (float, float, float):
    tp, fp, fn = 0, 0, 0
    matches = matched_segments(actual, predicted)

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


def precision_recall_pixel_score(actual, predicted) -> (float, float):
    tp = np.sum((actual == 1) & (predicted == 1))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    return precision, recall


def precision_recall_score(matches) -> (float, float):
    tp, fp, fn = 0, 0, 0

    for match in matches.values():
        if len(match) == 2:
            actual_match, predicted_match = match
            if actual_match and predicted_match:
                tp += 1
            elif not actual_match and predicted_match:
                fp += 1
            elif actual_match and not predicted_match:
                fn += 1
        else:
            if match:
                actual_match = match[0]
                if actual_match:
                    fn += 1  # Ground truth exists, but prediction does not
                else:
                    fp += 1  # Prediction exists, but ground truth does not
            else:
                fn += 1  # Neither ground truth nor prediction exists

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    return precision, recall


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

def euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the euclidean distance between two the points (x1, y1) and (x2, y2)."""
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))


def offset_euclidean(offset: List[tuple]) -> np.ndarray:
    """Calculates the euclidean distance based on row_column_offsets per coordinate."""
    return np.sqrt(np.sum(np.square(np.array(offset)), axis=-1))


def precision_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Precision score metric.

    Defined as ``tp / (tp + fp)`` where tp is the number of true positives and fp the number of false positives.
    Can be interpreted as the accuracy to not mislabel samples or how many selected items are relevant.
    The best value is 1 and the worst value is 0.

    NOTE – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
        true: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
    """
    selection = pred[..., 0] == 1
    precision = np.mean(true[selection, 0])
    return precision


def recall_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Recall score metric.

    Defined as ``tp / (tp + fn)`` where tp is the number of true positives and fn the number of false negatives.
    Can be interpreted as the accuracy of finding positive samples or how many relevant samples were selected.
    The best value is 1 and the worst value is 0.

    NOTE – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
        true: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
    """
    selection = true[..., 0] == 1
    recall = np.mean(pred[selection, 0])
    return recall


def f1_score(pred: np.ndarray, true: np.ndarray) -> Optional[float]:
    r"""F1 score metric.

    .. math::
        F1 = \frac{2 * precision * recall} / {precision + recall}.

    The equally weighted average of precision and recall.
    The best value is 1 and the worst value is 0.

    NOTE – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
        true: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
    """
    recall = recall_score(pred, true)
    precision = precision_score(pred, true)

    if recall == 0 and precision == 0:
        return None

    f1_value = (2 * precision * recall) / (precision + recall + EPS)
    return f1_value


def linear_sum_assignment(
    matrix: np.ndarray, cutoff: float = None
) -> Tuple[list, list]:
    """Solve the linear sum assignment problem with a cutoff.

    A problem instance is described by matrix matrix where each matrix[i, j]
    is the cost of matching i (worker) with j (job). The goal is to find the
    most optimal assignment of j to i if the given cost is below the cutoff.

    Args:
        matrix: Matrix containing cost/distance to assign cols to rows.
        cutoff: Maximum cost/distance value assignments can have.

    Returns:
        (rows, columns) corresponding to the matching assignment.
    """
    # Handle zero-sized matrices (occurs if true or pred has no items)
    if matrix.size == 0:
        return [], []

    # Prevent scipy to optimize on values above the cutoff
    if cutoff is not None and cutoff != 0:
        matrix = np.where(matrix >= cutoff, matrix.max(), matrix)

    row, col = scipy.optimize.linear_sum_assignment(matrix)

    if cutoff is None:
        return list(row), list(col)

    # As scipy will still assign all columns to rows
    # We here remove assigned values falling below the cutoff
    nrow = []
    ncol = []
    for r, c in zip(row, col):
        if matrix[r, c] <= cutoff:
            nrow.append(r)
            ncol.append(c)
    return nrow, ncol


def f1_integral(
    pred: np.ndarray,
    true: np.ndarray,
    mdist: float = 3.0,
    n_cutoffs: int = 50,
    return_raw: bool = False,
) -> Union[float, tuple]:
    """F1 integral calculation / area under F1 vs. cutoff.

    Compute the area under the curve when plotting F1 score vs cutoff values.
    Optimal score is ~1 (floating point inaccuracy) when F1 is achieved
    across all cutoff values including 0.

    Args:
        pred: Array of shape (n, 2) for predicted coordinates.
        true: Array of shape (n, 2) for ground truth coordinates.
        mdist: Maximum cutoff distance to calculate F1. Defaults to None.
        n_cutoffs: Number of intermediate cutoff steps. Defaults to 50.
        return_raw: If True, returns f1_scores, offsets, and cutoffs. Defaults to False.

    Returns:
        By default returns a single value in the f1_integral score.
        If return_raw is True, a tuple containing:
        * f1_scores: The non-integrated list of F1 values for all cutoffs.
        * offsets: Offset in r, c on predicted coords assigned to true coords
        * cutoffs: A list of all cutoffs used

    Notes:
        Scipy.spatial.distance.cdist((xa*n), (xb*n)) returns a matrix of shape (xa*xb). Here we use
        pred as xa and true as xb. This means that the matrix has all true coordinates along the row axis
        and all pred coordinates along the column axis. It's transpose has the opposite. The linear assignment
        takes in a cost matrix and returns the coordinates to assigned costs which fall below a defined cutoff.
        This assigment takes the rows as reference and assignes columns to them. Therefore, the transpose
        matrix resulting in row and column coordinates named "true_pred_r" and "true_pred_c" respectively
        uses true (along matrix row axis) as reference and pred (along matrix column axis) as assigments.
        In other terms the assigned predictions that are close to ground truth coordinates. To now calculate
        the offsets, we can use the "true_pred" rows and columns to find the originally referenced coordinates.
        As mentioned, the matrix has true along its row axis and pred along its column axis.
        Thereby we can use basic indexing. The [0] and [1] index refer to the coordinates' row and column value.
        This offset is now used two-fold. Once to plot the scatter pattern to make sure models aren't biased
        in one direction and secondly to compute the euclidean distance.

        The euclidean distance could not simply be summed up like with the F1 score because the different
        cutoffs actively influence the maximum euclidean distance score. Here, instead, we sum up all
        distances measured across every cutoff and then dividing by the total number of assigned coordinates.
        This automatically weighs models with more detections at lower cutoff scores.
    """
    cutoffs = np.linspace(start=0, stop=mdist, num=n_cutoffs)

    if pred.size == 0 or true.size == 0:
        warnings.warn(
            f"Pred ({pred.shape}) and true ({true.shape}) must have size != 0.",
            RuntimeWarning,
        )
        return 0.0 if not return_raw else (np.zeros(50), np.zeros(50), cutoffs)

    matrix = scipy.spatial.distance.cdist(pred, true, metric="euclidean")

    if not return_raw:
        f1_scores = [_f1_at_cutoff(matrix, pred, true, cutoff) for cutoff in cutoffs]
        return np.trapz(f1_scores, cutoffs) / mdist  # Norm. to 0-1

    f1_scores = []
    offsets = []
    for cutoff in cutoffs:
        f1_value, rows, cols = _f1_at_cutoff(
            matrix, pred, true, cutoff, return_raw=True
        )
        f1_scores.append(f1_value)
        offsets.append(_get_offsets(pred, true, rows, cols))

    return (f1_scores, offsets, list(cutoffs))


def _get_offsets(
    pred: np.ndarray, true: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> List[tuple]:
    """Return a list of (r, c) offsets for all assigned coordinates.

    Args:
        pred: List of all predicted coordinates.
        true: List of all ground truth coordinates.
        rows: Rows of the assigned coordinates (along "true"-axis).
        cols: Columns of the assigned coordinates (along "pred"-axis).
    """
    return [
        (true[r][0] - pred[c][0], true[r][1] - pred[c][1]) for r, c in zip(rows, cols)
    ]


# TODO - find suitable return type Union[float, tuple] does not work
def _f1_at_cutoff(
    matrix: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    cutoff: float,
    return_raw=False,
):
    """Compute a single F1 value at a given cutoff.

    Args:
        matrix: Cost matrix (euclidean distances) mapping true coordinates onto predicted
            coordinates. I.e. true along row-axis and pred along column-axis.
        pred: List of all predicted coordinates.
        true: List of all ground truth coordinates.
        cutoff: Single value to threshold cost values for assignments.
        return_raw: If True, the f1_score will be returned in addition to
            true_pred_r, and true_pred_c explained below. Defaults to False.

    Returns:
        By default returns a single value in the f1 score at the specified cutoff.
        If return_raw is True, a tuple containing:
        * f1_score: As by default.
        * true_pred_r: The rows from the true<-pred assignment. I.e. the indices of the
            true coordinates that have assigned pred coordinates.
        * true_pred_c: The columns from the true<-pred assignment. I.e. the indices of the
            pred coordinates that were assigned to true coordinates.
    """
    # Cannot assign coordinates on empty matrix
    if matrix.size == 0:
        return 0.0 if not return_raw else (0.0, [], [])

    # Assignment of pred<-true and true<-pred
    pred_true_r, _ = linear_sum_assignment(matrix, cutoff)
    true_pred_r, true_pred_c = linear_sum_assignment(matrix.T, cutoff)

    # Calculation of tp/fn/fp based on number of assignments
    tp = len(true_pred_r)
    fn = len(true) - len(true_pred_r)
    fp = len(pred) - len(pred_true_r)

    recall = tp / (tp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    f1_value = (2 * precision * recall) / (precision + recall + EPS)

    if return_raw:
        return f1_value, true_pred_r, true_pred_c

    return f1_value


def compute_metrics(
    pred: np.ndarray, true: np.ndarray, mdist: float = 3.0
) -> pd.DataFrame:
    """Calculate metric scores across cutoffs.

    Args:
        pred: Predicted set of coordinates.
        true: Ground truth set of coordinates.
        mdist: Maximum euclidean distance in px to which F1 scores will be calculated.

    Returns:
        DataFrame with one row per cutoff containing columns for:
            * f1_score: Harmonic mean of precision and recall based on the number of coordinates
                found at different distance cutoffs (around ground truth).
            * abs_euclidean: Average euclidean distance at each cutoff.
            * offset: List of (r, c) coordinates denoting offset in pixels.
            * f1_integral: Area under curve f1_score vs. cutoffs.
            * mean_euclidean: Normalized average euclidean distance based on the total number of assignments.
    """
    f1_scores, offsets, cutoffs = f1_integral(pred, true, mdist=mdist, n_cutoffs=50, return_raw=True)  # type: ignore

    abs_euclideans = []
    total_euclidean = 0
    total_assignments = 0

    # Find distances through offsets at every cutoff
    for c_offset in offsets:
        abs_euclideans.append(np.mean(offset_euclidean(c_offset)))
        total_euclidean += np.sum(offset_euclidean(c_offset))
        try:
            total_assignments += len(c_offset)
        except TypeError:
            continue

    df = pd.DataFrame(
        {
            "cutoff": cutoffs,
            "f1_score": f1_scores,
            "abs_euclidean": abs_euclideans,
            "offset": offsets,
        }
    )
    df["f1_integral"] = np.trapz(df["f1_score"], cutoffs) / mdist  # Norm. to 0-1
    df["mean_euclidean"] = total_euclidean / (total_assignments + 1e-10)

    return df
