"""Functions to calculate training loss on batches of images.

While functions are comparable to the ones found in the module metrics,
these rely on keras' backend and do not take raw numpy as input.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from monai.losses import DiceLoss
import warnings
import pdb


class CompoundLoss(nn.Module):
    def __init__(self, loss1, loss2=None, alpha1=1., alpha2=0.):
        super(CompoundLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, y_pred, y_true, y_true_sk=None):
        if y_true_sk:
            l1 = self.loss1(y_pred[0,1], y_true[0,1], y_true_sk)
            l2 = self.loss2(y_pred[0,2:], y_true[0,2:], y_true_sk)
        elif y_pred.shape[1] > 2:
            l1 = self.loss1(y_pred[0,1], y_true[0,1])
            l2 = self.loss2(y_pred[0,2:], y_true[0,2:])
        else:
            l1 = self.loss1(y_pred[0,1], y_true[0,1])
            l2 = self.loss2(y_pred[0,1], y_true[0,1])
        return self.alpha1*l1 + self.alpha2 * l2


class DSCLoss(nn.Module):
    def __init__(self, include_background=False):
        super(DSCLoss, self).__init__()
        self.loss = DiceLoss(softmax=True, include_background=include_background)
        self.check_softmax = True

    def forward(self, y_pred, y_true):
        if self.check_softmax:
            if y_pred.softmax(dim=1).flatten(2).sum(dim=1).mean(dim=1).mean() != 1.0:
                # flatten(2) flattens all after dim=2, sum over classes, take mean to get per-batch values, & take mean
                warnings.warn('check you did not apply softmax before loss computation')
            else: self.check_softmax = False
        return self.loss(y_pred, y_true)


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1),(1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img,(3, 3, 3),(1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img-img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img-img1)
        skel = skel + F.relu(delta-skel*delta)
    return skel


class clDiceLoss(torch.nn.Module):
    def __init__(self, iters=3, smooth=1., include_background=False):
        super(clDiceLoss, self).__init__()
        self.iters = iters
        self.smooth = smooth
        self.include_background = include_background
        self.check_softmax = True

    def forward(self, y_pred, y_true):
        y_pred = y_pred.softmax(dim=1)
        y_pred_sk = soft_skel(y_pred, self.iters)
        y_true_sk = soft_skel(y_true, self.iters)
        tprec = (torch.sum(torch.multiply(y_pred_sk, y_true)[:, 1:, ...]) + self.smooth)/(torch.sum(y_pred_sk[:, 1:, ...]) + self.smooth)
        tsens = (torch.sum(torch.multiply(y_true_sk, y_pred)[:, 1:, ...]) + self.smooth)/(torch.sum(y_true_sk[:, 1:, ...]) + self.smooth)
        cl_dice = 1. - 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


class clCELoss(torch.nn.Module):
    def __init__(self, iters=3, smooth=1., include_background=False):
        super(clCELoss, self).__init__()
        self.iters = iters
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, y_pred, y_true):
        l_unred = torch.nn.functional.cross_entropy(y_pred, y_true, reduction='none')
        y_pred = y_pred.softmax(dim=1)
        y_pred_sk = soft_skel(y_pred, self.iters)
        y_true_sk = soft_skel(y_true, self.iters)

        tprec = torch.mul(l_unred, y_true_sk[:, 1]).mean()
        tsens = torch.mul(l_unred, y_pred_sk[:, 1]).mean()
        cl_ce = (tprec+tsens)
        return cl_ce


def get_loss(loss1, loss2=None, alpha1=1., alpha2=0.):
    if loss1 == loss2 and alpha2 != 0.:
        warnings.warn('using same loss twice, you sure?')
    loss_dict = dict()
    loss_dict['ce'] = CELoss()
    loss_dict['dice'] = DSCLoss()
    loss_dict['cedice'] = CompoundLoss(CELoss(), DSCLoss(), alpha1=1., alpha2=1.)
    loss_dict['cldice'] = clDiceLoss()
    loss_dict['clce'] = clCELoss()

    loss_dict[None] = None

    loss_fn = CompoundLoss(loss_dict[loss1], loss_dict[loss2], alpha1, alpha2)

    return loss_fn


def dice_score(y_true, y_pred, smooth: int = 1):
    r"""Computes the dice coefficient on a batch of tensors.

    .. math::
        \textrm{Dice} = \frac{2 * {\lvert X \cup Y\rvert}}{\lvert X\rvert +\lvert Y\rvert}


    ref: https://arxiv.org/pdf/1606.04797v1.pdf

    Args:
        y_true: Ground truth masks.
        y_pred: Predicted masks.
        smooth: Epslion value to avoid division by zero.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    """Dice score loss corresponding to deepblink.losses.dice_score."""
    return 1 - dice_score(y_true, y_pred)


def recall_score(y_true, y_pred):
    """Recall score metric.

    Defined as ``tp / (tp + fn)`` where tp is the number of true positives and fn the number of false negatives.
    Can be interpreted as the accuracy of finding positive samples or how many relevant samples were selected.
    The best value is 1 and the worst value is 0.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_score(y_true, y_pred):
    """Precision score metric.

    Defined as ``tp / (tp + fp)`` where tp is the number of true positives and fp the number of false positives.
    Can be interpreted as the accuracy to not mislabel samples or how many selected items are relevant.
    The best value is 1 and the worst value is 0.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    r"""F1 score metric.

    .. math::
        F1 = \frac{2 * \textrm{precision} * \textrm{recall}}{\textrm{precision} + \textrm{recall}}

    The equally weighted average of precision and recall.
    The best value is 1 and the worst value is 0.
    """
    # Do not move outside of function. See RMSE.
    precision = precision_score(y_true[..., 0], y_pred[..., 0])
    recall = recall_score(y_true[..., 0], y_pred[..., 0])
    f1_value = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1_value


def f1_loss(y_true, y_pred):
    """F1 score loss corresponding to deepblink.losses.f1_score."""
    if not (
        y_true.ndim == y_pred.ndim == 3 and y_true.shape[2] == y_pred.shape[2] == 3
    ):
        raise ValueError(
            f"Tensors must have shape n*n*3. Tensors has shape y_true:{y_true.shape}, y_pred:{y_pred.shape}."
        )
    return 1 - f1_score(y_true, y_pred)


def rmse(y_true, y_pred):
    """Calculate root mean square error (rmse) between true and predicted coordinates."""
    # RMSE, takes in the full y_true/y_pred when used as metric.
    # Therefore, do not move the selection outside the function.
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    comparison = tf.equal(y_true, tf.constant(0, dtype=tf.float32))

    y_true_new = tf.where(comparison, tf.zeros_like(y_true), y_true)
    y_pred_new = tf.where(comparison, tf.zeros_like(y_pred), y_pred)

    sum_rc_coords = K.sum(y_true, axis=-1)
    n_true_spots = tf.math.count_nonzero(sum_rc_coords, dtype=tf.float32)

    squared_displacement_xy_summed = K.sum(K.square(y_true_new - y_pred_new), axis=-1)
    rmse_value = K.sqrt(
        K.sum(squared_displacement_xy_summed) / (n_true_spots + K.epsilon())
    )

    return rmse_value


def combined_f1_rmse(y_true, y_pred):
    """Difference between F1 score and root mean square error (rmse).

    The optimal values for F1 score and rmse are 1 and 0 respectively.
    Therefore, the combined optimal value is 1.
    """
    return f1_score(y_true, y_pred) - rmse(y_true, y_pred)


def combined_bce_rmse(y_true, y_pred):
    """Loss that combines binary cross entropy for probability and rmse for coordinates.

    The optimal values for binary crossentropy (bce) and rmse are both 0.
    """
    return (
        binary_crossentropy(y_true[..., 0], y_pred[..., 0]) + rmse(y_true, y_pred) * 2
    )


def combined_dice_rmse(y_true, y_pred):
    """Loss that combines dice for probability and rmse for coordinates.

    The optimal values for dice and rmse are both 0.
    """
    return dice_loss(y_true[..., 0], y_pred[..., 0]) + rmse(y_true, y_pred) * 2
