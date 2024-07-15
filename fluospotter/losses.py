"""Functions to calculate training loss on batches of images."""

import torch
import torch.nn.functional as F
import torch.nn as nn
from monai.losses import DiceLoss
import warnings
from skimage.segmentation import find_boundaries
from skimage.measure import label
import pdb
import numpy as np

epsilon = 1e-5


class CompoundLoss(nn.Module):
    def __init__(self, loss1, loss2=None, alpha1=1., alpha2=0.):
        super(CompoundLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, y_pred, y_true, y_true_sk=None):
        if y_true_sk:
            l1 = self.loss1(y_pred, y_true, y_true_sk)
            l2 = self.loss2(y_pred, y_true, y_true_sk)
        else:
            l1 = self.loss1(y_pred, y_true)
            l2 = self.loss2(y_pred, y_true)
        if self.alpha2 == 0 or self.loss2 is None:
            return self.alpha1*l1
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


class CompactnessLoss(nn.Module):
    def __init__(self):
        super(CompactnessLoss, self).__init__()

    def compactness(self, mask):
        mask = (mask > 0.5).to(torch.uint8)
        mask_np = label(mask.detach().cpu().numpy(), connectivity=3)
        compactness = 0
        for l in range(np.unique(mask_np)):
            if l == 0: continue
            mask_l = (mask_np == l).astype(np.uint8)
            for z in range(mask_np.shape[0]):
                perimeter = find_boundaries(mask_l, connectivity=2, mode='inner')
                compactness += (4 * np.pi * np.sum(mask_l[z])) / (np.sum(perimeter) ** 2 + epsilon)
            compactness /= mask_l.shape[0]
        return compactness

    def forward(self, y_pred):
        compactness_value = self.compactness(y_pred)
        loss = 1 / (compactness_value + epsilon)
        return loss


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
        tprec = (torch.sum(torch.multiply(y_pred_sk, y_true)[:, 1, ...]) + self.smooth)/(torch.sum(y_pred_sk[:, 1, ...]) + self.smooth)
        tsens = (torch.sum(torch.multiply(y_true_sk, y_pred)[:, 1, ...]) + self.smooth)/(torch.sum(y_true_sk[:, 1, ...]) + self.smooth)
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
    loss_dict['compactness'] = CompactnessLoss()

    loss_dict[None] = None

    loss_fn = CompoundLoss(loss_dict[loss1], loss_dict[loss2], alpha1, alpha2)

    return loss_fn