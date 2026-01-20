#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def rank_loss(y_gt,
              y_pred,
              mask):
    """
    Rank loss for multi-task classification.
    Reference: https://www.nature.com/articles/s41591-024-03118-z   

    Args:
        y_gt (torch.Tensor): Ground truth labels
        y_pred (torch.Tensor): Predicted logits
        mask (torch.Tensor): Mask tensor indicating valid entries

    Returns:
        torch.Tensor: Computed rank loss
    """
    assert y_pred.shape == y_gt.shape, "Shape should be same"
    assert y_pred.shape[1] > 2, "rank loss should be at least 2 categories"
    # convert to probability 
    pred_prob = F.sigmoid(y_pred)
    loss = 0 
    for i in range(pred_prob.shape[1] - 1):
        prob_i = pred_prob[:,  i]
        gt_i = y_gt[:, i]
        mask_i = mask[:, i].reshape((-1, ))
        for j in range(i + 1, pred_prob.shape[1]):
            prob_j = pred_prob[:,  j]
            gt_j = y_gt[:, j]
            mask_j = mask[:, j].reshape((-1, ))
            # compute mask
            case_mask = torch.logical_and(mask_i, mask_j)
            loss += F.margin_ranking_loss(prob_i[case_mask], 
                                        prob_j[case_mask], 
                                        gt_i[case_mask] - gt_j[case_mask],
                                        margin=0.25)
    return loss


def bce_loss(args,
             y_gt,
             y_pred,
             mask):
    """
    Binary cross entropy loss. 

    Args:
        args (argparse.Namespace): Arguments containing label smoothing values
        y_gt (torch.Tensor): Ground truth labels
        y_pred (torch.Tensor): Predicted logits
        mask (torch.Tensor): Mask tensor indicating valid entries

    Returns:
        torch.Tensor: Computed binary cross entropy loss
    """
    assert y_pred.shape == y_gt.shape, "Shape should be same"
    loss = 0
    for i in range(y_gt.shape[1]):
        mask_i = mask[:, i].reshape((-1,))
        valid_targets = y_gt[:, i][mask_i]
        valid_targets = label_smoothing(valid_targets, args.label_smooth[i], 
                                        K=2)
        loss += F.binary_cross_entropy_with_logits(
            y_pred[:, i][mask_i],
            valid_targets
        )
    return loss


def label_smoothing(targets,
                    alpha, 
                    K=2):
    """
    Label smoothing to increase generalization

    Args:
        targets (torch.Tensor): Ground truth labels
        alpha (float): Smoothing factor
        K (int, optional): Number of classes. Defaults to 2.

    Returns:
        torch.Tensor: Smoothed labels
    """
    return (1 - alpha) * targets + alpha / K

