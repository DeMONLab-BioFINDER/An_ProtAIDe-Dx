#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy import stats

import warnings
warnings.filterwarnings('ignore')


def clf_acc(gt_vec, pred_vec):
    """
    Calculate accuracy score

    Args:
        gt_vec (np.ndarray): Ground truth labels
        pred_vec (np.ndarray): Predicted labels
    Returns:
        float: Accuracy score
    """
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    return accuracy_score(gt_vec, pred_vec)

def clf_bas(gt_vec, pred_vec):
    """
    Calculate balanced accuracy score 

    Args:
        gt_vec (np.ndarray): Ground truth labels
        pred_vec (np.ndarray): Predicted labels
    Returns:
        float: Balanced accuracy score
    """
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    return balanced_accuracy_score(gt_vec, pred_vec)


def clf_recall(gt_vec, pred_vec):
    """
    Calculate recall: TP/(TP + FN). 

    Args:
        gt_vec (np.ndarray): Ground truth labels
        pred_vec (np.ndarray): Predicted labels
    Returns:
        float: Recall score
    """
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    if np.sum(gt_vec[~np.isnan(gt_vec)]) == 0:
        return np.nan
    return recall_score(gt_vec, pred_vec)


def clf_specificity(gt_vec, pred_vec):
    """
    Calculate specificity: TN/(TN + FP). 

    Args:
        gt_vec (np.ndarray): Ground truth labels
        pred_vec (np.ndarray): Predicted labels
    Returns:
        float: Specificity score
    """
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    if np.sum(1 - gt_vec[~np.isnan(gt_vec)]) == 0:
        return np.nan
    return recall_score(np.logical_not(gt_vec), np.logical_not(pred_vec))


def clf_precision(gt_vec, pred_vec):
    """
    Calculate precision: TP/(TP + FP). 

    Args:
        gt_vec (np.ndarray): Ground truth labels
        pred_vec (np.ndarray): Predicted labels
    Returns:
        float: Precision score
    """
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    return precision_score(gt_vec, pred_vec)


def clf_f1(gt_vec, pred_vec):
    """
    Calculate F1:  2 * TP/(2*TP + FP + FN)

    Args:
        gt_vec (np.ndarray): Ground truth labels
        pred_vec (np.ndarray): Predicted labels
    Returns:
        float: F1 score
    """
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    return f1_score(gt_vec, pred_vec)


def clf_auc(gt_vec, pred_prob,nb_class=2):
    """
    Compute AUC under ROC.

    Args:
        gt_vec (np.ndarray): Ground truth labels
        pred_prob (np.ndarray): Predicted probabilities
        nb_class (int, optional): Number of classes. Defaults to 2.

    Returns:
        float: AUC score
    """
    if nb_class > 2:
        assert pred_prob.shape[1] == nb_class, "Wrong shape"
    if len(np.unique(gt_vec)) == 1:
        return np.nan
    return roc_auc_score(gt_vec, pred_prob)


def prob2label(pred_prob, threshold=None):
    """
    Transfer probability prediction to labels.

    Args:
        pred_prob (np.ndarray): Predicted probabilities
        threshold (float, optional): 
            Threshold for binary classification. Defaults to None.
    Returns:
        np.ndarray: Predicted labels
    """
    if threshold is None:
        # multi-class
        assert pred_prob.shape[1] > 2, "Binary pred prob needs threshold!"
        return pred_prob.argmax(axis=1)
    else:
        # binary
        return (pred_prob >= threshold) * 1


def clf_metrics(gt_vec, pred_prob, nb_class=2, threshold=None):
    """
    Compute classification metrics and return a dictionary.

    Args:
        gt_vec (np.ndarray): Ground truth labels
        pred_prob (np.ndarray): Predicted probabilities
        nb_class (int, optional): Number of classes. Defaults to 2.
        threshold (float, optional): 
            Threshold for binary classification. Defaults to None.

    Returns:
        dict: Dictionary of classification metrics
    """
    if nb_class > 2:
        assert pred_prob.shape[1] == nb_class, "Wrong shape"
    if len(gt_vec) > 0:
        pred_label = prob2label(pred_prob, threshold)
        clf_met = dict()
        clf_met['auc'] = clf_auc(gt_vec, pred_prob, nb_class)
        clf_met['bas'] = clf_bas(gt_vec, pred_label)
    else:
        clf_met = {'auc':np.nan, 'bas':np.nan} 
    return clf_met


def reg_mae(gt_vec, pred_vec):
    """
    Compute mean absolute error (MAE). 

    Args:
        gt_vec (np.ndarray): Ground truth values
        pred_vec (np.ndarray): Predicted values
    Returns:
        float: Mean absolute error
    """
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    return mean_absolute_error(gt_vec, pred_vec)


def pearson_r(gt_vec, pred_vec):
    """
    Compute pearson correlation.

    Args:
        gt_vec (np.ndarray): Ground truth values
        pred_vec (np.ndarray): Predicted values
    Returns:
        tuple: Pearson correlation coefficient and p-value
    """
    assert gt_vec.ndim == 1, "1D vector expected"
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    r, pval = stats.pearsonr(gt_vec, pred_vec)
    return r, pval


def spearman_rho(gt_vec, pred_vec):
    """
    Compute spearman rho.

    Args:
        gt_vec (np.ndarray): Ground truth values
        pred_vec (np.ndarray): Predicted values
    Returns:
        tuple: Spearman correlation coefficient and p-value
    """
    assert gt_vec.ndim == 1, "1D vector expected"
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    rho, pval = stats.spearmanr(gt_vec, pred_vec)
    return rho, pval


def cod(gt_vec, pred_vec):
    """
    Compute coefficient of determination (R^2 score)

    Args:
        gt_vec (np.ndarray): Ground truth values
        pred_vec (np.ndarray): Predicted values
    Returns:
        float: Coefficient of determination (R^2 score)
    """
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    return r2_score(gt_vec, pred_vec)


def reg_metrics(gt_vec, pred_vec):
    """
    Compute all regression metrics and return a dictionary.

    Args:
        gt_vec (np.ndarray): Ground truth values
        pred_vec (np.ndarray): Predicted values
    Returns:
        dict: Dictionary of regression metrics
    """
    assert pred_vec.shape == gt_vec.shape, "Unequal shape"
    reg_met = dict()
    reg_met['mae'] = reg_mae(gt_vec, pred_vec)
    reg_met['r'], _ = pearson_r(gt_vec, pred_vec)
    reg_met['rho'], _ = spearman_rho(gt_vec, pred_vec)
    reg_met['cod'] = cod(gt_vec, pred_vec)
    return reg_met


def get_opt_threshold_PRC(gt_vec, pred_prob):
    """
    Find optimal threshold based on precision-recall curve.

    Args:
        gt_vec (np.ndarray): Ground truth labels
        pred_prob (np.ndarray): Predicted probabilities
    Returns:
        float: Optimal threshold based on precision-recall curve
    """
    # 1. compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(gt_vec, pred_prob)
    # 2. compute f1 score
    f1_score = (2 * precision * recall) / (precision + recall + 1e-12)
    idx = np.argmax(f1_score)
    return thresholds[idx]


def clf_multi_target_metric(gt_vec, pred_prob, mask,
                            threshold_func='get_opt_threshold_PRC', 
                            metric='clf_bas'):
    """
    Compute single metrics for multi target classification. 

    Args:
        gt_vec (np.ndarray): Ground truth labels
        pred_prob (np.ndarray): Predicted probabilities
        mask (np.ndarray): Mask indicating valid entries
        threshold_func (str, optional): 
            Function name to compute optimal threshold. 
            Defaults to 'get_opt_threshold_PRC'.
        metric (str, optional): 
            Metric function name to compute classification metric. 
            Defaults to 'clf_bas'.

    Returns:
        tuple: A tuple containing a list of computed metrics 
            and a list of optimal thresholds
    """
    metrics = []
    opt_thresholds = []
    for i in range(gt_vec.shape[1]):
        gt_i = gt_vec[:, i][mask[:, i]]
        pred_prob_i = pred_prob[:, i][mask[:, i]]
        if len(gt_i) > 0:
            opt_threshold = globals()[threshold_func](gt_i, pred_prob_i)
            pred_i = prob2label(pred_prob_i, opt_threshold)
            metric_i = globals()[metric](gt_i, pred_i)
            metrics.append(metric_i)
            opt_thresholds.append(opt_threshold)
        else:
            metrics.append(np.nan)
            opt_thresholds.append(np.nan)
    return metrics, opt_thresholds