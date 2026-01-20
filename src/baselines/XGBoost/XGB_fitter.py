#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import copy
import numpy as np
from src.baselines.XGBoost.XGB_model import XGBModule
from src.preproc.preproc_misc import stand_norm
from src.utils.io import dict2df
from src.utils.metrics import clf_metrics, reg_metrics


def eval_metric(pred, 
                gt,
                nb_class=-1,
                threshold=None,
                task='regress'):
    """
    Evaluation metrics:
        For `regress` task, gives r, mae, rho, cod;
        For `binary` & 'multi' tasks, gives 
            acc, bas, sensitivity, specificity, precision, f1, auc.

    Args:
        pred (np.ndarray): Predictions.
        gt (np.ndarray): Ground truth values.
        nb_class (int, optional): 
            Number of classes for classification tasks. Defaults to -1.
        threshold (float, optional): 
            Threshold for binary classification. Defaults to None.
        task (str, optional): 
            Task type ('regress', 'binary', 'multi'). Defaults to 'regress'.

    Returns:
        pd.DataFrame: Evaluation metrics.
    """
    if task == 'regress':
        pred, gt = pred.reshape((-1,)), gt.reshape((-1,))
        metrics = reg_metrics(gt, pred)
    else:
        gt = gt.reshape((-1, ))
        metrics = clf_metrics(gt, pred, nb_class, threshold)
    return dict2df(metrics)


def xgb_fit_infer(tr_df, 
                  val_df,
                  test_df,
                  features, 
                  target,
                  task,
                  nb_class=-1,
                  nb_boost=100):
    """
    Fit and infer, return metrics.

    Args:
        tr_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        features (list): List of feature column names.
        target (str): Target column name.
        task (str): Task type ('regress', 'binary', 'multi').
        nb_class (int, optional): 
            Number of classes for classification tasks. Defaults to -1.
        nb_boost (int, optional): 
            Number of boosting rounds. Defaults to 100.

    Returns:
        pd.DataFrame: Evaluation metrics.
    """
    # get test target nonan mask
    test_mask = ~np.isnan(test_df[target].values.reshape((-1, )))
    nb_valid_test = np.sum(test_mask)

    tr_df_copy = copy.deepcopy(tr_df)
    val_df_copy = copy.deepcopy(val_df)
    
    # remove missing lable rows
    tr_df.dropna(subset=[target], inplace=True)
    val_df.dropna(subset=[target], inplace=True)
    # z norm
    if task == 'regress':
        cols2norm = [target] + features
        tr_std = tr_df[cols2norm].std()
        tr_mean = tr_df[cols2norm].mean()
    else:
        cols2norm = features
    norm_tr_df, scaler = stand_norm(
        tr_df, cols2norm, None, isTrain=True)
    norm_val_df, _ = stand_norm(
        val_df, cols2norm, scaler, isTrain=False)
    
    # renorm for all train_df/val_df
    # for the purpose of making all predictions for all tr/val samples
    tr_df_copy_norm, _ = stand_norm(
        tr_df_copy, cols2norm, scaler, isTrain=False)
    val_df_copy_norm, _ = stand_norm(
        val_df_copy, cols2norm, scaler, isTrain=False)    

    if task == 'regress':
        predictor = XGBModule(
            features=features, target=target,
            task=task, nb_boost=nb_boost)
    else:
        predictor = XGBModule(
            features=features, target=target,
            task=task, nb_boost=nb_boost, nb_class=nb_class)
    # tune by grid search
    best_model, best_hyper_params =\
        predictor.tune(norm_tr_df, norm_val_df)
    importance = best_model.get_score(importance_type='total_gain')
    

    # testset performance
    norm_test_df, _ = stand_norm(
        test_df, cols2norm, scaler, isTrain=False)
    test_pred = predictor.predict(best_model, norm_test_df)
    train_pred = predictor.predict(best_model, tr_df_copy_norm)
    val_pred = predictor.predict(best_model, val_df_copy_norm)

    if nb_valid_test > 0:
        if task == 'regress':
            train_pred = train_pred * tr_std[target] + tr_mean[target]
            val_pred = val_pred * tr_std[target] + tr_mean[target]
            test_pred = test_pred * tr_std[target] + tr_mean[target]
            if target == 'MMSE':
                test_pred = np.clip(test_pred, 0, 30)
                val_pred = np.clip(val_pred, 0, 30)
                train_pred = np.clip(train_pred, 0, 30)
            gt = (norm_test_df[target].values) * tr_std[target] + \
                tr_mean[target]
            performance = eval_metric(test_pred[test_mask], 
                                      gt[test_mask], 
                                      nb_class, None, task='regress')
        elif task == 'binary':
            gt = norm_test_df[target].values
            performance = eval_metric(test_pred[test_mask], 
                                      gt[test_mask], 
                                      nb_class, 
                                      best_hyper_params['threshold'],
                                      task='binary')
        else:
            gt = norm_test_df[target].values
            performance = eval_metric(test_pred[test_mask], 
                                      gt[test_mask], 
                                      nb_class,
                                      None,
                                      task='multi')
    else:
        if task == 'regress':
            performance = {'mae':np.nan, 'r':np.nan,
                           'rho':np.nan, 'cod':np.nan}
        else:
            performance = {'acc':np.nan, 'bas':np.nan, 
                           'sensitivity':np.nan, 'specificity':np.nan,
                           'precision': np.nan, 'f1':np.nan, 'auc':np.nan}
        performance = dict2df(performance)
    return best_model, dict2df(best_hyper_params), performance, \
        importance, train_pred, val_pred, test_pred