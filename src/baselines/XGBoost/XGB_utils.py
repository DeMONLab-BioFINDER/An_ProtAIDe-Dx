#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import copy
import itertools
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
from src.utils.metrics import prob2label

import warnings
warnings.filterwarnings('ignore')


def gen_xgb_input(df, 
                  features, 
                  target=None):
    """
    Generate input file for XGBoost models.

    Args:
        df (pd.DataFrame): Input data frame
        features (list): List of feature names
        target (str, optional): Target variable name. Defaults to None.

    Returns:
        xgb.DMatrix: XGBoost DMatrix object
    """
    if target is None:
        x = df[features].values
        return xgb.DMatrix(x, feature_names=features)
    
    else:
        x, y = df[features].values, df[target].values
        return xgb.DMatrix(x, y, feature_names=features)


def model_fit(xg_train, 
              model_params, 
              nb_boosts=100):
    """
    Fit a XGBoost model and infer on validation set.

    Args:
        xg_train (xgb.DMatrix): Input for train
        model_params (Dict): Hyperparameters
        nb_boosts (Int): Number of boosting iterations

    Returns:
        xgb.Booster: Fitted XGBoost model
    """
    if 'threshold' in model_params.keys():
        # binary classifier
        xgb_model_params = copy.deepcopy(model_params)
        del xgb_model_params['threshold']
        # fit model
        model = xgb.train(
            params=xgb_model_params,
            dtrain=xg_train,
            num_boost_round=nb_boosts,
            verbose_eval=False)
    else:
        # fit model
        model = xgb.train(
            params=model_params,
            dtrain=xg_train,
            num_boost_round=nb_boosts,
            verbose_eval=False)

    return model


def model_inference(model,
                    df,
                    features):
    """
    Inference with trained XGBoost model.

    Args:
        model (xgb.Booster): Trained XGBoost model
        df (pd.DataFrame): Input data frame
        features (list): Input features

    Returns:
        np.ndarray: Model predictions
    """
    return model.predict(gen_xgb_input(df, features))


def grid_search(train_df, 
                val_df, 
                features, 
                target, 
                params, 
                nb_boosts, 
                task,
                search_range):
    """
    Grid search on validation set for best performance.

    Args:
        train_df (pd.DataFrame): Input for train
        val_df (pd.DataFrame): Input for validation
        features (list): Input features
        target (str): Target variable
        params (dict): Hyperparameters & search range
        nb_boosts (int): Number of boosting iterations
        task (str): Binary, Multi or Regress
        search_range (dict): Search range for hyperparameters

    Returns:
        xgb.Booster: Best fitted XGBoost model
        dict: Best hyperparameters
    """
    # generate input for XGBoost model
    xg_train = gen_xgb_input(train_df, features, target)
    y_val = val_df[target].values

    # get initial performance with default hyper-parameters
    best_model = model_fit(xg_train, params, nb_boosts)
    val_pred = model_inference(best_model, val_df, features)
    if task == 'binary':
        best_score = metric_func(
            prob2label(val_pred, params['threshold']), y_val, task)
    elif task == 'multi':
        best_score = metric_func(prob2label(val_pred), y_val, task)
    else:
        best_score = metric_func(val_pred, y_val, task)

    # grid search for hyper-parameters
    def get_hyper_params_grid(search_range):
        """
        Get a grid combination for hyper-parameters.

        Args:
            search_range (dict): Search range for hyperparameters

        Returns:
            list: List of hyperparameter combinations
            tuple: Keys of the hyperparameters
        """
        keys, values = zip(*search_range.items())
        grids = [dict(zip(keys, v)) for v in itertools.product(*values)]

        return grids, keys

    # note that threshold is a hyper-params no need for train
    if task == 'binary':
        search_range_ = copy.deepcopy(search_range)
        del search_range['threshold']
    hyper_params_combs, hyper_params = get_hyper_params_grid(search_range)
    
    # initialize best hyper-parameters
    best_hyperparams = dict()
    for hyper_parma in hyper_params:
        best_hyperparams[hyper_parma] = params[hyper_parma]
    best_hyperparams['threshold'] = 0.5

    # grid search
    for hyper_params_comb in hyper_params_combs:
        for hyper_parma in hyper_params:
            params[hyper_parma] = hyper_params_comb[hyper_parma]
        # get prediction
        model = model_fit(xg_train, params, nb_boosts)
        val_pred = model_inference(model, val_df, features)
        if task == 'binary':
            best_threshold_score = 0
            best_threshold = search_range_['threshold'][0]
            for threshold in search_range_['threshold']:
                hyper_params_comb['threshold'] = threshold
                score = metric_func(
                    y_val, prob2label(val_pred, 
                                      hyper_params_comb['threshold']),
                                      task)
                if score > best_threshold_score:
                    best_threshold_score = score
                    best_threshold = threshold
                if best_threshold_score > best_score:
                    best_score = best_threshold_score
                    best_model = model
                    best_hyperparams = copy.deepcopy(hyper_params_comb)
                    best_hyperparams['threshold'] = best_threshold
            
        else:
            if task == 'multi':
                score = metric_func(y_val, prob2label(val_pred), task)
            else:
                score = metric_func(y_val, val_pred,  task)
            if score > best_score:
                best_score = score
                best_model = model
                best_hyperparams = copy.deepcopy(hyper_params_comb)

    return best_model, best_hyperparams


def metric_func(gt, pred, task='regress'):
    """
    Metrics for evaluation prediction performances, larger equals better.

    Args:
        gt (array-like): Ground truth values
        pred (array-like): Predicted values
        task (str, optional): 
            Task type ('regress', 'binary', 'multi'). Defaults to 'regress'.

    Returns:
        float: Evaluation metric score
        nd.array: Model predictions
    """
    if task == 'regress':
        return -1 * mean_squared_error(
            gt.reshape((-1, 1)), pred.reshape((-1, 1)))
    else:
        return accuracy_score(gt.reshape((-1, 1)), pred.reshape((-1, 1)))


def construct_model_params(hyper_params, 
                           task='regress', 
                           nb_class=1):
    """
    Construct model params by feeding hyper_parameters.

    Args:
        hyper_params (dict): Hyper-parameters for the model
        task (str, optional): 
            Task type ('regress', 'binary', 'multi'). Defaults to 'regress'.
        nb_class (int, optional): 
            Number of classes for multi-class classification. Defaults to 1.

    Returns:
        dict: Constructed model parameters
    """
    if hyper_params is None:
        model_params = {
            'booster': 'gbtree',
            'nthread': 1,
            'max_depth': 6,
            'subsample': 1.0,
        }
        if task == 'regress':
            model_params['objective'] = 'reg:squarederror'
            model_params['eval_metric'] = 'rmse'
        elif task == 'multi':
            model_params['objective'] = 'multi:softprob'
            model_params['eval_metric'] = 'merror'
            model_params['num_class'] = nb_class
        else:
            model_params['objective'] = 'binary:logistic'
            model_params['eval_metric'] = 'error'
            model_params['threshold'] = 0.5

        return model_params
    else:
        # with hyper-parameters
        model_params = {'booster': 'gbtree', 'nthread': 1}
        for key in list(hyper_params.keys()):
            model_params[key] = hyper_params[key]
        if task == 'regress':
            model_params['objective'] = 'reg:squarederror'
            model_params['eval_metric'] = 'rmse'
        elif task == 'multi':
            model_params['objective'] = 'multi:softprob'
            model_params['eval_metric'] = 'merror'
            model_params['num_class'] = nb_class
        else:
            model_params['objective'] = 'binary:logistic'
            model_params['eval_metric'] = 'error'

        return model_params