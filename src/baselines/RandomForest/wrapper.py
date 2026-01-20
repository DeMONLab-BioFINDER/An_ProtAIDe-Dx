#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os
import time
import argparse
import pickle
import pandas as pd
import numpy as np
from src.utils.metrics import clf_metrics
from src.preproc.preproc_misc import stand_norm
from src.utils.io import txt2list, dict2df
from src.baselines.RandomForest.RF_model import RFClfModule
from src.preproc.split_misc import df_tr_val_test_split, df_cross_validation


def args_parser():
    """
    Parse command-line arguments for the Random Forest classifier.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(prog='RFArgs')
    # global parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prefix', type=str, default='RF_')
    
    parser.add_argument('--input_dir', type=str, default='/')
    parser.add_argument('--output_dir', type=str, default='/')
    parser.add_argument('--ckpt_dir', type=str, default='/')
    parser.add_argument('--sub_dir_prefix', type=str, default='fold_')
    
    parser.add_argument('--splitGiven', action='store_true', default=False)
    parser.add_argument('--CrossValidation', action='store_true',
                        default=False)
    parser.add_argument('--SkipExistingResults', action='store_true',
                        default=False)
    
    parser.add_argument('--nb_folds', type=int, default=10)
    parser.add_argument('--tr_portion', type=float, default=0.8)
    parser.add_argument('--val_portion', type=float, default=0.1)
    
    parser.add_argument('--tr_file', type=str, default='train.csv')
    parser.add_argument('--val_file', type=str, default='val.csv')
    parser.add_argument('--test_file', type=str, default='test.csv')

    parser.add_argument('--features_file', type=str, default='')
    parser.add_argument('--FoldSpecificFeatures',
                        action='store_true', default=False)
    parser.add_argument('--target', type=str, default='Age')

    # model parameters
    parser.add_argument('--nb_estimators', type=int, default=100)

    rf_args, _ = parser.parse_known_args() 
    return rf_args


def RF_clf(tr_df, 
           val_df,
           test_df,
           features, 
           target):
    """
    Train, tune, and test a Random Forest classifier.

    Args:
        tr_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        features (list): List of feature column names.
        target (str): Target column name.
    Returns:
        tuple: Best model, 
            best hyperparameters, performance metrics, and test predictions.
    """
    # get test target nonan mask
    test_mask = ~np.isnan(test_df[target].values.reshape((-1, )))
    nb_valid_test = np.sum(test_mask)
    
    # remove missing label rows
    tr_df.dropna(subset=[target], inplace=True)
    val_df.dropna(subset=[target], inplace=True)

    # run normalization
    cols2norm = features
    norm_tr_df, scaler = stand_norm(
        tr_df, cols2norm, None, isTrain=True)
    norm_val_df, _ = stand_norm(
        val_df, cols2norm, scaler, isTrain=False)
    norm_test_df, _ = stand_norm(
            test_df, cols2norm, scaler, isTrain=False)
    
    # fit a RF class
    # this RF class could tune hyperparameters
    predictor = RFClfModule(
        features=features,
        target=target,
        nb_est=100
    )
    best_model, best_hyper_params = predictor.tune(
        norm_tr_df, norm_val_df)
    
    # test performances 
    test_pred = predictor.predict(best_model, norm_test_df)
    if nb_valid_test > 0:
        gt = norm_test_df[target].values.reshape((-1, ))
        performance =  clf_metrics(
            gt[test_mask], 
            test_pred.reshape((-1, ))[test_mask], 2, 0.5)
    else:
        # no valid test labels
        performance = {'acc':np.nan, 'bas':np.nan, 
                       'sensitivity':np.nan, 'specificity':np.nan,
                        'precision': np.nan, 'f1':np.nan, 'auc':np.nan}
    return best_model, dict2df(best_hyper_params), \
        dict2df(performance),test_pred


def wrapper(rf_args):
    """
    Wrapper function for Random Forest classifier training and evaluation.

    Args:
        rf_args (argparse.Namespace): 
            Parsed command-line arguments for Random Forest configuration.
    """
    save_prefix = rf_args.prefix + rf_args.target + '_'
    if rf_args.CrossValidation:
        if not rf_args.splitGiven:
            # Cross validation, no given split yet
            if rf_args.data_file != '':
                df = pd.read_csv(
                    os.path.join(rf_args.input_dir, rf_args.data_file))
            else:
                # for nested-CV settings
                tr_df = pd.read_csv(
                    os.path.join(rf_args.input_dir, rf_args.tr_file))
                val_df = pd.read_csv(
                    os.path.join(rf_args.input_dir, rf_args.val_file))
                
                df = pd.concat([tr_df, val_df], ignore_index=True, axis=0)
            split = df_cross_validation(df,
                                        rf_args.target,
                                        rf_args.seed,
                                        rf_args.nb_folds)

        for fold in range(rf_args.nb_folds):
            fold_dir = rf_args.sub_dir_prefix + str(fold)
            fold_data_dir = os.path.join(rf_args.input_dir, fold_dir)
            fold_output_dir = os.path.join(rf_args.output_dir, fold_dir)
            fold_ckpt_dir = os.path.join(rf_args.ckpt_dir, fold_dir)
            if rf_args.FoldSpecificFeatures:
                features = txt2list(
                    os.path.join(fold_data_dir, rf_args.features_file))
            else:
                features = txt2list(rf_args.features_file)
            
            # whether split is given or not
            if rf_args.splitGiven:
                fold_tr_df = pd.read_csv(os.path.join(
                    fold_data_dir, rf_args.tr_file))
                fold_val_df = pd.read_csv(os.path.join(
                    fold_data_dir, rf_args.val_file))
                fold_test_df = pd.read_csv(os.path.join(
                    fold_data_dir, rf_args.test_file))
            else:
                fold_tr_df, fold_val_df, fold_test_df =\
                    split[fold]['tr'], split[fold]['val'], split[fold]['test']
                
            results_csv_path = os.path.join(
                fold_output_dir, save_prefix + 'metrics.csv')
            if rf_args.SkipExistingResults and \
                os.path.isfile(results_csv_path):
                print("Results exits! Skip......", flush=True)
            else:
                # get test set performance
                best_model, best_hyper_params, performance, test_pred = RF_clf(
                    fold_tr_df, fold_val_df, fold_test_df, features,
                    rf_args.target
                )
                print(performance, time.ctime(), flush=True)
                # save model 
                model_save_path = os.path.join(
                    fold_ckpt_dir, save_prefix + 'model.pkl')
                with open(model_save_path, 'wb') as f:
                    pickle.dump(best_model, f)
                # save best hyper parameters
                best_hyper_params.to_csv(
                    os.path.join(
                        fold_ckpt_dir, save_prefix + 'hyper-params.csv'),
                    index=False)
                # save testset performances
                performance.to_csv(results_csv_path, index=False)
                np.save(os.path.join(fold_output_dir, 
                                     save_prefix+'test_pred.npy'), test_pred)
    else:
        # for leave one site out
        features = txt2list(rf_args.features_file)
        if rf_args.splitGiven:
            tr_df = pd.read_csv(
                os.path.join(rf_args.input_dir, rf_args.tr_file))
            val_df = pd.read_csv(
                os.path.join(rf_args.input_dir, rf_args.val_file))
            test_df = pd.read_csv(
                os.path.join(rf_args.input_dir, rf_args.test_file)) 
        else:
            df = pd.read_csv(
                os.path.join(rf_args.input_dir, rf_args.data_file))
            split = df_tr_val_test_split(df, 
                                         rf_args.target, 
                                         rf_args.tr_portion,
                                         rf_args.val_portion)
            tr_df, val_df, test_df = split['tr'], split['val'], split['test']
        best_model, best_hyper_params, performance, test_pred = RF_clf(
                tr_df, val_df, test_df, features,
                rf_args.target
            )
        print(performance, time.ctime(), flush=True)
        # save model 
        model_save_path = os.path.join(
            rf_args.ckpt_dir, save_prefix + 'model.pkl')
        with open(model_save_path, 'wb') as f:
            pickle.dump(best_model, f)
        # save best hyper parameters
        best_hyper_params.to_csv(
            os.path.join(rf_args.ckpt_dir, save_prefix + 'hyper-params.csv'),
            index=False)
        # save testset performances
        results_csv_path = os.path.join(rf_args.output_dir, 
                                        save_prefix + 'metrics.csv')
        performance.to_csv(results_csv_path, index=False)
        np.save(os.path.join(rf_args.output_dir, 
                                save_prefix+'test_pred.npy'), test_pred)


if __name__ == '__main__':
    wrapper(args_parser())