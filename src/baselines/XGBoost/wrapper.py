#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os
import time
import argparse
import pandas as pd
import numpy as np
from src.preproc.split_misc import df_tr_val_test_split, df_cross_validation
from src.utils.io import txt2list, save_pkl
from src.baselines.XGBoost.XGB_fitter import xgb_fit_infer


def args_parser():
    """
    Parse command-line arguments for the XGBoost classifier.

    Returns:
        argparse.Namespace: 
            Parsed command-line arguments for XGBoost configuration.
    """
    parser = argparse.ArgumentParser(prog='XGBArgs')
    # global parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prefix', type=str, default='XGB_')
    
    parser.add_argument('--input_dir', type=str, default='/')
    parser.add_argument('--output_dir', type=str, default='/')
    parser.add_argument('--ckpt_dir', type=str, default='/')
    parser.add_argument('--sub_dir_prefix', type=str, default='fold_')
    
    parser.add_argument('--splitGiven', action='store_true', default=False)
    parser.add_argument('--saveTrValPred', action='store_true', default=False)
    parser.add_argument('--CrossValidation', action='store_true',
                        default=False)
    parser.add_argument('--SkipExistingResults', action='store_true',
                        default=False)
    
    parser.add_argument('--nb_folds', type=int, default=10)
    parser.add_argument('--tr_portion', type=float, default=0.8)
    parser.add_argument('--val_portion', type=float, default=0.1)


    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--tr_file', type=str, default='train.csv')
    parser.add_argument('--val_file', type=str, default='val.csv')
    parser.add_argument('--test_file', type=str, default='test.csv')
    

    parser.add_argument('--features_file', type=str, default='')
    parser.add_argument('--FoldSpecificFeatures',
                        action='store_true', default=False)
    parser.add_argument('--target', type=str, default='Age')
    # model parameters
    parser.add_argument('--nb_boost', type=int, default=100)

    xgb_args, _ = parser.parse_known_args() 
    return xgb_args



def wrapper(xgb_args):
    """
    Wrapper function for XGBoost pipeline.

    Args:
        xgb_args (argparse.Namespace): 
            Parsed command-line arguments for XGBoost configuration.
    """
    task_dict = {'Recruited Control': 'binary', 
                 'CU': 'binary',
                 'AD': 'binary',
                 'PD': 'binary', 
                 'FTD': 'binary', 
                 'ALS': 'binary',
                 'StrokeTIA': 'binary', 
                 'MCI-SCI': 'binary',
                 'MMSE': 'regress'}
    classes_dict = {'Recruited Control': 2, 
                    'CU': 2,
                    'AD': 2,
                    'PD': 2, 
                    'FTD': 2, 
                    'ALS': 2,
                    'StrokeTIA': 2, 
                    'MCI-SCI': 2,
                    'MMSE': -1}
    
    save_prefix = xgb_args.prefix + xgb_args.target + '_'
    if xgb_args.CrossValidation:
        if not xgb_args.splitGiven:
            # Cross validation, no given split yet
            if xgb_args.data_file != '':
                df = pd.read_csv(
                    os.path.join(xgb_args.input_dir, xgb_args.data_file))
            else:
                # for nested-CV settings
                tr_df = pd.read_csv(
                    os.path.join(xgb_args.input_dir, xgb_args.tr_file))
                val_df = pd.read_csv(
                    os.path.join(xgb_args.input_dir, xgb_args.val_file))
                
                df = pd.concat([tr_df, val_df], ignore_index=True, axis=0)
            split = df_cross_validation(df,
                                        xgb_args.target,
                                        xgb_args.seed,
                                        xgb_args.nb_folds)

        for fold in range(xgb_args.nb_folds):
            fold_dir = xgb_args.sub_dir_prefix + str(fold)
            fold_data_dir = os.path.join(xgb_args.input_dir, fold_dir)
            fold_output_dir = os.path.join(xgb_args.output_dir, fold_dir)
            fold_ckpt_dir = os.path.join(xgb_args.ckpt_dir, fold_dir)
            
            if xgb_args.FoldSpecificFeatures:
                features = txt2list(
                    os.path.join(fold_data_dir, xgb_args.features_file))
            else:
                features = txt2list(xgb_args.features_file)
            
            # whether split is given or not
            if xgb_args.splitGiven:
                fold_tr_df = pd.read_csv(os.path.join(
                    fold_data_dir, xgb_args.tr_file))
                fold_val_df = pd.read_csv(os.path.join(
                    fold_data_dir, xgb_args.val_file))
                fold_test_df = pd.read_csv(os.path.join(
                    fold_data_dir, xgb_args.test_file))
            else:
                fold_tr_df, fold_val_df, fold_test_df =\
                    split[fold]['tr'], split[fold]['val'], split[fold]['test']
                
            results_csv_path = os.path.join(
                fold_output_dir, save_prefix + 'metrics.csv')

            if xgb_args.SkipExistingResults and os.path.isfile(
                    results_csv_path):
                print("Results exits! Skip......", flush=True)
            else:
                # get test set performance & feature importance
                best_model, best_hyper_params, performance, \
                    importance, train_pred, val_pred, test_pred  = \
                    xgb_fit_infer(fold_tr_df, 
                                fold_val_df,
                                fold_test_df,
                                features,
                                xgb_args.target,
                                task_dict[xgb_args.target],
                                classes_dict[xgb_args.target],
                                xgb_args.nb_boost)
                print(performance, time.ctime(), flush=True)
                best_model.save_model(
                    os.path.join(fold_ckpt_dir, save_prefix + 'model.json'))
                best_hyper_params.to_csv(
                    os.path.join(fold_ckpt_dir, 
                                 save_prefix + 'hyper-params.csv'),
                    index=False)
                save_pkl(importance,
                         os.path.join(
                             fold_ckpt_dir, save_prefix+'importance.pkl'))
                # save as results
                performance.to_csv(results_csv_path,
                                    index=False)
                np.save(
                    os.path.join(
                        fold_output_dir,
                        save_prefix+'test_pred.npy'), test_pred)

                if xgb_args.saveTrValPred:
                    # we need to save Tra and validation prediction results 
                    np.save(
                        os.path.join(
                            fold_output_dir,
                            save_prefix+'train_pred.npy'), train_pred)
                    np.save(
                        os.path.join(
                            fold_output_dir,
                            save_prefix+'val_pred.npy'), val_pred)
             
    else:
        features = txt2list(xgb_args.features_file)
        if xgb_args.splitGiven:
            # just normal tr/val/test split
            tr_df = pd.read_csv(
                os.path.join(xgb_args.input_dir, xgb_args.tr_file))
            val_df = pd.read_csv(
                os.path.join(xgb_args.input_dir, xgb_args.val_file))
            test_df = pd.read_csv(
                os.path.join(xgb_args.input_dir, xgb_args.test_file))                     
        else:
            df = pd.read_csv(
                os.path.join(xgb_args.input_dir, xgb_args.data_file))
            split = df_tr_val_test_split(df, 
                                         xgb_args.target, 
                                         xgb_args.tr_portion,
                                         xgb_args.val_portion)
            tr_df, val_df, test_df = split['tr'], split['val'], split['test']

        # get test set performance & feature importance
        best_model, best_hyper_params, performance, \
            importance, train_pred, val_pred, test_pred  = \
            xgb_fit_infer(tr_df, 
                          val_df,
                          test_df,
                          features,
                          xgb_args.target,
                          task_dict[xgb_args.target],
                          classes_dict[xgb_args.target],
                          xgb_args.nb_boost)
        print(performance, time.ctime(), flush=True)
        best_model.save_model(
            os.path.join(xgb_args.ckpt_dir, save_prefix + 'model.json'))
        best_hyper_params.to_csv(
            os.path.join(xgb_args.ckpt_dir, save_prefix + 'hyper-params.csv'),
            index=False)
        save_pkl(importance,
                 os.path.join(xgb_args.ckpt_dir, save_prefix+'importance.pkl'))
        # save as results
        performance.to_csv(
            os.path.join(xgb_args.output_dir, save_prefix + 'metrics.csv'),
            index=False)
        np.save(
            os.path.join(xgb_args.output_dir, save_prefix+'test_pred.npy'),
            test_pred)

        if xgb_args.saveTrValPred:
            # we need to save Tra and validation prediction results 
            np.save(
                os.path.join(
                    xgb_args.output_dir, save_prefix+'train_pred.npy'),
                train_pred)
            np.save(
                os.path.join(xgb_args.output_dir, save_prefix+'val_pred.npy'),
                val_pred)


if __name__ == '__main__':
    wrapper(args_parser())