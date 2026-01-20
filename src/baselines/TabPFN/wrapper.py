#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os
import time
import copy
import argparse 
import pandas as pd
import numpy as np 
from tabpfn import TabPFNClassifier
from tabpfn.model.loading import save_fitted_tabpfn_model
from src.preproc.preproc_misc import avgProt_norm, gauss_norm, knn_impute
from src.utils.io import txt2list, list2txt, dict2df
from src.utils.metrics import clf_metrics, get_opt_threshold_PRC


def args_parser():
    """
    Parse command-line arguments for the TabPFN classifier.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(prog='TabPFNArgs')
    # general parameters 
    parser.add_argument('--input_dir', type=str, default='/')
    parser.add_argument('--results_dir', type=str, default='/')
    parser.add_argument('--checkpoint_dir', type=str, default='/')
    parser.add_argument('--features_file', type=str, 
                        default='input_aptamers')

    parser.add_argument('--train_data', type=str, default='train.csv')
    parser.add_argument('--val_data', type=str, default='val.csv')
    parser.add_argument('--test_data', type=str, default='test.csv')

    parser.add_argument('--SkipExistingResults', action='store_true',
                        default=False)

    parser.add_argument('--target_order', type=list,
                        default=['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'])

    args, _ = parser.parse_known_args()
    return args


def main(args):
    """
    Main function for fitting TabPFN models  

    Args:
        args (argparse.Namespace): 
            Parsed command-line arguments for TabPFN configuration.
    """
    # load data
    features = txt2list(args.features_file)
    cols = args.target_order + features 
    train_df = pd.read_csv(os.path.join(args.input_dir, args.train_data),
                           low_memory=False,
                           usecols=cols)
    val_df = pd.read_csv(os.path.join(args.input_dir, args.val_data),
                         low_memory=False,
                         usecols=cols)
    test_df = pd.read_csv(os.path.join(args.input_dir, args.test_data),
                          low_memory=False,
                          usecols=cols)

    # preprocessing following ProtAIDe
    # normalization
    train_df = avgProt_norm(train_df, features)
    val_df = avgProt_norm(val_df, features)
    test_df = avgProt_norm(test_df, features)

    # train
    train_df_imputed, knn_imputer = knn_impute(
        train_df, features, isTrain=True)
    train_df_norm_imputed, scaler = gauss_norm(
        train_df_imputed, features, None, isTrain=True)
    # val
    val_df_imputed, _ = knn_impute(
        val_df, features, knn_imputer, isTrain=False)
    val_df_norm_imputed, _ = gauss_norm(
        val_df_imputed, features, scaler, isTrain=False)
    # test
    test_df_imputed, _ = knn_impute(
        test_df, features, knn_imputer, isTrain=False)
    test_df_norm_imputed, _ = gauss_norm(
        test_df_imputed, features, scaler, isTrain=False)


    # loop over each target to fit TabPFN
    for t in args.target_order:
        print('... Fit TabPFN for ', t, time.ctime(), flush=True)
        results_csv_path = os.path.join(
            args.results_dir, 'TabPFN_' + t + '_metrics.csv')
        if os.path.isfile(results_csv_path) and args.SkipExistingResults:
            # skip
            print('Results exist, skip...', flush=True)
            continue
        train_df_copy = copy.deepcopy(train_df_norm_imputed)
        val_df_copy = copy.deepcopy(val_df_norm_imputed)
        test_df_copy = copy.deepcopy(test_df_norm_imputed)
        train_df_copy.dropna(subset=[t], inplace=True)

        # get data for TabPFN
        X_train, y_train = \
            train_df_copy[features].values, train_df_copy[t].values 
        X_val, y_val = val_df_copy[features], val_df_copy[t].values 
        X_test, y_test = test_df_copy[features], test_df_copy[t].values

        # call TabPFN
        clf = TabPFNClassifier(ignore_pretraining_limits=True, 
                               balance_probabilities=True,
                               device='cuda')
        # fit 
        clf.fit(X_train, y_train)

        # on validation set to find best threshold 
        proba_val_full = clf.predict_proba(X_val)
        proba_val = proba_val_full[:, 1]
        mask_val = ~np.isnan(y_val)

        opt_threshold = get_opt_threshold_PRC(y_val[mask_val],
                                              proba_val[mask_val])
        
        # test set performances 
        proba_test_full = clf.predict_proba(X_test)
        proba_test = proba_test_full[:, 1]
        mask_test = ~np.isnan(y_test)
        metrics = clf_metrics(y_test[mask_test], 
                              proba_test[mask_test],
                              2, opt_threshold)
        
        np.save(
            os.path.join(args.results_dir, 'TabPFN_' + t + '_predproba.npy'), 
            proba_test)
        
        metrics_df = dict2df(metrics)
        metrics_df.to_csv(results_csv_path, index=False)

        # save fitted clf 
        save_fitted_tabpfn_model(
            clf,
            os.path.join(args.checkpoint_dir, 'TabPFN_' + t + '.tabpfn_fit')
        )
        list2txt([opt_threshold],
                 os.path.join(args.checkpoint_dir, 'Threshold_' + t + '.txt'))
    
        del clf, train_df_copy, val_df_copy, test_df_copy


if __name__ == '__main__':
    main(args_parser())