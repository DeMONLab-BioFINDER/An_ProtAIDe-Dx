#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os 
import joblib
import argparse
import numpy as np
import pandas as pd
from src.utils.io import save_pkl, txt2list
from src.preproc.preproc_misc import avgProt_norm, gauss_norm, knn_impute, \
    construct_input_dict, append_continuous_target, append_categorical_target

import warnings
warnings.filterwarnings('ignore')

site_mapper = {
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4,
    'F':5,
    'G':6,
    'H':7,
    'I':8,
    'J':9,
    'K':10,
    'L':11,
    'M':12,
    'N':13,
    'O':14,
    'P':15,
    'Q':16,
    'R':17,
    'S':18,
    'T':10,
    'U':20,
    'V':21,
    'W':22,
    'X':23,
    'Y':24,
    'Z':25
}


def gen_1split(train_df,
               val_df,
               test_df,
               prots,
               output_dir,
               ckpt_dir,
               categ_vars,
               conti_vars):
    """
    Generate input for one split.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        prots (list): List of proteins.
        output_dir (str): Output directory.
        ckpt_dir (str): Checkpoint directory.
        categ_vars (list): List of categorical variables.
        conti_vars (list): List of continuous variables.
    """
    # individual normalization
    train_df = avgProt_norm(train_df, prots)
    val_df = avgProt_norm(val_df, prots)
    test_df = avgProt_norm(test_df, prots)

    # impute missing proteins with KNN 
    train_df_imputed, knn_imputer = knn_impute(train_df,prots, isTrain=True)
    val_df_imputed, _ = knn_impute(val_df,
                                   prots,
                                   imputer=knn_imputer,
                                   isTrain=False)
    test_df_imputed, _ = knn_impute(test_df,
                                    prots,
                                    imputer=knn_imputer,
                                    isTrain=False)
    # normalize proteins with Gauss Rank scaler
    train_df_norm_imputed, scaler = gauss_norm(train_df_imputed,
                                               prots,
                                               None,
                                               isTrain=True)
    val_df_norm_imputed, _ = gauss_norm(val_df_imputed,
                                        prots,
                                        scaler,
                                        isTrain=False)
    test_df_norm_imputed, _ = gauss_norm(test_df_imputed,
                                         prots,
                                         scaler,
                                         isTrain=False)

    # construct input dict
    train_pkl = construct_input_dict(
        train_df_norm_imputed[prots].values, isTrain=True)
    val_pkl = construct_input_dict(
        val_df_norm_imputed[prots].values, isTrain=False)
    test_pkl = construct_input_dict(
        test_df_norm_imputed[prots].values, isTrain=False)
    # add continuous targets
    for target_var in conti_vars:
        train_pkl, mean, std = append_continuous_target(
            train_pkl, 
            train_df_norm_imputed, 
            target_var, isTrain=True)
        val_pkl, _, _ = append_continuous_target(val_pkl, 
                                                 val_df_norm_imputed,
                                                 target_var,
                                                 mean,
                                                 std,
                                                 isTrain=False)
        test_pkl, _, _ = append_continuous_target(test_pkl, 
                                                  test_df_norm_imputed,
                                                  target_var, 
                                                  mean,
                                                  std,
                                                  isTrain=False)
    # add categorical targets 
    for target_var in categ_vars:
        train_pkl = append_categorical_target(train_pkl,
                                              train_df_norm_imputed,
                                              target_var)
        val_pkl = append_categorical_target(val_pkl,
                                            val_df_norm_imputed,
                                            target_var)
        test_pkl = append_categorical_target(test_pkl,
                                             test_df_norm_imputed,
                                             target_var)

    # add site variable
    train_pkl['site'] = train_df['Contributor_Code'].map(
        site_mapper).to_numpy()
    val_pkl['site'] = val_df['Contributor_Code'].map(
        site_mapper).to_numpy()
    test_pkl['site'] = test_df['Contributor_Code'].map(
        site_mapper).to_numpy()

    # save 
    save_pkl(train_pkl, os.path.join(output_dir, 'train.pkl'))
    save_pkl(val_pkl, os.path.join(output_dir, 'val.pkl'))
    save_pkl(test_pkl, os.path.join(output_dir, 'test.pkl'))

    if ckpt_dir is None:
        pass 
    else:
        joblib.dump(knn_imputer, os.path.join(ckpt_dir, 'knn_imputer.sav'))
        joblib.dump(scaler, os.path.join(ckpt_dir, 'gaussrank_scaler.sav'))


def input_gen_CV(splits_dir,
                 output_dir,
                 ckpt_dir,
                 save_ckpt=True,
                 nb_folds=10,
                 categ_vars=['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'],
                 conti_vars=[]):
    """
    Generate input for cross-validation.

    Args:
        splits_dir (str): Directory containing the data splits.
        output_dir (str): Directory to save the generated inputs.
        ckpt_dir (str): Directory to save checkpoints.
        save_ckpt (bool, optional): 
            Whether to save checkpoints. Defaults to True.
        nb_folds (int, optional): 
            Number of folds for cross-validation. Defaults to 10.
        categ_vars (list, optional): 
            List of categorical variables. 
            Defaults to ['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'].
        conti_vars (list, optional): 
            List of continuous variables. Defaults to [].
    """

    # for each split 
    for fold in range(nb_folds):
        fold_str = 'fold_' + str(fold)
        fold_input_dir = os.path.join(splits_dir, fold_str)
        fold_output_dir = os.path.join(output_dir, fold_str)
        fold_ckpt_dir = os.path.join(ckpt_dir, fold_str)

        input_aptamers = txt2list(
            os.path.join(fold_ckpt_dir, 'input_aptamers.txt'))

        # load train, val, test
        train_df = pd.read_csv(os.path.join(fold_input_dir, 'train.csv'),
                              low_memory=False)
        val_df = pd.read_csv(os.path.join(fold_input_dir, 'val.csv'),
                             low_memory=False)
        test_df = pd.read_csv(os.path.join(fold_input_dir, 'test.csv'),
                             low_memory=False)
        if save_ckpt:
            gen_1split(train_df, 
                       val_df,
                       test_df,
                       input_aptamers,
                       fold_output_dir,
                       fold_ckpt_dir,
                       categ_vars,
                       conti_vars)
        else:
            gen_1split(train_df, 
                       val_df,
                       test_df,
                       input_aptamers,
                       fold_output_dir,
                       None,
                       categ_vars,
                       conti_vars)         


def input_gen_LOSO(splits_dir,
                   output_dir,
                   ckpt_dir,
                   save_ckpt=True,
                   sites=['A', 'C', 'D', 'E', 'F',
                          'G', 'I', 'J', 'L', 'M',
                          'N', 'P', 'Q', 'R'],
                   categ_vars=['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'],
                   conti_vars=[]):
    """
    Generate input for leave-one-site-out.

    Args:
        splits_dir (str): Directory containing the data splits.
        output_dir (str): Directory to save the generated inputs.
        ckpt_dir (str): Directory to save checkpoints.
        save_ckpt (bool, optional): 
            Whether to save checkpoints. Defaults to True.
        sites (list, optional): 
            List of sites. 
            Defaults to ['A', 'C', 'D', 'E', 'F', 'G', 
                         'I', 'J', 'L', 'M', 'N', 'P', 'Q', 'R'].
        categ_vars (list, optional): 
            List of categorical variables. 
            Defaults to ['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'].
        conti_vars (list, optional): 
            List of continuous variables. Defaults to [].
    """
    # for each split 
    for site in sites:
        site_str = 'site_' + str(site)
        site_input_dir = os.path.join(splits_dir, site_str)
        site_output_dir = os.path.join(output_dir, site_str)
        site_ckpt_dir = os.path.join(ckpt_dir, site_str)

        input_aptamers = txt2list(
            os.path.join(site_ckpt_dir, 'input_aptamers.txt'))

        # load train, val, test
        train_df = pd.read_csv(os.path.join(site_input_dir, 'train.csv'),
                              low_memory=False)
        val_df = pd.read_csv(os.path.join(site_input_dir, 'val.csv'),
                             low_memory=False)
        test_df = pd.read_csv(os.path.join(site_input_dir, 'test.csv'),
                             low_memory=False)
        if save_ckpt:
            gen_1split(train_df, 
                      val_df,
                      test_df,
                      input_aptamers,
                      site_output_dir,
                      site_ckpt_dir,
                      categ_vars,
                      conti_vars)
        else:
            gen_1split(train_df, 
                       val_df,
                       test_df,
                       input_aptamers,
                       site_output_dir,
                       None,
                       categ_vars,
                       conti_vars)
            

def gen_input_inference(input_csv_path, 
                        output_dir, 
                        ckpt_dir,
                        categ_vars,
                        conti_vars):
    """
    Generate input for inference.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_dir (str): Directory to save the generated inputs.
        ckpt_dir (str): Directory containing checkpoints.
        categ_vars (list): List of categorical variables.
        conti_vars (list): List of continuous variables.
    """
    # load data 
    input_df = pd.read_csv(input_csv_path, low_memory=False)
    # load ckpts
    knn_imputer = joblib.load(os.path.join(ckpt_dir, 'knn_imputer.sav'))
    scaler = joblib.load(os.path.join(ckpt_dir, 'gaussrank_scaler.sav'))
    # load input aptamers
    input_aptamers = txt2list(os.path.join(ckpt_dir, 'input_aptamers.txt'))

    input_df = avgProt_norm(input_df, input_aptamers)

    input_df_imputed, _ = knn_impute(input_df,
                                    input_aptamers, 
                                    imputer=knn_imputer, 
                                    isTrain=False)
    input_df_norm_imputed, _ = gauss_norm(input_df_imputed,
                                         input_aptamers,
                                         scaler,
                                         isTrain=False)
    # construct input dict
    input_pkl = construct_input_dict(
        input_df_norm_imputed[input_aptamers].values, isTrain=False)
    # add continuous targets
    for target_var in conti_vars:
        input_pkl, _, _ = append_continuous_target(input_pkl, 
                                                  input_df_norm_imputed, 
                                                  target_var, 
                                                  isTrain=False)
    # add categorical targets
    for target_var in categ_vars:
        input_pkl = append_categorical_target(input_pkl,
                                              input_df_norm_imputed,
                                              target_var)
    # save data 
    save_pkl(input_pkl, os.path.join(output_dir, 
                                     'test.pkl'))

def args_parser(): 
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(prog='Args')
    parser.add_argument('--GNPC', action='store_true', default=False)
    parser.add_argument('--BF2', action='store_true', default=False)
    parser.add_argument('--example', action='store_true', default=False)

    args, _ = parser.parse_known_args()
    return args


def main(args):
    """
    Main function for generating inputs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    if args.example:
        pass 
    if args.GNPC:
        # cross-validation 
        input_gen_CV(
            './data/replica/CV/splits',
            './data/replica/CV/deep_input',
            './checkpoints/data_proc/CV',
            save_ckpt=False
        )

        # leave-one-site-out
        input_gen_LOSO(
            './data/replica/LOSO/splits',
            './data/replica/LOSO/deep_input',
            './checkpoints/data_proc/LOSO',
            save_ckpt=False
        )
    if args.BF2:
        gen_input_inference(
            './data/replica/BF2/splits/BF2_Soma7k_Baseline.csv',
            './data/replica/BF2/deep_input',
            './checkpoints/data_proc/LOSO/site_C',
            categ_vars=['Normal Control',
                        'AD',
                        'LBD',
                        'NonAD MCI',
                        'FTD Spectrum',
                        '4R Tauopathies', 
                        'StrokeTIA'],
            conti_vars=[]
        )


if __name__ == '__main__':
    main(args_parser())

