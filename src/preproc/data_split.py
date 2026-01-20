#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os
import pandas as pd
from src.utils.io import load_pkl


def split_CV(data_csv_path,
             splits_pkl_path,
             output_dir,
             nb_folds=10):
    """
    Split data into cross-validation folds.

    Args:
        data_csv_path (str): Path to the CSV file containing the data
        splits_pkl_path (str): Path to the pickle file containing split indices
        output_dir (str): Directory to save the split CSV files
        nb_folds (int, optional): Number of folds. Defaults to 10.
    """
    assert nb_folds == 10, "Only allow 10-fold CV for replication!"

    data_df = pd.read_csv(data_csv_path, 
                          low_memory=False)
    split_dict = load_pkl(splits_pkl_path)

    for fold in range(nb_folds):
        fold_str = 'fold_' + str(fold)
        fold_split_dict = split_dict[fold_str]

        df = data_df.copy()

        # index train,val, test subjects 
        train_ids = fold_split_dict['train']
        val_ids = fold_split_dict['val']
        test_ids = fold_split_dict['test']

        train_df = df[df['PersonGroup_ID'].isin(train_ids)]
        val_df = df[df['PersonGroup_ID'].isin(val_ids)]
        test_df = df[df['PersonGroup_ID'].isin(test_ids)]

        train_df_order = train_df.set_index(
            "PersonGroup_ID").loc[train_ids].reset_index()
        val_df_order = val_df.set_index(
            "PersonGroup_ID").loc[val_ids].reset_index()
        test_df_order = test_df.set_index(
            "PersonGroup_ID").loc[test_ids].reset_index()
        
        train_df_order.reset_index(inplace=True, drop=True)
        val_df_order.reset_index(inplace=True, drop=True)
        test_df_order.reset_index(inplace=True, drop=True)

        # save 
        train_df_order.to_csv(os.path.join(output_dir, fold_str, 'train.csv'),
                        index=False, sep=',')
        val_df_order.to_csv(os.path.join(output_dir, fold_str, 'val.csv'),
                        index=False, sep=',')
        test_df_order.to_csv(os.path.join(output_dir, fold_str, 'test.csv'),
                        index=False, sep=',')
    


def split_LOSO(data_csv_path,
               splits_pkl_path,
               output_dir,
               sites=['A', 'C', 'D', 'E', 'F',
                      'G', 'I', 'J', 'L', 'M',
                      'N', 'P', 'Q', 'R']):
    """
    Split data into leave-one-site-out folds.

    Args:
        data_csv_path (str): Path to the CSV file containing the data
        splits_pkl_path (str): Path to the pickle file containing split indices
        output_dir (str): Directory to save the split CSV files
        sites (list, optional): _description_. 
        Defaults to ['A', 'C', 'D', 'E', 'F', 'G', 'I', 
                     'J', 'L', 'M', 'N', 'P', 'Q', 'R'].
    """
    assert len(sites) == 14, "Only allow 14-site LOSO for replication!"
    
    data_df = pd.read_csv(data_csv_path, 
                          low_memory=False)
    split_dict = load_pkl(splits_pkl_path)

    for site in sites:
        site_str = 'site_' + str(site)
        site_split_dict = split_dict[site_str]

        df = data_df.copy()

        # index train,val, test subjects 
        train_ids = site_split_dict['train']
        val_ids = site_split_dict['val']
        test_ids = site_split_dict['test']

        train_df = df[df['PersonGroup_ID'].isin(train_ids)]
        val_df = df[df['PersonGroup_ID'].isin(val_ids)]
        test_df = df[df['PersonGroup_ID'].isin(test_ids)]

        train_df_order = train_df.set_index(
            "PersonGroup_ID").loc[train_ids].reset_index()
        val_df_order = val_df.set_index(
            "PersonGroup_ID").loc[val_ids].reset_index()
        test_df_order = test_df.set_index(
            "PersonGroup_ID").loc[test_ids].reset_index()
        
        train_df_order.reset_index(inplace=True, drop=True)
        val_df_order.reset_index(inplace=True, drop=True)
        test_df_order.reset_index(inplace=True, drop=True)

        # save 
        train_df_order.to_csv(os.path.join(output_dir, site_str, 'train.csv'),
                        index=False, sep=',')
        val_df_order.to_csv(os.path.join(output_dir, site_str, 'val.csv'),
                        index=False, sep=',')
        test_df_order.to_csv(os.path.join(output_dir, site_str, 'test.csv'),
                        index=False, sep=',') 



if __name__ == '__main__':
    data_csv_path = './data/replica/raw/GNPC_Soma7k_Baseline.csv'
    
    # split for 10-fold cross-validation
    CV_split_path = './data/CV_splits.pkl'
    CV_output_dir = './data/replica/CV/splits'
    split_CV(data_csv_path,
             CV_split_path,
             CV_output_dir)

    # split for 14-site leave-one-site-out
    LOSO_split_path = './data/LOSO_splits.pkl'
    LOSO_output_dir = './data/replica/LOSO/splits'
    split_LOSO(data_csv_path,
               LOSO_split_path,
               LOSO_output_dir)


