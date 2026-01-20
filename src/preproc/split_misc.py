#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import pandas as pd
from sklearn import model_selection


def df_tr_val_test_split(df,
                         target,
                         seed,
                         tr_portion,
                         val_portion):
    """
    Split df into train, val, and test.

    Args:
        df (pd.DataFrame): Input dataframe
        target (str): Target column name
        seed (int): Random seed for reproducibility
        tr_portion (float): Proportion of data for training
        val_portion (float): Proportion of data for validation

    Returns:
        dict: Dictionary containing train, val, and test dataframes
    """
    # 1. remove subject with missing target in df
    df.dropna(subset=[target], inplace=True)
    df.reset_index(inplace=True)
    # 2. split
    tr_df, no_tr_df = model_selection.train_test_split(
        df, test_size=1-tr_portion, random_state=seed)
    val_df, test_df = model_selection.train_test_split(
        no_tr_df,
        test_size=(1 - val_portion - tr_portion)/(1 - tr_portion),
        random_state=seed)
    tr_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    split = dict()
    split['tr'] = tr_df
    split['val'] = val_df
    split['test'] = test_df

    return split


def df_cross_validation(df, 
                        target,
                        seed,
                        nb_folds=10):
    """
    Split into K-fold cross validation: Train/Val/Test: 8/1/1.

    Args:
        df (pd.DataFrame): Input dataframe
        target (str): Target column name
        seed (int): Random seed for reproducibility
        nb_folds (int, optional): _description_. Defaults to 10.

    Returns:
        dict: Dictionary containing train, val, and test dataframes
    """
    # 1. remove subject with missing target in df
    df.dropna(subset=[target], inplace=True)
    df.reset_index(inplace=True)
    # 2. split
    kf = model_selection.KFold(
        n_splits=nb_folds, 
        shuffle=True, 
        random_state=seed)
    # Prepare folds
    folds = list(kf.split(df))

    split = dict()
    # 9 folds for train_val, 1 fold for test
    for fold_index, (train_val_idx, test_idx) in enumerate(folds):
        # Split train_val_idx into training and validation (8:1 split)
        train_idx, val_idx = train_val_idx[:-len(train_val_idx) // 9], \
            train_val_idx[-len(train_val_idx) // 9:]
        # Generate DataFrames for each split
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]

        split[fold_index] = dict()
        split[fold_index]['tr'] = train_df.reset_index(
            inplace=False, drop=True)
        split[fold_index]['val'] = val_df.reset_index(
            inplace=False, drop=True)
        split[fold_index]['test'] = test_df.reset_index(
            inplace=False, drop=True)
    return split



def df_cross_validation_inner(df,  
                              target,
                              seed,
                              nb_folds=10):
    """
    Split "Inner" K-fold cross-validation,
    Train/Val: 9/1

    Args:
        df (pd.DataFrame): Input dataframe
        target (str): Target column name
        seed (int): Random seed for reproducibility
        nb_folds (int, optional): _description_. Defaults to 10.

    Returns:
        dict: Dictionary containing train and validation dataframes
    """
    # 1. remove subject with missing target in df
    df.dropna(subset=[target], inplace=True)
    df.reset_index(inplace=True)
    # 3. split
    kf = model_selection.KFold(
        n_splits=nb_folds, 
        shuffle=True, 
        random_state=seed)
    # Prepare folds
    folds = list(kf.split(df))

    split = dict()

    for fold_index, (train_idx, val_idx) in enumerate(folds):
        # Generate DataFrames for each split
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        split[fold_index] = dict()
        split[fold_index]['tr'] = train_df.reset_index(
            inplace=False, drop=True)
        split[fold_index]['val'] = val_df.reset_index(
            inplace=False, drop=True)
    return split


def train_val_split(train_val_index, 
                    label_series,
                    val_portion=0.25):
    """
    Split train and validation indices.

    Args:
        train_val_index (list or np.ndarray): 
            Indices for train and validation
        val_portion (float, optional): 
            Proportion of data for validation. Defaults to 0.25.
    """
    train_index, val_index = model_selection.train_test_split(
        train_val_index, 
        test_size=val_portion, 
        random_state=12,
        stratify=label_series.iloc[train_val_index])
    
    return train_index, val_index