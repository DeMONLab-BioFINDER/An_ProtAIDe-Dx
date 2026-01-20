#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import numpy as np 
import pandas as pd


def sample_df(df, 
              N, 
              seed):
    """
    Sample N rows from a DataFrame with a specified random seed.

    Args:
        df (pd.DataFrame): Input DataFrame
        N (int): Number of rows to sample
        seed (int): Random seed for reproducibility

    Returns:
        tuple: A tuple containing two DataFrames:
            - selected_df (pd.DataFrame): Sampled DataFrame with N rows
            - unselected_df (pd.DataFrame): Remaining DataFrame after sampling
    """
    # Sample N rows with a specified seed
    selected_df = df.sample(n=N, 
                            random_state=seed)
    unselected_df = df.drop(selected_df.index)
    selected_df = selected_df.reset_index(drop=True)
    unselected_df = unselected_df.reset_index(drop=True)

    return selected_df, unselected_df


def k_shot_sampling(df, 
                    target, 
                    K, 
                    random_seed):
    """
    Randomly select K shots with balance.

    Args:
        df (pd.DataFrame): Input DataFrame
        target (str): Target variable name
        K (int): Number of shots to sample
        random_seed (int): Random seed for reproducibility

    Returns:
        tuple: A tuple containing two DataFrames:
            - K_df (pd.DataFrame): Sampled DataFrame with K shots
            - test_df (pd.DataFrame): Remaining DataFrame after sampling
    """
    df.dropna(subset=[target], inplace=True, ignore_index=True)
    pos_df = df[df[target] == 1]
    neg_df = df[df[target] == 0]
    if pos_df.shape[0] > neg_df.shape[0]:
        minor_class = 0
    else:
        minor_class = 1
    minor_df = df[df[target] == minor_class]
    major_df = df[df[target] == int(1 - minor_class)]

    if minor_df.shape[0] < K:
        minor_K_df, minor_test_df = sample_df(
            minor_df, int((minor_df.shape[0] + 1) // 2), random_seed)
    else:
        minor_K_df, minor_test_df = sample_df(
            minor_df, int(K/2), random_seed)
    major_K_df, major_test_df = sample_df(
        major_df, int(K - minor_K_df.shape[0]), random_seed)
    
    K_df = pd.concat(
        [minor_K_df, major_K_df], 
        axis=0, ignore_index=True)
    test_df = pd.concat(
        [minor_test_df, major_test_df], 
        axis=0, ignore_index=True)

    assert K_df.shape[0] == K, "Wrong K sampled!"
    assert K_df.shape[0] + test_df.shape[0] == df.shape[0], "Data lost!"

    return K_df, test_df


def calc_k_list(nb_subjects, 
                perc_threshold, 
                K_shot_list):
    """
    Calculate a list of K values based on 
    the number of subjects and a percentage threshold.

    Args:
        nb_subjects (int): Number of subjects
        perc_threshold (float): Percentage threshold
        K_shot_list (list): List of K shot values

    Returns:
        list: Filtered list of K shot values based on the percentage threshold
    """
    K_shot_list = np.array(K_shot_list)
    return list(K_shot_list[K_shot_list <= int(nb_subjects * perc_threshold)])