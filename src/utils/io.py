#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os
import pickle
import pandas as pd


## general I/O functions
def txt2list(txt_path):
    """
    Read a txt file and return a list

    Args:
        txt_path (str): Path to the txt file
    """
    return [l.strip() for l in open(txt_path)]


def list2txt(list_obj, save_path):
    """
    Save a list to a txt file

    Args:
        list_obj (list): List of items to save
        save_path (str): Path to save the txt file
    """
    with open(save_path, 'w') as f:
        for item in list_obj:
            f.write("%s\n" % str(item))
    f.close()


def load_pkl(pkl_path):
    """
    Load a pkl file and return a dictionary

    Args:
        pkl_path (str): Path to the pickle file
    
    Returns:
        dict: Loaded dictionary from the pickle file
    """
    fobj = open(pkl_path, 'rb')
    data = pickle.load(fobj)
    fobj.close()

    return data


def save_pkl(dict_obj, save_path):
    """
    Save a dictionary to a pkl file

    Args:
        dict_obj (dict): Dictionary object to save
        save_path (str): Path to save the pickle file
    """
    fobj = open(save_path, 'wb')
    pickle.dump(dict_obj, fobj, protocol=2)
    fobj.close()


def dict2df(dict_f):
    """
    Convert dictionary to dataframe.

    Args:
        dict_f (dict): Dictionary to convert to dataframe

    Returns:
        pd.DataFrame: Dataframe representation of the dictionary
    """
    return pd.DataFrame(data=[dict_f])


def df2csv(df, save_path, *header):
    """
    Save a dataframe to a csv file

    Args:
        df (pd.DataFrame): Dataframe to save
        save_path (str): Path to save the csv file
        header (list, optional): List of column headers
    """
    if header:
        df.to_csv(save_path, sep=',', index=False, header=header)
    else:
        df.to_csv(save_path, sep=',', index=False)
