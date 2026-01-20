#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.utils.io import load_pkl


def construct_label_mask_array(pkl, label_order):
    """
    Construct label array and mask array with given order.

    Args:
        pkl (dict): Loaded pickle data containing 'input', 'target', and 'mask'
        label_order (list): List of target names in the desired order
    Returns:
        tuple: A tuple containing label array and mask array
    """
    nb_subjects = pkl['input'].shape[0]
    nb_targets = len(label_order)
    label_arr = np.zeros((nb_subjects, nb_targets))
    mask_arr = np.zeros((nb_subjects, nb_targets))
    for i, target in enumerate(label_order):
        label_arr[:, i] = pkl['target'][target].reshape((-1, ))
        mask_arr[:, i] = pkl['mask'][target].reshape((-1, ))

    return label_arr, mask_arr


class protDataset(Dataset):
    """
    A Proteomics Dataset class.

    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset class
    """
    def __init__(self,
                 data_arr, 
                 label_arr,
                 mask_arr):
        """
        A Proteomics Dataset class.

        Args:
            data_arr (np.ndarray): Array of input data
            label_arr (np.ndarray): Array of labels
            mask_arr (np.ndarray): Array of masks
        """
        super().__init__()
        self.data_arr = data_arr
        self.label_arr = label_arr
        self.mask_arr = mask_arr

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            tuple: A tuple containing a sample, label, and mask
        """
        sample = self.data_arr[index]
        label = self.label_arr[index]
        mask = self.mask_arr[index]
        
        return sample, label, mask
    
    def __len__(self):
        return self.data_arr.shape[0]


def get_train_dataloader(train_pkl_path,
                         target_order, 
                         batch_size, 
                         shuffle=True,
                         drop_last=False):
    """
    Get train dataloader.

    Args:
        train_pkl_path (str): Path to the training pickle file
        target_order (list): List of target names in the desired order
        batch_size (int): Batch size for the dataloader
        shuffle (bool, optional): 
            Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): 
            Whether to drop the last incomplete batch. Defaults to False.
    Returns:
        torch.utils.data.DataLoader: DataLoader for the training data
    """
    # 1. load train data 
    train_pkl = load_pkl(train_pkl_path)
    # 2. build dataset class
    train_label_arr, train_mask_arr = construct_label_mask_array(
        train_pkl, target_order)
    train_dataset = protDataset(
        data_arr=train_pkl['input'],
        label_arr=train_label_arr,
        mask_arr=train_mask_arr
    )    
    
    # 3. build train dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=1)
    
    ratio_list = get_negative_positive_ratio(train_label_arr, target_order)
    
    return train_dataloader, ratio_list



def load_val_data(val_pkl_path, target_order, device):
    """
    Load validation data. 

    Args:
        val_pkl_path (str): Path to the validation pickle file
        target_order (list): List of target names in the desired order
        device (torch.device): Device to load the data onto

    Returns:
        tuple: A tuple containing validation input, labels, and masks
    """
    val_pkl = load_pkl(val_pkl_path)
    val_input = val_pkl['input']
    val_label, val_mask = construct_label_mask_array(val_pkl, target_order)
    val_input, val_label, val_mask = \
        torch.tensor(val_input), \
            torch.tensor(val_label), torch.tensor(val_mask)
    val_input, val_label, val_mask = \
        val_input.to(device), val_label.to(device), val_mask.to(device)
    return val_input.float(), val_label, val_mask.bool()


def get_negative_positive_ratio(label_arr, target_order):
    """
    Compute the ratio of negative/positive class, for balanced trining purpose. 

    Args:
        label_arr (np.ndarray): Array of labels
        target_order (list): List of target names in the desired order

    Returns:
        list: List of negative to positive ratios for each target
    """
    ratio_list = []
    for i, _ in enumerate(target_order):
        negative = np.sum(label_arr[:, i] == 0)
        positive = np.sum(label_arr[:, i] == 1)
        ratio_list.append(negative / positive)
    return ratio_list