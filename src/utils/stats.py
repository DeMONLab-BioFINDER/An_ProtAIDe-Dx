#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import numpy as np
from scipy.stats import t
import scipy.stats as stats 

def FDR(p_list, alpha=0.05, corrected=False):
    """
    Computes the False Discovery Rate according to 
        Benjamini and Hochberg (1995).

    Args:
        p_list (list): List of p values.
        alpha (float, optional): Threshold. Defaults to 0.05.
        corrected (bool, optional): Defaults to False.

    Returns:
        float: FDR corrected threshold.
    """
    n_vals = len(p_list)
    p_list = np.reshape(p_list, (n_vals, ))
    num_tests = n_vals
    p_sorted = sorted(p_list, reverse=True)
    final_threshold = 0
    # generate k * alpha / num_tests
    ascend_ind = np.array([i for i in range(1, num_tests + 1)]).astype(float)
    descend_ind = np.array([i for i in range(num_tests, 0, -1)]).astype(float)
    if corrected:
        comp = descend_ind * alpha / num_tests
        comp = comp / sum(ascend_ind / num_tests)
    else:
        comp = descend_ind * alpha / num_tests
    end = len(comp)
    comp = comp[end - n_vals:end]
    index_array = np.where(p_sorted <= comp)[0]
    # print(index_array)
    if len(index_array) == 0:
        final_threshold = 0
    else:
        final_threshold = p_sorted[index_array[0]]
    return final_threshold
