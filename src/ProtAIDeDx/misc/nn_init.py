#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import random
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init


def xavier_uniform_init(m, init_gain=1.0):
    """
    Xavier uniform initialization.

    Args:
        m (torch.nn.Module): Module to initialize
        init_gain (float, optional): Initialization gain. Defaults to 1.0.
    """
    if type(m) == nn.Linear:
        init.xavier_uniform_(m.weight, gain=init_gain)
        m.bias.data.fill_(0)


def set_random_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True