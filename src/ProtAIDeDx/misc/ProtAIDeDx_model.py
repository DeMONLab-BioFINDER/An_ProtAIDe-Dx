#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import torch
from torch import nn
from src.ProtAIDeDx.misc.nn_init import xavier_uniform_init


def fnn_layer(in_features, out_features, p_drop=0.0):
    """
    1 layer Fully-connected neural network (FNN).

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        p_drop (float, optional): Dropout probability. Defaults to 0.0.

    Returns:
        nn.Sequential: A sequential container of the FNN layer components
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=p_drop)
    )


class FNNencoder(nn.Module):
    """
    FNN encoder class.
    """
    def __init__(self, 
                 in_dim=2119,  
                 hidden_dims=[1024, 256],
                 p_drop=0.1):
        """
        Initialization function.

        Args:
            in_dim (int, optional): Number of input features. Defaults to 2119.
            hidden_dims (list, optional): 
                List of hidden layer dimensions. Defaults to [1024, 256].
            p_drop (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        modules = []
        for h_dim_ in hidden_dims:
            modules.append(fnn_layer(in_dim, h_dim_, p_drop))
            in_dim = h_dim_
        self.encoder = nn.Sequential(*modules)
        self.init_parameters()
    
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded output tensor
        """
        return self.encoder(x)
    
    def init_parameters(self):
        """
        Initialize model parameters.
        """
        self.encoder.apply(xavier_uniform_init)


class InputHead(nn.Module):
    """
    Input head class.

    Args:
        nn (torch.nn.Module): Base class for all neural network modules
    """
    def __init__(self,
                 in_dim,
                 head_dim,
                 p_dropout=0.1):
        """
        Initialization function.

        Args:
            in_dim (int): Number of input features
            head_dim (int): Number of output features
            p_dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.input_head = fnn_layer(
            in_dim,head_dim, p_dropout)
        # initialization
        self.init_parameters()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.input_head(x)

    def init_parameters(self):
        """
        Initialize model parameters.
        """
        self.input_head.apply(xavier_uniform_init)


class ProtAIDe(nn.Module):
    """
    ProtAIDe class.
    """
    def __init__(self,
                 input_head_dim,
                 encoder_dims,
                 p_dropout):
        """
        Initialization function. 

        Args:
            input_head_dim (int): Number of input features for the input head
            encoder_dims (list): List of encoder layer dimensions
            p_dropout (float): Dropout probability
        """
        super().__init__()
        # Build encoder
        self.encoder = FNNencoder(input_head_dim,
                                  encoder_dims,
                                  p_dropout)
        # Output layers
        self.fc2control = nn.Linear(encoder_dims[-1], 1)
        self.fc2ad = nn.Linear(encoder_dims[-1], 1)
        self.fc2pd = nn.Linear(encoder_dims[-1], 1)
        self.fc2ftd = nn.Linear(encoder_dims[-1], 1)
        self.fc2als = nn.Linear(encoder_dims[-1], 1)
        self.fc2stroketia = nn.Linear(encoder_dims[-1], 1)
        # initialization
        self.init_parameters()

    def forward(self, input_head):
        """
        Forward pass.

        Args:
            input_head (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Concatenated predictions tensor
        """
        z = self.encoder(input_head)
        pred = torch.cat([self.fc2control(z),
                          self.fc2ad(z),
                          self.fc2pd(z),
                          self.fc2ftd(z),
                          self.fc2als(z),
                          self.fc2stroketia(z)], dim=1)
        return pred, z
    
    def init_parameters(self):
        """
        Initialize model parameters.
        """
        self.fc2control.apply((xavier_uniform_init))
        self.fc2ad.apply((xavier_uniform_init))
        self.fc2pd.apply((xavier_uniform_init))
        self.fc2ftd.apply((xavier_uniform_init))
        self.fc2als.apply((xavier_uniform_init))
        self.fc2stroketia.apply((xavier_uniform_init))


def build_ProtAIDeDx(in_dim, 
                     input_head_dim,
                     encoder_dims, 
                     p_dropout):
    """
    Build ProtAIDeDx models with given hyperparameters

    Args:
        in_dim (int): Number of input features
        input_head_dim (int): Number of input features for the input head
        encoder_dims (list): List of encoder layer dimensions
        p_dropout (float): Dropout probability

    Returns:
        tuple: Tuple containing the InputHeadLayer and ProtAIDeModel instances
    """
    InputHeadLayer = InputHead(
        in_dim=in_dim,
        head_dim=input_head_dim,
        p_dropout=p_dropout)
    ProtAIDeModel = ProtAIDe(
        input_head_dim=input_head_dim,
        encoder_dims=encoder_dims,
        p_dropout=p_dropout)
    return InputHeadLayer, ProtAIDeModel
