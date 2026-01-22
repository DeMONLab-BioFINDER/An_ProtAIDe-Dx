#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os
import torch
import joblib 
import pandas as pd
from src.utils.io import txt2list
from src.ProtAIDeDx.main import args_parser as ProtAIDeDx_args_parser
from src.ProtAIDeDx.misc.ProtAIDeDx_model import build_ProtAIDeDx
from src.preproc.preproc_misc import avgProt_norm, gauss_norm, knn_impute
from src.ProtAIDeDx.misc.nn_helper import load_pretrained_weights, \
    load_hyperParams


def load_example_data(model='fold_0'):
    """
    Load simulated SomaLogic data for example usage.
    By default, loads data for 'fold_0' model. Feel free to change.

    Args:
        model (str, optional): Model name. Defaults to 'fold_0'.

    Returns:
        X (pd.DataFrame): Input X.
        y (pd.DataFrame): Target y.
    """
    # get data directories
    root_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir,'data')
    data_ckpt_dir = os.path.join(root_dir, 'checkpoints', 'data_proc', 'CV')
    # load simulated data
    simulated_df = pd.read_csv(
        os.path.join(data_dir, 'Simulated_SomaLogic_120Subjects.csv'))
    # get X, y
    targets = ['Control', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA']
    input_aptamers = txt2list(
        os.path.join(data_ckpt_dir, 
                     model, 
                     'input_aptamers.txt'))
    
    X = simulated_df[input_aptamers]
    y = simulated_df[targets]

    return X, y


def gen_example_input(X, 
                      model='fold_0',
                      device=torch.device('cpu')):
    """
    Generate example input tensor for model inference.
    By default, generate input for 'fold_0' model. Feel free to change.

    Args:
        X (pd.DataFrame): Input X.
        model (str, optional): Model name. Defaults to 'fold_0'.
        device (torch.device, optional): Device for tensor. 
            Defaults to torch.device('cpu').

    Returns:
        torch.Tensor: Example input tensor for model inference.
    """
    # get data directories
    root_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
    data_ckpt_dir = os.path.join(root_dir, 'checkpoints', 'data_proc', 'CV')

    # load fitted data processing pipelines
    knn_imputer = joblib.load(os.path.join(data_ckpt_dir, 
                                            model,
                                           'knn_imputer.sav'))
    scaler = joblib.load(os.path.join(data_ckpt_dir, 
                                      model,
                                      'gaussrank_scaler.sav')) 
    # get input aptamers
    input_aptamers = txt2list(
        os.path.join(data_ckpt_dir, 
                     model, 
                     'input_aptamers.txt'))

    # data processing 
    X_avgNormed = avgProt_norm(X, input_aptamers)

    X_imputed, _ = knn_impute(X_avgNormed,
                              input_aptamers, 
                              imputer=knn_imputer, 
                              isTrain=False)
    X_scaled, _ = gauss_norm(X_imputed,
                             input_aptamers,
                             scaler,
                             isTrain=False)
    
    # convert to tensor 
    X_tensor = torch.tensor(
        X_scaled[input_aptamers].values,
        dtype=torch.float32,
        device=device) 
    
    return X_tensor


def load_ProtAIDeDx_model(model='fold_0',
                          device=torch.device('cpu')):
    """
    Load ProtAIDe-Dx model for example inference.
    By default, generate input for 'fold_0' model. Feel free to change.

    Args:
        model (str, optional): Model name. Defaults to 'fold_0'.
        device (torch.device, optional): Device for model. 
            Defaults to torch.device('cpu').
    """
    # get data directories
    root_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
    model_ckpt_dir = os.path.join(root_dir, 'checkpoints', 'ProtAIDeDx')
    data_ckpt_dir = os.path.join(root_dir, 'checkpoints', 'data_proc', 'CV')

    # get input aptamers
    input_aptamers = txt2list(
        os.path.join(data_ckpt_dir, 
                     model, 
                     'input_aptamers.txt'))
    
    # load model
    ProtAIDeDx_args = ProtAIDeDx_args_parser()
    ProtAIDeDx_args.suffix = 'CV'
    ProtAIDeDx_args.split = model
    ProtAIDeDx_args.device = device
    ProtAIDeDx_args.checkpoint_dir = model_ckpt_dir
    ProtAIDeDx_args.hyperParam_path = os.path.join(
        model_ckpt_dir,
        'ProtAIDeDx_HyperParams.csv')
    ProtAIDeDx_args.probaThresholds_path = os.path.join(
        model_ckpt_dir,
        'ProtAIDeDx_ProbaThresholds.csv')

    input_head_dim, encoder_dims, drop_out = load_hyperParams(
        ProtAIDeDx_args.hyperParam_path,
        ProtAIDeDx_args,
        True)
    InputHead, ProAIDeModel = build_ProtAIDeDx(len(input_aptamers),
                                               input_head_dim,
                                               encoder_dims,
                                               drop_out)
    InputHead, ProAIDeModel = load_pretrained_weights(
        InputHead,
        ProAIDeModel,
        ProtAIDeDx_args.checkpoint_dir,
        ProtAIDeDx_args.suffix,
        ProtAIDeDx_args.split,
        ProtAIDeDx_args.device
    )

    return [InputHead, ProAIDeModel]