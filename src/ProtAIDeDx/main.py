#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
from src.utils.io import load_pkl, df2csv, txt2list
from src.ProtAIDeDx.misc.ProtAIDeDx_model import build_ProtAIDeDx
from src.ProtAIDeDx.misc.nn_data import construct_label_mask_array
from src.ProtAIDeDx.misc.nn_helper import load_hyperParams, \
    load_ProbaThresholds, \
    load_pretrained_weights, eval_on_test, infer_one_set


def args_parser():
    """
    Argument parser for command-line options.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(prog='Args')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--split', type=str, default='fold_0')
    parser.add_argument('--new', action='store_true', default=False)
    parser.add_argument('--NoEval', action='store_true', default=False)

    parser.add_argument('--input_dir', type=str, default='/')
    parser.add_argument('--checkpoint_dir', type=str, default='/')
    parser.add_argument('--results_dir', type=str, default='/')
    parser.add_argument('--splits_dir', type=str, default='/')
    parser.add_argument('--hyperParam_path', type=str, default='/')
    parser.add_argument('--probaThresholds_path', type=str, default='/')
    parser.add_argument('--features_path', type=str, default='/')

    parser.add_argument('--z_dim', type=int, default=0)
    parser.add_argument('--target_order', type=list,
                        default=['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'])

    parser.add_argument('--train_pkl', type=str, default='train.pkl')
    parser.add_argument('--val_pkl', type=str, default='val.pkl')
    parser.add_argument('--test_pkl', type=str, default='test.pkl')

    parser.add_argument('--train_raw', type=str, default='train.csv')
    parser.add_argument('--val_raw', type=str, default='val.csv')
    parser.add_argument('--test_raw', type=str, default='test.csv')

    main_args, _ = parser.parse_known_args()
    return main_args


def main(args):
    """
    Making prediction for one fold or one site data.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    # 1. load parameters
    threshold_list = load_ProbaThresholds(args.probaThresholds_path,
                                          args.suffix,
                                          args.split)
    threshold_list = np.array(threshold_list, dtype=np.float32)

    input_head_dim, encoder_dims, drop_out = load_hyperParams(
        args.hyperParam_path,
        args,
        True
    )
    z_dim = int(encoder_dims[-1])
    input_aptamers = txt2list(args.features_path)

    # 2. get models
    InputHead, ProAIDeModel = build_ProtAIDeDx(len(input_aptamers),
                                               input_head_dim,
                                               encoder_dims,
                                               drop_out)
    InputHead, ProAIDeModel = load_pretrained_weights(
        InputHead,
        ProAIDeModel,
        args.checkpoint_dir,
        args.suffix,
        args.split,
        args.device
    )

    # 3. load data 
    if not args.new:         
        train_pkl = load_pkl(os.path.join(args.input_dir, args.train_pkl))
        train_df = pd.read_csv(
            os.path.join(args.splits_dir, args.train_raw), low_memory=False)

        val_pkl = load_pkl(os.path.join(args.input_dir, args.val_pkl))
        val_df = pd.read_csv(
            os.path.join(args.splits_dir, args.val_raw), low_memory=False)

    test_pkl = load_pkl(os.path.join(args.input_dir, args.test_pkl))
    test_df = pd.read_csv(
        os.path.join(args.splits_dir, args.test_raw), low_memory=False)

    # 4. inference
    if not args.new:         
        infer_one_set([InputHead, ProAIDeModel], 
                    train_pkl['input'], train_df, 
                    args.target_order, threshold_list,
                    z_dim, 
                    os.path.join(args.results_dir, 'train_results.csv'))
        infer_one_set([InputHead, ProAIDeModel], 
                    val_pkl['input'], val_df, 
                    args.target_order, threshold_list,
                    z_dim, 
                    os.path.join(args.results_dir, 'val_results.csv'))
        infer_one_set([InputHead, ProAIDeModel], 
                    test_pkl['input'], test_df, 
                    args.target_order, threshold_list,
                    z_dim, 
                    os.path.join(args.results_dir, 'test_results.csv'))
    else:
        infer_one_set([InputHead, ProAIDeModel], 
                    test_pkl['input'], test_df, 
                    args.target_order, threshold_list,
                    z_dim, 
                    os.path.join(args.results_dir, 'test_results.csv'),
                    noGTLabel=True)


    # 4. evaluate on test set to get metric scores
    if not args.NoEval:
        test_label, test_mask = construct_label_mask_array(
            test_pkl, args.target_order)
        test_x = torch.tensor(test_pkl['input']).float()
        test_label = torch.tensor(test_label)
        test_mask = torch.tensor(test_mask).bool() 
        test_met_df = eval_on_test(args, test_x, test_label, test_mask,
                                InputHead, ProAIDeModel, 
                                threshold_list)
        test_met_df = test_met_df.rename(
            columns={"index": "target",
                    "auc": "AUC",
                    "bas": "BCA"})
        df2csv(test_met_df, os.path.join(
            args.results_dir, 'test_metrics.csv'))



if __name__ == '__main__':
    main(args_parser())