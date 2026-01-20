#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os
import torch
import argparse
import numpy as np 
import pandas as pd
import scipy.stats as stats
import torch.nn.functional as F
from src.utils.io import load_pkl, df2csv, txt2list
from src.ProtAIDeDx.misc.nn_helper import model_infer, \
    load_pretrained_weights, load_hyperParams
from src.ProtAIDeDx.misc.ProtAIDeDx_model import build_ProtAIDeDx


def args_parser():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(prog='PermFITArgs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--suffix', type=str, default='CV')
    parser.add_argument('--split', type=str, default='fold_0')

    parser.add_argument('--input_dir', type=str, default='/')
    parser.add_argument('--checkpoint_dir', type=str, default='/')

    parser.add_argument('--features_path', type=str, default='/')
    parser.add_argument('--hyperParam_path', type=str, default='/')
    parser.add_argument('--test_pkl', type=str, default='test.pkl')

    parser.add_argument('--PermFIT_results_dir', type=str, default='/')
    
    parser.add_argument('--target_order', type=list,
                        default=['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'])
    parser.add_argument('--nb_perms', type=int, default=100)

    args, _ = parser.parse_known_args()
    return args


def perm_feature(X, j, seed=0):
    """
    Permute the values of a specific feature column in the input matrix.

    Args:
        X (np.ndarray): Input feature matrix.
        j (int): Index of the feature column to permute.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        np.ndarray: Feature matrix with the specified column permuted.
    """
    X_permuted = X.copy()
    np.random.seed(seed)
    np.random.shuffle(X_permuted[:, j])
    return X_permuted


def compute_importance_score(y, 
                             predprob, 
                             predprob_perm,
                             prob_min=1e-10, 
                             prob_max=1-1e-10):
    """
    Compute importance score M_j

    Args:
        y (np.ndarray): True labels.
        predprob (np.ndarray): Predicted probabilities.
        predprob_perm (np.ndarray): 
            Predicted probabilities with permuted feature.
        prob_min (float, optional): 
            Minimum probability value for clipping. Defaults to 1e-10.
        prob_max (float, optional): 
            Maximum probability value for clipping. Defaults to 1-1e-10.

    Returns:
        np.ndarray: Importance scores.
    """
    predprob = np.clip(predprob, prob_min, prob_max)
    predprob_perm = np.clip(predprob_perm, prob_min, prob_max)

    score = y * np.log(predprob/predprob_perm) + \
        (1 - y) * np.log((1 - predprob) / (1 - predprob_perm))

    return score


def main(args):
    """
    Main function to perform PermFIT feature importance analysis.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    test_pkl = load_pkl(os.path.join(args.input_dir, args.test_pkl))
    X = test_pkl['input']
    input_aptamers = txt2list(args.features_path)
    input_head_dim, encoder_dims, drop_out = load_hyperParams(
        args.hyperParam_path,
        args,
        True
    )
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

    # get original prediction probabilities
    predprob, _ = model_infer([InputHead, ProAIDeModel], X)
    # input feature permutation
    permfit_stats = []
    for k in range(len(args.target_order)):
        y_k = test_pkl['target'][args.target_order[k]].reshape((-1, ))
        y_mask = ~np.isnan(y_k)
        nb_test = np.sum(y_mask)
        feature_importance_matrix = np.empty(
            (args.nb_perms, nb_test, X.shape[1]))
        for seed in range(args.nb_perms):
            for j in range(X.shape[1]):
                X_permuted = perm_feature(test_pkl['input'], j, seed)
                predprob_perm, _ = model_infer(
                    [InputHead, ProAIDeModel], X_permuted)
                predprob_kj = predprob[:, k].reshape((-1, ))
                predprob_perm_kj = predprob_perm[:, k].reshape((-1, ))
                score_kj = compute_importance_score(
                    y_k[y_mask],
                    predprob_kj[y_mask],
                    predprob_perm_kj[y_mask])
                feature_importance_matrix[seed, :, j] = score_kj
        
        # average across subjects then across nb_perms
        # the variance comes from nb_perms
        feature_score = np.nanmean(np.nanmean(
            feature_importance_matrix, axis=1), axis=0)
        feature_std = np.nanstd(
            np.nanmean(feature_importance_matrix, axis=1), axis=0)
        feature_pval = 1 - stats.norm.cdf(feature_score / feature_std)
        for j in range(X.shape[1]):
            stat_row = [args.target_order[k], 
                        input_aptamers[j], 
                        feature_score[j], 
                        feature_std[j], 
                        feature_pval[j]]
            permfit_stats.append(stat_row)

    stats_df = pd.DataFrame(data=permfit_stats,
                            columns=['Target', 
                                     'Protein', 
                                     'ScoreMean', 
                                     'ScoreStd', 
                                     'PVal'])
    
    df2csv(stats_df, os.path.join(args.PermFIT_results_dir, 
                                  'feature_importance_PermFIT.csv'))


if __name__ == '__main__':
    main(args_parser())
