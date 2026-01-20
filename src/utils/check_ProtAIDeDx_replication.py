#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os 
import argparse
import numpy as np
import pandas as pd 
from src.utils.io import load_pkl


def args_parser():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(prog='Args')
    parser.add_argument('--suffix', type=str, default='CV')
    parser.add_argument('--results_dir', type=str, default='/')
    parser.add_argument('--ref_dir', type=str, default='/')

    parser.add_argument('--nb_folds', type=int, default=10)
    parser.add_argument('--sites', type=list,
                        default=['A', 'C', 'D', 'E', 'F', 'G', 'I',
                                 'J', 'L', 'M', 'N', 'P', 'Q', 'R'])
    
    main_args, _ = parser.parse_known_args()
    return main_args


def check_ProtAIDeDx_replication(results_dir,
                                 sub_dirs_list,
                                 ref_pkl_path):
    """
    Check if ProtAIDe-Dx results replicate reference results.

    Args:
        results_dir (str): Directory containing the results
        sub_dirs_list (list): List of sub-directory names to check
        ref_pkl_path (str): Path to the reference pickle file
    """
    ref_dict = load_pkl(ref_pkl_path)
    
    for sub in sub_dirs_list:
        csv_path = os.path.join(results_dir, sub, "test_metrics.csv")
        df = pd.read_csv(csv_path)[["target", "AUC", "BCA"]]

        AUC_vec = df['AUC'].values.reshape((-1, 1))
        BCA_vec = df['BCA'].values.reshape((-1, 1))

        assert np.allclose(ref_dict[sub]['AUC'],
                           AUC_vec,
                           equal_nan=True), "AUC not replicated! "+sub
        assert np.allclose(ref_dict[sub]['BCA'],
                           BCA_vec,
                           equal_nan=True), "BCA not replicated! "+sub
    

def main(args):
    """
    Main function to check if ProtAIDe-Dx results replicate reference results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    if args.suffix == 'CV':
        sub_dirs_list = [
            'fold_' + str(fold) for fold in range(args.nb_folds)
        ]
        check_ProtAIDeDx_replication(
            args.results_dir,
            sub_dirs_list,
            os.path.join(args.ref_dir, 'Fig1_GNPC_ref.pkl')
        )
        print("Congrats! You have replicated Fig1: ProtAIDe-Dx (" \
              + args.suffix + ')')

    elif args.suffix == 'LOSO':
        sub_dirs_list = [
            'site_' + str(site) for site in args.sites
        ]
        check_ProtAIDeDx_replication(
            args.results_dir,
            sub_dirs_list,
            os.path.join(args.ref_dir, 'Fig4_GNPC_ref.pkl')
        )
        print("Congrats! You have replicated Fig4: ProtAIDe-Dx (" \
              + args.suffix + ')')
    else:
        raise ValueError("Only support CV or LOSO!")


if __name__ == '__main__':
    main(args_parser())