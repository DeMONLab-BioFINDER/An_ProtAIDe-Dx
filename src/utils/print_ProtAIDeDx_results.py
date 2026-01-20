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


def args_parser():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(prog='Args')
    parser.add_argument('--suffix', type=str, default='CV')
    parser.add_argument('--results_dir', type=str, default='/')
    parser.add_argument('--targets2pred_path', type=str, default='/')

    parser.add_argument('--nb_folds', type=int, default=10)
    parser.add_argument('--sites', type=list,
                        default=['A', 'C', 'D', 'E', 'F', 'G', 'I',
                                 'J', 'L', 'M', 'N', 'P', 'Q', 'R'])
    
    main_args, _ = parser.parse_known_args()
    return main_args


def print_ProtAIDeDx_results_CV(results_dir,
                                sub_dirs_list):
    """
    Print ProtAIDeDx cross-validation results summary.

    Args:
        results_dir (str): Directory containing results.
        sub_dirs_list (list): List of sub-directory names.
    """
    # Collect per-fold arrays by target
    auc_by_tgt = {}
    bca_by_tgt = {}

    for sub in sub_dirs_list:
        csv_path = os.path.join(results_dir, sub, "test_metrics.csv")
        df = pd.read_csv(csv_path)[["target", "AUC", "BCA"]]

        for _, r in df.iterrows():
            tgt = r["target"]
            auc_by_tgt.setdefault(tgt, []).append(float(r["AUC"]))
            bca_by_tgt.setdefault(tgt, []).append(float(r["BCA"]))

    # Build summary rows
    targets_order = ["CU", "AD", "PD", "FTD", "ALS", "StrokeTIA"]
    rows = []
    for tgt in targets_order:
        auc = np.array(auc_by_tgt[tgt], dtype=float)
        bca = np.array(bca_by_tgt[tgt], dtype=float)

        rows.append(
            {
                "Target": "Control" if tgt == "CU" else tgt,
                "AUC mean": np.mean(auc),
                "AUC std": np.std(auc),
                "BCA mean": np.mean(bca),
                "BCA std": np.std(bca)
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=["Target", "AUC mean", "AUC std", 
                 "BCA mean", "BCA std"],
    )

    summary["Target"] = summary["Target"].replace({"CU": "Control"})

    desired_order = ["Control", "AD", "PD", "FTD", "ALS", "StrokeTIA"]
    summary["Target"] = pd.Categorical(summary["Target"], 
                                       categories=desired_order, ordered=True)
    summary = summary.sort_values("Target").reset_index(drop=True)

    with pd.option_context("display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))


def print_ProtAIDeDx_results_LOSO(results_dir,
                                  sub_dirs_list,
                                  target2pred):
    """
    Print ProtAIDeDx leave-one-site-out results summary.

    Args:
        results_dir (str): Directory containing results.
        sub_dirs_list (list): List of sub-directory names.
    """
    # Collect per-fold arrays by target
    auc_by_tgt = {}
    bca_by_tgt = {}

    for sub in sub_dirs_list:
        csv_path = os.path.join(results_dir, sub, "test_metrics.csv")
        df = pd.read_csv(csv_path)[["target", "AUC", "BCA"]]
        site = sub.split('_')[1]

        for _, r in df.iterrows():
            tgt = r["target"]
            if tgt in target2pred[site]:
                auc_by_tgt.setdefault(tgt, []).append(float(r["AUC"]))
                bca_by_tgt.setdefault(tgt, []).append(float(r["BCA"]))

    # Build summary rows
    targets_order = ["CU", "AD", "PD", "FTD", "ALS", "StrokeTIA"]
    rows = []
    for tgt in targets_order:
        auc = np.array(auc_by_tgt[tgt], dtype=float)
        bca = np.array(bca_by_tgt[tgt], dtype=float)

        rows.append(
            {
                "Target": "Control" if tgt == "CU" else tgt,
                "AUC mean": np.mean(auc),
                "AUC std": np.std(auc),
                "BCA mean": np.mean(bca),
                "BCA std": np.std(bca)
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=["Target", "AUC mean", "AUC std", 
                 "BCA mean", "BCA std"],
    )

    summary["Target"] = summary["Target"].replace({"CU": "Control"})

    desired_order = ["Control", "AD", "PD", "FTD", "ALS", "StrokeTIA"]
    summary["Target"] = pd.Categorical(summary["Target"], 
                                       categories=desired_order, ordered=True)
    summary = summary.sort_values("Target").reset_index(drop=True)

    with pd.option_context("display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))


def main(args):
    """
    Main function to print ProtAIDeDx results summary.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    if args.suffix == 'CV':
        sub_dirs_list = [
            'fold_' + str(fold) for fold in range(args.nb_folds)
        ]
        print_ProtAIDeDx_results_CV(
            args.results_dir,
            sub_dirs_list
        )

    elif args.suffix == 'LOSO':
        sub_dirs_list = [
            'site_' + str(site) for site in args.sites
        ]

        # load predictable targets 
        targets2pred_df = pd.read_csv(
            args.targets2pred_path)
        

        def sites_to_pos_targets(df):
            targets = ['CU', 'AD', 'PD', 'FTD', 'ALS', 
                       'StrokeTIA']
            return {
                row["Site"]:[t for t in targets if row[t] == 1]
                for _, row in df.iterrows()
            }

        print_ProtAIDeDx_results_LOSO(
            args.results_dir,
            sub_dirs_list,
            sites_to_pos_targets(targets2pred_df)
        )
    else:
        raise ValueError("Only support CV or LOSO!")


if __name__ == '__main__':
    main(args_parser())
        