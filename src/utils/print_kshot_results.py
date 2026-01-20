#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os 
import numpy as np
import pandas as pd 
from src.utils.metrics import clf_auc, clf_bas

target_mapper = {
    'Normal Control': 'Control',
    'AD': 'AD',
    'LBD': 'PD',
    'FTD Spectrum': 'FTD',
    'StrokeTIA': 'StrokeTIA'
}

metics_mapper = {
    'auc': 'AUC',
    'bas': 'BCA'
}


def print_kshot_results(
        results_dir='./results/replica/Fig4_BF2/kshot',
        nb_seeds=20,
        targets=['Normal Control', 'AD', 'LBD', 'FTD Spectrum', 'StrokeTIA'],
        metrics=['auc', 'bas']):
    """
    Print k-shot learning results summary.

    Args:
        results_dir (str, optional): 
            Directory containing results. 
            Defaults to './results/replica/Fig4_BF2/kshot'.
        nb_seeds (int, optional): Number of random seeds. Defaults to 20.
        targets (list, optional): 
            List of target labels. 
            Defaults to ['Normal Control', 'AD', 'LBD', 
                         'FTD Spectrum', 'StrokeTIA'].
        metrics (list, optional): 
            List of metrics to include. Defaults to ['auc', 'bas'].
    """
    metrics_dict = {
        target_mapper[t]: {
            metics_mapper[metric]: [] for metric in metrics} for t in targets}

    for t in targets:
        for seed in range(nb_seeds):
            file_path = os.path.join(
                results_dir,
                f"LR_{t}_K100_Seed{seed}_metrics.csv"
            )
            df = pd.read_csv(file_path)
            for metric in metrics:
                if metric in df.columns:
                    metrics_dict[
                        target_mapper[t]][metics_mapper[metric]].append(
                            df[metric].values[0])
    # Build summary rows
    rows = []
    for tgt in target_mapper.values():
        row = {"Target": tgt}
        for metric in metrics:
            metric_name = metics_mapper[metric]
            values = np.array(
                metrics_dict[tgt][metric_name], dtype=float)
            row[f"{metric_name} mean"] = np.mean(values)
            row[f"{metric_name} std"] = np.std(values)
        rows.append(row)
    
    summary = pd.DataFrame(
        rows,
        columns=["Target", "AUC mean", "AUC std", 
                 "BCA mean", "BCA std"],
    )

    summary["Target"] = summary["Target"].replace({"CU": "Control"})

    desired_order = ["Control", "AD", "PD", "FTD", "StrokeTIA"]
    summary["Target"] = pd.Categorical(
        summary["Target"], categories=desired_order, ordered=True)
    summary = summary.sort_values("Target").reset_index(drop=True)

    with pd.option_context("display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))
    

if __name__ == "__main__":
    print_kshot_results()