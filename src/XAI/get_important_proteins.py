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
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.cluster.hierarchy import linkage, dendrogram
from statsmodels.stats.multitest import multipletests
from src.utils.io import load_pkl


def compute_permFIT_FoldCounts(results_dir,
                               aptamers_info_path,
                               nb_folds=10,
                               p_threshold=0.05):
    """
    Summarize importance by permFIT across folds.

    Args:
        results_dir (str): Directory containing results.
        aptamers_info_path (_type_): _description_
        nb_folds (int, optional): _description_. Defaults to 10.
        p_threshold (float, optional): _description_. Defaults to 0.05.

    Returns:
        pd.DataFrame: Summary dataframe of protein importance across folds.
    """
    for fold in range(nb_folds):
        fold_dir = 'fold_' + str(fold)
        results_df = pd.read_csv(
            os.path.join(results_dir, fold_dir, 
                         'feature_importance_PermFIT.csv'))
        
        results_df['PVal_adj'] = multipletests(
            results_df['PVal'], method='fdr_bh')[1]
        results_df['FoldCount'] = (results_df['PVal_adj'] <= p_threshold) * 1
        if fold == 0:
            stats_df = deepcopy(results_df)
        else:
            stats_df = pd.concat(
                [stats_df, results_df], axis=0, ignore_index=True)

    stats_df = stats_df[[ 'Protein', 'Target', 'FoldCount']]
    stats_df = stats_df.groupby(
        ['Protein', 'Target'], as_index=False)['FoldCount'].sum()


    summary_df = stats_df.pivot_table(
        index='Protein', columns='Target', values='FoldCount', 
        aggfunc='sum', fill_value=0)
    summary_df.columns = [f"{col}_FoldCount" for col in summary_df.columns]
    summary_df = summary_df.reset_index()
    summary_df = summary_df[['Protein', 'CU_FoldCount',
                             'AD_FoldCount', 'PD_FoldCount', 'FTD_FoldCount',
                             'ALS_FoldCount', 'StrokeTIA_FoldCount']]
    summary_df = summary_df.rename(columns={"Protein": "Feature"})
    
    # append EntrezGeneSymbol column
    aptamers_info_df = pd.read_csv(aptamers_info_path,
                                   usecols=['Feature', 'EntrezGeneSymbol'])
    
    summary_df = summary_df.merge(aptamers_info_df,
                                  on=['Feature'],
                                  how='left')
    # save 
    summary_df.to_csv(os.path.join(results_dir, 'protein_importance.csv'),
                      index=False)
    
    return summary_df


def importance_plot(df, fig_save_path):
    """
    Plot a heatmap of protein importance across conditions.

    Args:
        df (pd.DataFrame): DataFrame containing protein importance data.
        fig_save_path (str): Path to save the generated heatmap figure.
    """
   
    # Extract conditions (rows) and proteins (columns)
    conditions = ["CU", "AD", "PD", "FTD", "ALS", "StrokeTIA"]
    count_cols = [f"{cond}_FoldCount" for cond in conditions]
    df = df[(df[count_cols] >= 4).any(axis=1)]
    
    proteins = df['EntrezGeneSymbol'].to_list()
    
    # Get foldCount
    
    
    # Transform data for heatmap
    count_values = df[count_cols].T # Shape: (6 conditions, N proteins)

    count_values.index = conditions
    count_values.columns = proteins

    # reorder
    # Perform hierarchical clustering on columns (proteins)
    linkage_matrix = linkage(count_values.T, method='ward')

    # Get the order of columns based on clustering
    dendro = dendrogram(linkage_matrix, no_plot=True)
    ordered_columns = np.array(count_values.columns)[dendro['leaves']]
    # Reorder columns in z_values and z_values_masked
    count_values = count_values[ordered_columns]
    
    colors = [
        "#CFD8DC",  # grey
        "#cceeff",  # Light blue
        "#99ddff",  # Sky blue
        "#66ccff",  # Light cyan
        "#33bbff",  # Deep cyan
        "#0099ff",  # Bright blue
        "#007acc",  # Medium blue
        "#005b99",  # Steel blue
        "#003d66",  # Dark blue
        "#002040",  # Navy
        "#001020",  # Deepest navy
    ]
    cmap = ListedColormap(colors)

    # Create heatmap
    bound = np.arange(0, 12)
    norm = BoundaryNorm(boundaries=bound, ncolors=11)
    fig, ax = plt.subplots(figsize=(24, 6))
    sns.heatmap(count_values, 
                cmap=cmap, 
                norm=norm,
                annot=False,
                linewidths=0.5, 
                ax=ax,
                square=True,
                cbar=False)

    # Formatting
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center",
                       fontsize=20)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels(), 
                       rotation=0, ha='center', fontsize=18)


    for col in range(1, count_values.shape[0]):
        ax.hlines(col, xmin=0, xmax=count_values.shape[1], 
                  colors='black', linewidth=3)
    for col in range(1, count_values.shape[1]):
        ax.vlines(col, ymin=0, ymax=count_values.shape[0], 
                  colors='black', linewidth=3)
    heatmap_box = Rectangle(
        (0, 0), count_values.shape[1], count_values.shape[0],
        fill=False, edgecolor='black', linewidth=5, clip_on=False
    )
    ax.tick_params(axis='x', length=10, width=5)
    ax.tick_params(axis='y', length=10, width=5, pad=25)

    ax.add_patch(heatmap_box)
    ax.set_title("")
    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=300)
    plt.cla()


def args_parser():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(prog='Args')
    parser.add_argument('--results_dir', type=str, default='/')
    parser.add_argument('--aptamers_info_path', type=str, default='/')
    parser.add_argument('--ref_results_path', type=str, default='/')

    parser.add_argument('--nb_folds', type=int, default=10)
    parser.add_argument('--p_threshold', type=float, default=0.05)

    args, _ = parser.parse_known_args()
    return args


def main(args):
    """
    Main function to compute protein importance and generate heatmap.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    summary_df = compute_permFIT_FoldCounts(
        args.results_dir,
        args.aptamers_info_path,
        args.nb_folds,
        args.p_threshold
    )

    importance_plot(summary_df,
                    os.path.join(args.results_dir, 
                                 'Fig3_ImportantProteins.png'))
    
    # check whether has been replicated

    ref_dict = load_pkl(args.ref_results_path)
    conditions = ["CU", "AD", "PD", "FTD", "ALS", "StrokeTIA"]
    count_cols = [f"{cond}_FoldCount" for cond in conditions]
    assert np.allclose(summary_df[count_cols].values,
                       ref_dict['FoldCountArray']), "Replication Failed!"
    
    print("Congrats! You have replicated Fig3: permFIT")


if __name__ == '__main__':
    main(args_parser())


