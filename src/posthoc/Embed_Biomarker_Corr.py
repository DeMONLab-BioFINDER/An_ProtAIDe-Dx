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
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, pointbiserialr
from scipy.cluster.hierarchy import linkage, leaves_list
from src.utils.stats import FDR
from src.utils.io import load_pkl


binary_variables = ['Sex', 'CSF_Ab42', 'CSF_SAA']

biomarkers = ['Age', 'Sex', 'MMSE', 
              'CSF_Ab42/40', 'CSF_pTau217', 'TauPET_MetaROI', 
              'MRI_CTADSign', 'MRI_WholeBrainCT', 'MRI_VentricleVol',
              'MRI_WMH', 'UPDRS', 'CSF_SAA',
              'CSF_GFAP', 'CSF_NFL', 'CSF_YKL40', 
              'CSF_sTREM2', 'CSF_SYT1', 'CSF_SNAP25',
              'CSF_NPTX2', 'CSF_PDGFRB', 'CSF_S100']


def compute_corr_p_mat(ProtAIDeDx_outputs_path,
                       z_dims=32):
    """
    Compute correlation and p-value matrices between embeddings and biomarkers.

    Args:
        ProtAIDeDx_outputs_path (str): Path to the ProtAIDeDx outputs CSV file
        z_dims (int, optional): Number of embedding dimensions. Defaults to 32.
    Returns:
        tuple: Correlation matrix and p-value matrices
    """
    z_cols = ['Z' + str(i) for i in range(z_dims)]
    embedding_df = pd.read_csv(ProtAIDeDx_outputs_path)
    embedding_df.rename(columns={'Age_at_Visit': 'Age'}, inplace=True)
    mapper = {'NEG': 0, 'POS': 1, np.nan: np.nan}
    embedding_df['CSF_SAA'] = embedding_df['CSF_SAA'].map(mapper) 

    z_arr = embedding_df[z_cols].values

    corr_mat = np.zeros((z_dims, len(biomarkers)))
    p_mat = np.zeros((z_dims, len(biomarkers)))

    for i in range(z_dims):
        for j, bio_marker in enumerate(biomarkers):
            z_vec = z_arr[:, i].reshape((-1, ))
            bm_vec = embedding_df[bio_marker].values.reshape((-1, ))
            mask = ~np.isnan(bm_vec)

            if np.sum(mask) < 2:
                r, p = np.nan, np.nan
            else:
                if bio_marker in binary_variables:
                    r, p = pointbiserialr(z_vec[mask], bm_vec[mask])
                else:
                     r, p = pearsonr(z_vec[mask], bm_vec[mask])
            corr_mat[i, j], p_mat[i, j] = r, p
    return corr_mat, p_mat


def reorder_by_clustering(corr_mat, p_mat):
    """
    Reorder correlation and p-value matrices by hierarchical clustering.

    Args:
        corr_mat (np.ndarray): Correlation matrix
        p_mat (np.ndarray): P-value matrix
    Returns:
        tuple: 
            Reordered correlation matrix, 
            reordered p-value matrix, 
            and row order indices
    """
    # Perform hierarchical clustering on rows based on column similarity
    linkage_matrix_rows = linkage(corr_mat, method="average")
    row_order = leaves_list(linkage_matrix_rows)  # Get reordered row indices
    
    # Reorder corr_mat and p_mat by row
    corr_mat_reordered = corr_mat[row_order, :]
    p_mat_reordered = p_mat[row_order, :]
    
    return corr_mat_reordered, p_mat_reordered, row_order


def corr_heatmap_with_star(corr_mat, 
                           p_mat,
                           z_cols,
                           biomarkers,
                           fig_save_path,
                           fdr_alpha=0.05):
    """
    Plot a heatmap of correlations with FDR-corrected significance stars.

    Args:
        corr_mat (np.ndarray): Correlation matrix
        p_mat (np.ndarray): P-value matrix
        z_cols (list): List of embedding dimension column names
        biomarkers (list): List of biomarker names
        fig_save_path (str): Path to save the figure
        fdr_alpha (float, optional): FDR significance level. Defaults to 0.05.
    """
    p_list = list(p_mat.reshape((-1, )))
    p_threshold = FDR(p_list, alpha=fdr_alpha)

    # reorder by clustering similarity
    corr_mat_reordered, p_mat_reordered, row_order = \
        reorder_by_clustering(corr_mat, p_mat)
    
    # annotate significant with stars
    sign_annots = np.full_like(corr_mat_reordered, "", dtype=object)
    sign_annots[p_mat_reordered < p_threshold] = "*"

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        corr_mat_reordered,
        annot=False, 
        xticklabels=biomarkers,
        yticklabels=[z_cols[i] for i in row_order], 
        fmt='',
        cmap='coolwarm',
        cbar_kws={'label': 'Corr'},
        vmin=-0.2, 
        vmax=0.2,
        linewidths=0.5
    )
    for i in range(corr_mat_reordered.shape[1]):  # Columns
        for j in range(corr_mat_reordered.shape[0]):  # Rows
            ax.text(
                i + 0.5, j + 1.4,
                sign_annots[j, i],
                ha='center', va='bottom',
                fontsize=16, color="white"
            )
    

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), 
                       rotation=90, ha='center', fontsize=10)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels(), 
                       rotation=0, ha='right', fontsize=10)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('left')
    for col in range(1, corr_mat_reordered.shape[0]):
        ax.hlines(col, xmin=0, xmax=corr_mat_reordered.shape[1], 
                  colors='black', linewidth=2)
    for col in range(1, corr_mat_reordered.shape[1]):
        ax.vlines(col, ymin=0, ymax=corr_mat_reordered.shape[0], 
                  colors='black', linewidth=2)
    heatmap_box = Rectangle(
        (0, 0), corr_mat_reordered.shape[1], corr_mat_reordered.shape[0],
        fill=False, edgecolor='black', linewidth=3, clip_on=False
    )
    ax.tick_params(axis='x', length=5, width=3)
    ax.tick_params(axis='y', length=5, width=3, pad=20)
    ax.add_patch(heatmap_box)
    
    plt.title('Correlation(Embedding, Biomarkers)', fontsize=14)
    plt.xlabel('Biomarkers', fontsize=14)
    plt.ylabel('Embedding', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=400)
    plt.close()


def get_corr_p_table_withCI(
    corr_mat,
    p_mat,
    p_threshold,
    x_tick_labels,
    y_tick_labels,
    results_save_path,
    n: int = 1787,
    alpha: float = 0.05,
    corr_decimals: int = 2,
    ci_decimals: int = 2,
):
    """
    Generate a table with correlation coefficients, CIs, and significance.

    Args:
        corr_mat (np.ndarray): Correlation matrix
        p_mat (np.ndarray): P-value matrix
        p_threshold (float): P-value significance threshold
        x_tick_labels (list): List of x-axis tick labels
        y_tick_labels (list): List of y-axis tick labels
        results_save_path (str): Path to save the results table
        n (int, optional): 
            Sample size for confidence interval calculation. Defaults to 1787.
        alpha (float, optional): 
            Significance level for confidence intervals. Defaults to 0.05.
        corr_decimals (int, optional): 
            Decimal places for correlation coefficients. Defaults to 2.
        ci_decimals (int, optional): 
            Decimal places for confidence intervals. Defaults to 2.
    """
    corr = np.asarray(corr_mat, dtype=float)
    pval = np.asarray(p_mat, dtype=float)

    if corr.shape != pval.shape:
        raise ValueError(
            f"corr_mat shape {corr.shape} != p_mat shape {pval.shape}")
    n_embed, P = corr.shape

    if len(x_tick_labels) != P:
        raise ValueError(f"len(x_tick_labels)={len(x_tick_labels)} != P={P}")
    if len(y_tick_labels) != n_embed:
        raise ValueError(
            f"len(y_tick_labels)={len(y_tick_labels)} != N={n_embed}")

    if not (0 < float(p_threshold) <= 1):
        raise ValueError("p_threshold must be in (0, 1].")
    if not (isinstance(n, int) and n > 3):
        raise ValueError("n must be an int > 3 for Fisher z CI.")
    if not (0 < float(alpha) < 1):
        raise ValueError("alpha must be in (0, 1).")

    # z critical values (common levels); fallback to 1.96
    zcrit_lookup = {
        0.10: 1.6448536269514722,  # 90%
        0.05: 1.959963984540054,   # 95%
        0.01: 2.5758293035489004,  # 99%
    }
    zcrit = zcrit_lookup.get(float(alpha), 1.959963984540054)
    se = 1.0 / np.sqrt(n - 3)

    def fisher_ci_str(r: float) -> str:
        if not np.isfinite(r):
            return np.nan
        r_clip = float(np.clip(r, -0.999999999, 0.999999999))
        z = np.arctanh(r_clip)
        lo = np.tanh(z - zcrit * se)
        hi = np.tanh(z + zcrit * se)
        return f"[{lo:.{ci_decimals}f}, {hi:.{ci_decimals}f}]"


    tabel = pd.DataFrame(
        index=pd.Index(y_tick_labels, name="Embedding"))

    for j, biomarker in enumerate(x_tick_labels):
        r = corr[:, j]
        p = pval[:, j]

        tabel[f"{biomarker}_Corr"] = np.round(r, corr_decimals)
        tabel[f"{biomarker}_CI"] = [fisher_ci_str(v) for v in r]
        tabel[f"{biomarker}_Sig"] = (p <= float(p_threshold))

    tabel = tabel.reset_index()
    tabel.to_csv(results_save_path, index=False)


def args_parser():
    """
    Argument parser for Embedding-Biomarker Correlation analysis.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(prog='EmbBMCorrArgs')

    parser.add_argument(
        '--ProtAIDeDx_outputs_path', type=str,
        default='./results/replica/Fig4_BF2/ProtAIDeDx/ProtAIDeDx_outputs.csv')
    parser.add_argument('--output_dir', type=str, 
                        default='./results/replica/Fig4_BF2/EmbBiomarkerCorr')
    parser.add_argument('--ref_results_path', type=str, default='/')
    parser.add_argument('--z_dims', type=int, default=32)
    parser.add_argument('--fdr_alpha', type=float, default=0.05)

    args, _ = parser.parse_known_args()
    return args


def main(args):
    """
    Main function for Embedding-Biomarker Correlation analysis.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    corr_mat, p_mat = compute_corr_p_mat(
        ProtAIDeDx_outputs_path=args.ProtAIDeDx_outputs_path,
        z_dims=args.z_dims
    )

    z_cols = ['Z' + str(i) for i in range(args.z_dims)]

    # plot 
    corr_heatmap_with_star(
        corr_mat=corr_mat,
        p_mat=p_mat,
        z_cols=z_cols,
        biomarkers=biomarkers,
        fig_save_path=os.path.join(
            args.output_dir, 'Fig4_BF2_EmbBiomarkerCorr.png'),
        fdr_alpha=args.fdr_alpha
    )
    # save results
    p_list = list(p_mat.reshape((-1, )))
    p_threshold = FDR(p_list, alpha=args.fdr_alpha)

    get_corr_p_table_withCI(
        corr_mat=corr_mat,
        p_mat=p_mat,
        p_threshold=p_threshold,
        x_tick_labels=biomarkers,
        y_tick_labels=z_cols,
        results_save_path=os.path.join(
            args.output_dir, 'Fig4_BF2_EmbBiomarkerCorr_results.csv'),
        n=1787,
        alpha=0.05,
        corr_decimals=2,
        ci_decimals=2
    )

    ref_dict = load_pkl(args.ref_results_path)

    pred_arr = np.round(corr_mat, 2)
    ref_arr = ref_dict['biomarker_corr']

    assert np.allclose(pred_arr, ref_arr), "Replication Failed!"
    print("Congrats! You have replicated Fig4: Embedding-Biomarker Corr")


if __name__ == '__main__':
    main(args_parser())