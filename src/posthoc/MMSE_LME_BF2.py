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
from patsy import dmatrix
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
from src.posthoc.MMSE_LME_GNPC import lme_result_postproc
from src.utils.io import load_pkl

import warnings 
warnings.filterwarnings('ignore')


def lme_input_by_predGroup(
    MMSE_longitudinal_csv_path,
    kshot_predictions_csv_path,
    baselineDx_groups=['MCI']):
    """
    Generate LME input data by baseline prediction groups.

    Args:
        MMSE_longitudinal_csv_path (str): 
            Path to the MMSE longitudinal CSV file
        kshot_predictions_csv_path (str): 
            Path to the k-shot predictions CSV file
        baselineDx_groups (list, optional): 
            List of baseline diagnosis groups to include. Defaults to ['MCI'].

    Returns:
        pd.DataFrame: Merged DataFrame suitable for LME analysis
    """
    # load data 
    long_df = pd.read_csv(MMSE_longitudinal_csv_path)
    pred_df = pd.read_csv(kshot_predictions_csv_path)
    pred_df['Visit'] = 0

    # filter data
    pred_cols = ['Normal Control-PredLabel', 'AD-PredLabel']
    pred_df = pred_df[
        ['PersonGroup_ID', 'Visit', 'diagnosis_baseline_variable'] + pred_cols]
    pred_df = pred_df[
        pred_df['diagnosis_baseline_variable'].isin(baselineDx_groups)]
    
    long_cols = ['PersonGroup_ID', 'Visit', 'Age_at_Visit', 'Sex', 
                 'MMSE', 'Year', 'diagnosis_baseline_variable']
    long_df = long_df[long_cols]
    long_df = long_df[
        long_df['diagnosis_baseline_variable'].isin(baselineDx_groups)]
    

    # merge pred_df and long_df 
    subj_list = list(np.unique(pred_df['PersonGroup_ID'].values))
    rows = []
    for subj in subj_list:
        pred_vec = list(pred_df[
            pred_df['PersonGroup_ID'] == subj][pred_cols].values.reshape(
                (len(pred_cols), )))
        visits = list(
            long_df[long_df['PersonGroup_ID'] == subj]['Visit'].values)
        for visit in visits:            
            long_visit_vec = long_df[
                (long_df['PersonGroup_ID'] == subj) & (
                    long_df['Visit'] == visit)][
                    long_cols].values 
            long_visit_vec = list(long_visit_vec.reshape((len(long_cols), )))
            row = long_visit_vec + pred_vec
            rows.append(row)
    merged_cols = long_cols + pred_cols
    merged_df = pd.DataFrame(data=rows, columns=merged_cols)
    assert len(pred_df) == len(
        np.unique(merged_df['PersonGroup_ID'].values)), "Unmatched"

    # remap columns 
    merged_df = merged_df.rename(
        columns={
            'PersonGroup_ID': 'SubjID',
            'Age_at_Visit': 'Age',
            'Sex': 'Sex',
            'MMSE': 'MMSE',
            'diagnosis_baseline_variable': 'BaselineDX'
        }
    )

    # filter conflicted predictions 
    merged_df = merged_df[merged_df[pred_cols].sum(axis=1) == 1]
    merged_df['Group'] = merged_df[
        pred_cols].idxmax(axis=1)
    group_map = {
        'Normal Control-PredLabel': 'Control',
        'AD-PredLabel': 'AD',
        'LBD-PredLabel': 'PD',
        'FTD Spectrum-PredLabel': 'FTD',
        'StrokeTIA-PredLabel': 'Stroke'
    }
    merged_df['Group'] = merged_df['Group'].map(group_map)

    cols2return = [
        'SubjID', 'Visit', 'Year', 'Age', 'Sex', 'MMSE', 'BaselineDX', 
        'Group']
    merged_df = merged_df[cols2return]
    merged_df.dropna(subset=cols2return, inplace=True)

    return merged_df


def fit_lme_byBlPred(df,
                     subjCol='SubjID',
                     random_effect="~Year"):
    """
    Fit a linear mixed-effects model by baseline prediction groups.

    Args:
        df (pd.DataFrame): Input DataFrame for LME analysis
        subjCol (str, optional): 
            Subject identifier column name. Defaults to 'SubjID'.
        random_effect (str, optional): 
            Random effects formula. Defaults to "~Year".

    Returns:
        statsmodels.regression.mixed_linear_model.MixedLMResults: 
        Fitted LME model results
    """
    fixed_effect = \
        "MMSE ~ Age + C(Sex) + C(BaselineDX) + C(Group) * Year"

    model = smf.mixedlm(fixed_effect, data=df, groups=df[subjCol], 
                        re_formula=random_effect)
    fitted_lme = model.fit(method="lbfgs")

    return fitted_lme


def lme_plot_CI(df,
                group_color_dict, 
                lme_results, 
                fig_save_path):
    """
    Only plot 95% Confidence interval of fitted LME fixed effects 

    Args:
        df (pd.DataFrame): Input DataFrame for LME analysis
        group_color_dict (dict): Dictionary mapping groups to colors
        lme_results : Fitted LME model results
        fig_save_path (str): Path to save the figure
    """
    year_grid = np.linspace(df["Year"].min(), df["Year"].max(), 100)

    # LME fixed-effects predictions (population level)
    Age_mean = df["Age"].mean()
    Sex_ref  = df["Sex"].value_counts().idxmax()
    BaselineDX_ref = df["BaselineDX"].value_counts().idxmax()

    pred_rows = [{"Age": Age_mean, 
                  "Sex": Sex_ref,
                  'BaselineDX': BaselineDX_ref,
                  "Group": g, 
                  "Year": y}
                    for g in np.unique(df["Group"].values) for y in year_grid]

    pred_df = pd.DataFrame(pred_rows)
    pred_df["pred_fixed"] = lme_results.predict(exog=pred_df)

    # Design matrix aligned with the model's fixed effects
    X = dmatrix("1 + Age + Sex + BaselineDX + Group*Year", 
                pred_df, return_type="dataframe")
    fe_names = lme_results.fe_params.index.tolist()
    X = X.reindex(columns=fe_names, fill_value=0.0)

    # Fixed-effects covariance block
    cov_all = lme_results.cov_params()
    cov_fe  = cov_all.loc[fe_names, fe_names]  # <-- key fix

    se = np.sqrt(np.sum((X.values @ cov_fe.values) * X.values, axis=1))
    pred_df["ci_low"]  = pred_df["pred_fixed"] - 1.96 * se
    pred_df["ci_high"] = pred_df["pred_fixed"] + 1.96 * se

    # Plot with CI bands
    plt.figure(figsize=(3, 5))
    for g in np.unique(df["Group"].values):
        sub = pred_df[pred_df["Group"] == g]
        plt.plot(sub["Year"], sub["pred_fixed"], 
                 color=group_color_dict[g],
                 label=f"Prediction: {g}",
                 lw=3)
        plt.fill_between(sub["Year"], sub["ci_low"], sub["ci_high"], 
                         color=group_color_dict[g],
                         alpha=0.1)

    plt.title('MMSE trajectory by BlPred')
    plt.xlabel('Years from baseline')
    plt.ylabel('MMSE')
    plt.legend(title='Baseline Prediction')
    

    ax = plt.gca()
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)

    plt.tick_params(axis='x', length=5, width=3)
    plt.tick_params(axis='y', length=5, width=3)

    xtick_pos = [0, 1, 2, 3, 4, 5, 6]
    plt.xticks(ticks=xtick_pos, 
               labels=[str(x) for x in xtick_pos])
    y_ticks = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    plt.yticks(y_ticks, 
               labels=[str(x) for x in y_ticks])
    plt.ylim(10, 31)
    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=400)


def args_parser():
    """
    Parse command-line arguments for LME analysis.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(prog='LMEArgs')

    parser.add_argument(
        '--MMSE_longitudinal_csv_path', type=str, 
        default='./data/replica/raw/BF2_MMSE_Long.csv')
    parser.add_argument(
        '--kshot_predictions_csv_path', type=str, 
        default='./results/replica/Fig4_BF2/kshot/KShot_LR_Predictions.csv')
    parser.add_argument(
        '--output_dir', type=str, 
        default='./results/replica/Fig5_BF2/MMSE_Trajectory')
    parser.add_argument('--ref_results_path', type=str, default='/')
    parser.add_argument('--baselineDx_groups', type=list, default=['MCI'])

    args, _ = parser.parse_known_args()
    return args


def main(args):
    """
    Main function to run LME analysis and plotting.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    # prepare LME input data 
    LME_input_df = lme_input_by_predGroup(
        MMSE_longitudinal_csv_path=args.MMSE_longitudinal_csv_path,
        kshot_predictions_csv_path=args.kshot_predictions_csv_path,
        baselineDx_groups=args.baselineDx_groups
    )

    # fit LME model 
    fitted_lme = fit_lme_byBlPred(LME_input_df)

    # post-process LME results 
    lme_results_df = lme_result_postproc(
        fitted_lme,
        result_save_path=os.path.join(
            args.output_dir, 'Fig5_BF2_MMSE_MCI.csv')
    )

    # plot LME results with CI bands 
    group_color_dict = {
        'Control': 'blue',
        'AD': 'red',
    }
    lme_plot_CI(
        LME_input_df,
        group_color_dict,
        fitted_lme,
        fig_save_path=os.path.join(
            args.output_dir, 'Fig5_BF2_MMSE_MCI.png')
    )

    # load reference results
    ref_dict = load_pkl(args.ref_results_path)
    ref_z = ref_dict['MMSE_LME_Z']
    z_vec = lme_results_df['z_or_t'].values.reshape(-1, 1)
    assert np.allclose(z_vec, ref_z), "Replication Failed!"
    print("Congrats! You have replicated Fig5: MMSE Trajectory (BF2)")


if __name__ == '__main__':
    main(args_parser())