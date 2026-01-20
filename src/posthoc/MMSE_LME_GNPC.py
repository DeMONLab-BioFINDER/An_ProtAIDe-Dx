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
import matplotlib.pyplot as plt
from patsy import dmatrix
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from src.utils.io import load_pkl

import warnings 
warnings.filterwarnings('ignore')


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
        "MMSE ~ Age + C(Sex) + C(Site) + C(BaselineDX) + C(Group) * Year" 

    model = smf.mixedlm(fixed_effect, data=df, groups=df[subjCol], 
                        re_formula=random_effect)
    fitted_lme = model.fit(method="lbfgs")

    return fitted_lme


def lme_result_postproc(fitted_lme, result_save_path):
    """
    Post-process LME results and save to a CSV file.

    Args:
        fitted_lme (statsmodels.regression.mixed_linear_model.MixedLMResults): 
            Fitted LME model results
        result_save_path (str): Path to save the results CSV file
    Returns:
        pd.DataFrame: DataFrame containing the post-processed LME results
    """
    # Build a tidy fixed-effects table
    fe_names = fitted_lme.fe_params.index  # fixed-effect coefficient names
    fe_coefs = fitted_lme.fe_params
    fe_se = fitted_lme.bse.reindex(fe_names)
    fe_t = fitted_lme.tvalues.reindex(fe_names)
    fe_p = fitted_lme.pvalues.reindex(fe_names)
    fe_ci = fitted_lme.conf_int().loc[fe_names]
    fe_ci.columns = ["ci_low", "ci_high"]

    # FDR-BH correction on fixed-effects p-values (ignore NaNs safely)
    pvals = fe_p.values.astype(float)
    mask = np.isfinite(pvals)
    # If all p-values are NaN, avoid calling multipletests
    if mask.any():
        _, p_fdr, _, _ = multipletests(pvals[mask], method='fdr_bh')
        p_fdr_full = np.full_like(pvals, np.nan, dtype=float)
        p_fdr_full[mask] = p_fdr
    else:
        p_fdr_full = np.full_like(pvals, np.nan, dtype=float)

    fe_table = pd.concat(
        [
            fe_coefs.rename("coef"),
            fe_se.rename("se"),
            fe_t.rename("z_or_t"),
            fe_p.rename("p"),
            pd.Series(p_fdr_full, index=fe_names, name="p_fdr_bh"),
            fe_ci,
        ],
        axis=1
    )

    # save results 
    fe_table.to_csv(result_save_path)

    return fe_table


def lme_plot_byBlPred(df, 
                      group_color_dict,
                      fitted_lme,
                      fig_save_path):
    """
    Plot MMSE trajectories by baseline prediction groups with 95% CIs.

    Args:
        df (pd.DataFrame): Input DataFrame for LME analysis
        group_color_dict (dict): Dictionary mapping groups to colors
        fitted_lme (statsmodels.regression.mixed_linear_model.MixedLMResults): 
            Fitted LME model results
        fig_save_path (str): Path to save the figure
    """
    year_grid = np.linspace(
        df["Year"].min(), df["Year"].max(), 100)

    # LME fixed-effects predictions (population level)
    Age_mean = df["Age"].mean()
    Sex_ref  = df["Sex"].value_counts().idxmax()
    Site_ref = df["Site"].value_counts().idxmax()
    BaselineDX_ref = df["BaselineDX"].value_counts().idxmax()

    pred_rows = [{"Age": Age_mean, 
                  "Sex": Sex_ref,
                  "Site": Site_ref,
                  "BaselineDX": BaselineDX_ref,
                  "Group": g, "Year": y}
                    for g in np.unique(df["Group"].values) for y in year_grid]
    pred_df = pd.DataFrame(pred_rows)
    pred_df["pred_fixed"] = fitted_lme.predict(exog=pred_df)

    X = dmatrix(
        "1 + Age + Sex + Site + BaselineDX + Group*Year", 
        pred_df, return_type="dataframe")
    
    # Names/order of fixed effects in the fitted model
    fe_names = fitted_lme.fe_params.index.tolist()

    # Align X to fixed-effects order and take only those columns
    X = X.reindex(columns=fe_names, fill_value=0.0)

    # Fixed-effects covariance block
    cov_all = fitted_lme.cov_params()
    cov_fe  = cov_all.loc[fe_names, fe_names]

    # Standard errors via delta method for X
    se = np.sqrt(np.sum((X.values @ cov_fe.values) * X.values, axis=1))
    pred_df["ci_low"]  = pred_df["pred_fixed"] - 1.96 * se
    pred_df["ci_high"] = pred_df["pred_fixed"] + 1.96 * se

    # Plot with CI bands
    plt.figure(figsize=(3, 5))
    for g in np.unique(df["Group"].values):
        sub = pred_df[pred_df["Group"] == g]
        plt.plot(sub["Year"], 
                 sub["pred_fixed"], 
                 color=group_color_dict[g],
                 label=f"Prediction: {g}",
                 lw=3)
        plt.fill_between(sub["Year"], 
                         sub["ci_low"], 
                         sub["ci_high"], 
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

    xtick_pos = [0, 5, 10, 15, 20]
    plt.xticks(ticks=xtick_pos, labels=[str(x) for x in xtick_pos])
    y_ticks = [16, 18, 20, 22, 24, 26, 28, 30]
    plt.yticks(y_ticks, labels=[str(y) for y in y_ticks])
    plt.ylim(16, 33)
    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=400)


def get_ProtAIDeDx_prediction(model_prediction_dir,
                              nb_folds=10):
    """
    Get ProtAIDeDx model predictions aggregated across folds.

    Args:
        model_prediction_dir (str): 
            Directory containing model prediction subdirectories for each fold
        nb_folds (int, optional): Number of folds. Defaults to 10.
    Returns:
        pd.DataFrame: 
        Aggregated DataFrame of ProtAIDeDx model predictions across folds
    """
    targets = ['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA']
    gt_cols = [t + '-GT' for t in targets]
    pred_cols = [t + '-PredLabel' for t in targets]
    cols2read = ['PersonGroup_ID'] + gt_cols + pred_cols

    # get model predictions
    pred_df = pd.DataFrame(columns=cols2read + ['Fold'])
    for fold in range(nb_folds):
        fold_dir = 'fold_' + str(fold)
        fold_pred_df = pd.read_csv(
            os.path.join(model_prediction_dir, fold_dir, 'test_results.csv'),
            usecols=cols2read)
        fold_pred_df['Fold'] = fold

        pred_df = pd.concat([pred_df, fold_pred_df], axis=0)
    
    # select subjects
    pred_df = pred_df[pred_df[gt_cols].sum(axis=1) == 1]
    pred_df = pred_df[pred_df[pred_cols].sum(axis=1) == 1]
    pred_df['BaselineDX'] = np.nan 
    pred_df.loc[pred_df["CU-GT"] == 1, "BaselineDX"] = "CU"
    pred_df.loc[pred_df["AD-GT"] == 1, "BaselineDX"] = "AD"
    pred_df.loc[pred_df["PD-GT"] == 1, "BaselineDX"] = "PD"
    pred_df.loc[pred_df["FTD-GT"] == 1, "BaselineDX"] = "FTD"
    pred_df.loc[pred_df["ALS-GT"] == 1, "BaselineDX"] = "ALS"
    pred_df.loc[pred_df["StrokeTIA-GT"] == 1, "BaselineDX"] = "StrokeTIA"

    return pred_df


def lme_input_by_predGroup(multiVisit_df,
                           subj_id_list,
                           predGroup):
    """
    Prepare input DataFrame for LME analysis based on prediction group.

    Args:
        multiVisit_df (pd.DataFrame): DataFrame containing multi-visit data
        subj_id_list (list): List of subject IDs to include
        predGroup (str): Prediction group label

    Returns:
        pd.DataFrame: Prepared DataFrame for LME analysis
    """

    selected_df = multiVisit_df[
        multiVisit_df['PersonGroup_ID'].isin(subj_id_list)]
    # compute internals in unit of years
    selected_df = selected_df.sort_values(
        ['PersonGroup_ID', 'Visit']).reset_index(drop=True)
    selected_df['Year'] = selected_df.groupby(
        'PersonGroup_ID')['Age_at_Visit'].transform(lambda x: x - x.iloc[0])
    # additional processing steps
    selected_df['Group'] = predGroup
    selected_df.dropna(subset=["MMSE", "Age_at_Visit", "Sex"], inplace=True)
    selected_df['Sex'] -= 1
    # rename
    selected_df = selected_df[
        ['PersonGroup_ID', 'Visit', 'Contributor_Code',
         'Year', 'MMSE', 'Age_at_Visit', 'Sex', 'Group']]
    selected_df = selected_df.rename(columns={
        "PersonGroup_ID": 'SubjID',
        'Age_at_Visit': 'Age',
        'Contributor_Code': 'Site'
    })
    
    return selected_df


def args_parser():
    """
    Parse command-line arguments for LME analysis.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(prog='LMEArgs')

    parser.add_argument('--ProtAIDeDx_prediction_dir', type=str, default='/')
    parser.add_argument('--multiVisit_data_path', type=str, default='/')
    parser.add_argument('--nb_folds', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='/')
    parser.add_argument('--ref_results_path', type=str, default='/')

    args, _ = parser.parse_known_args()
    return args


def main(args):
    """
    Main function to run LME on MMSE trajectories by ProtAIDeDx predictions.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    pred_df = get_ProtAIDeDx_prediction(args.ProtAIDeDx_prediction_dir)
    multiVisit_df = pd.read_csv(args.multiVisit_data_path, 
                                low_memory=False)
    
    prediction_groups = ['CU', 'AD', 'PD', 'StrokeTIA']
    plot_df = pd.DataFrame(
        columns=['SubjID', 'Visit', 'Site', 
                 'Year', 'MMSE', 'Age', 'Sex', 'Group'])

    # select subjects by prediction group
    for y_pred in prediction_groups:        
        subj_ID_list = list(np.unique(pred_df[
            (pred_df[y_pred + '-PredLabel'] == 1)]['PersonGroup_ID'].values))
        group_plot_df = lme_input_by_predGroup(
            multiVisit_df, subj_ID_list, y_pred)
        plot_df = pd.concat([plot_df, group_plot_df], axis=0)

    # z-normalization 
    plot_df['Age'] = (
        plot_df['Age'] - plot_df['Age'].mean())/ plot_df['Age'].std()
    
    # append BaselineDX 
    plot_df['BaselineDX'] = np.nan 
    subjs2plot = list(np.unique(plot_df['SubjID']))
    for subj in subjs2plot:
        baselineDX = pred_df[
            pred_df['PersonGroup_ID'] == subj]['BaselineDX'].values[0]
        plot_df.loc[plot_df["SubjID"] == subj, 'BaselineDX'] = baselineDX
    plot_df.dropna(subset=['BaselineDX'], inplace=True)

    # fit LME models 
    fitted_lme = fit_lme_byBlPred(plot_df)
    # save results 
    lme_results_df = lme_result_postproc(fitted_lme, 
                                         os.path.join(
                                             args.output_dir, 
                                             'MMSE_LME_byBlPred_results.csv'))

    # plot 
    group_color_dict = {"CU": "blue", "AD": "red", "PD":"green",
                        'FTD': 'orange', 'ALS': 'purple',
                        'StrokeTIA': 'black'}
    
    lme_plot_byBlPred(plot_df,
                      group_color_dict,
                      fitted_lme,
                      os.path.join(args.output_dir, 'Fig5_LME_GNPC.png'))
    
    # check reference 
    z_vec = lme_results_df['z_or_t'].values.reshape((-1, 1))
    ref_dict = load_pkl(args.ref_results_path)
    assert np.allclose(z_vec, ref_dict['z']), "Replication Failed!"
    print("Congrats! You have replicated Fig5: MMSE Trajectory (GNPC)")
    

if __name__ == '__main__':
    main(args_parser())

