#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os 
import copy
import argparse
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from src.k_shot.misc import k_shot_sampling, calc_k_list
from src.utils.io import df2csv, load_pkl, dict2df
from src.utils.metrics import clf_metrics
from src.ProtAIDeDx.misc.nn_helper import load_ProbaThresholds


target_order = {
    "Recruited Control": 0, "AD": 1, "PD": 2,
    "FTD": 3, "ALS": 4, "StrokeTIA": 5}

map_dict={
    'Normal Control': 'CU',
    'AD': 'AD',
    'LBD': 'PD',
    'FTD Spectrum': 'FTD',
    'StrokeTIA': 'StrokeTIA'
}


def args_parser():
    """
    Parse command-line arguments for k-shot learning.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(prog='KShotEmbedArgs')
    parser.add_argument('--model', type=str, default='LR')
    parser.add_argument('--suffix', type=str, default='LOSO')
    parser.add_argument('--split', type=str, default='site_C')

    parser.add_argument('--input_dir', type=str, default='/')
    parser.add_argument('--results_dir', type=str, default='/')
    parser.add_argument('--ref_results_path', type=str, default='/')

    parser.add_argument('--probaThresholds_path', type=str, default='/')
    parser.add_argument('--embed_file', type=str, 
                        default='ProtAIDeDx_outputs.csv')
    parser.add_argument('--nb_repeats', type=int, default=20)
    parser.add_argument('--k_shot_list', type=list,
                        default=[100])

    args, _ = parser.parse_known_args() 
    return args


def kshot_learning_LR(K_dev_df,
                      K_test_df,
                      features,
                      target_gt,
                      K, 
                      seed):
    """
    KShot learning with Logistic Regression.

    Args:
        K_dev_df (pd.DataFrame): Development DataFrame for k-shot learning
        K_test_df (pd.DataFrame): Test DataFrame for k-shot learning
        features (list): List of feature column names
        target_gt (str): Ground truth target column name
        K (int): Number of shots
        seed (int): Random seed for reproducibility
    Returns:
        tuple: A tuple containing:
            - clf_met (dict): Classification metrics
            - pred_df (pd.DataFrame): 
                DataFrame with predictions and related information
    """
    dev_X, dev_y = K_dev_df[features].values, K_dev_df[target_gt].values 
    nb_pos = len(K_dev_df[K_dev_df[target_gt] == 1]) 

    if len(features) > 10:
        sm = SMOTE(k_neighbors=nb_pos-1, random_state=42)
        dev_X_resampled, dev_y_resampled = sm.fit_resample(
            dev_X, dev_y)
    else:
        dev_X_resampled, dev_y_resampled = dev_X, dev_y
    
    # Logistic Regression as classifier
    clf = BaggingClassifier(
        estimator=LogisticRegression(
            class_weight='balanced',
            C=0.3,
            penalty='l2'),
        n_estimators=1000,
        max_features=0.8, 
        max_samples=0.8,
        random_state=seed)

    clf.fit(dev_X_resampled, dev_y_resampled)
    K_test_predprob = clf.predict_proba(K_test_df[features])[:, 1]
    K_threshold = 0.5

    # evaluate on test set
    gt = K_test_df[target_gt].values.astype(int)
    mask = ~np.isnan(gt)
    clf_met = clf_metrics(gt[mask], 
                          K_test_predprob[mask], 
                          nb_class=2, threshold=K_threshold)
    
    # get predictions 
    ID_vec = K_test_df['PersonGroup_ID'].values.reshape((-1, 1))
    gt_vec = gt.reshape((-1, 1))
    predprob_vec = K_test_predprob.reshape((-1, 1))
    threshold_vec = np.ones((len(gt_vec), 1)) * K_threshold
    pred_df = pd.DataFrame(
        data=np.concatenate(
            (ID_vec, gt_vec, predprob_vec, threshold_vec), axis=1),
        columns=['PersonGroup_ID', target_gt, 
                 target_gt + '-PredProb', 'Threshold'])
    pred_df['K'] = K
    pred_df['Seed'] = seed

    return clf_met, pred_df


def main(args):
    """
    Main function for k-shot learning with embeddings.

    Args:
        args (argparse.Namespace): Command-line arguments
    """
    # 1. load data & parameters
    embed_df = pd.read_csv(
        os.path.join(args.input_dir, args.embed_file), 
        low_memory=False)

    threshold_list = load_ProbaThresholds(args.probaThresholds_path,
                                          args.suffix,
                                          args.split)
    threshold_list = np.array(threshold_list, dtype=np.float32)

    threshold_dict = {
        'CU': threshold_list[0],
        'AD': threshold_list[1],
        'PD': threshold_list[2],
        'FTD': threshold_list[3],
        'StrokeTIA': threshold_list[5]
    }

    targets2pred = ['Normal Control', 'AD', 'LBD', 
                    'FTD Spectrum', 'StrokeTIA']
    
    z_cols = [
        ele for ele in embed_df.columns.to_list() if ele.startswith('Z')]
    
    # 2. k-shot learning
    for target_gt in targets2pred:
        df_ = copy.deepcopy(embed_df)
        df_.dropna(subset=[target_gt], inplace=True)
        k_shots = calc_k_list(len(df_), 0.5, args.k_shot_list)
        for K in k_shots:
            for seed in range(args.nb_repeats):
                K_dev_df, K_test_df = k_shot_sampling(
                    df_, target_gt, K, seed)
                # kshot learning with LR
                metrics, pred_df = kshot_learning_LR(
                    K_dev_df,
                    K_test_df,
                    z_cols,
                    target_gt,
                    K,
                    seed)
                # save results 
                save_name = \
                    args.model + '_' + target_gt + '_K' + str(K) + '_Seed' \
                        + str(seed) + '_'
                df2csv(
                    dict2df(metrics),
                    os.path.join(
                        args.results_dir,
                        save_name + 'metrics.csv'))
                df2csv(
                    pred_df,
                    os.path.join(
                        args.results_dir,
                        save_name + 'pred.csv'))
    # summarize results across targets & seeds 
    cols2keep = [
        'Visit', 
        'Contributor_Code',
        'Age_at_Visit',
        'Sex',
        'APOE',
        'Years_of_Education',
        'cognitive_status_baseline_variable',
        'diagnosis_baseline_variable',
        'DxGroup',
        'StrokeTIA',
        'MMSE',
        'UPDRS',
        'Plasma_pTau217',
        'Plasma_NFL',
        'TauPET_MetaROI',
        'MRI_CTADSign',
        'MRI_WholeBrainCT',
        'MRI_VentricleVol',
        'MRI_WMH',
        'CSF_Ab42/40',
        'CSF_pTau217',
        'CSF_SAA',
        'CSF_GFAP',
        'CSF_NFL',
        'CSF_YKL40',
        'CSF_sTREM2',
        'CSF_SYT1',
        'CSF_SNAP25',
        'CSF_NPTX2',
        'CSF_PDGFRB',
        'CSF_S100'
    ] + z_cols 

    subjects = np.unique(embed_df['PersonGroup_ID'].values)
    t_df_list = []
    for target_gt in targets2pred:
        summary_df = pd.DataFrame(columns=['PersonGroup_ID', 
                                           target_gt + '-PredLabel', 
                                           target_gt + '-PredProb'])
        for seed in range(args.nb_repeats):
            pred_csv_path = os.path.join(
                args.results_dir,
                args.model + '_' + \
                    target_gt + '_K100' + '_Seed' + str(seed) + '_pred.csv'
            )
            
            pred_df = pd.read_csv(pred_csv_path)
            if np.sum(np.isnan(pred_df[target_gt + '-PredProb'].values)) > 0:
                print(target_gt, 'seed', seed,
                      'has NaN in PredProb!')
            pred_df[target_gt + '-PredLabel'] = (
                pred_df[target_gt + '-PredProb'] > pred_df['Threshold']) * 1
            summary_df = pd.concat([summary_df, 
                                    pred_df[['PersonGroup_ID', 
                                             target_gt + '-PredLabel', 
                                             target_gt + '-PredProb']]],
                                   axis=0)
        # compute major PredLabel and mean PredProb
        summary_df_avg = summary_df.groupby('PersonGroup_ID', 
                                            as_index=False).mean()
        summary_df_avg[target_gt + '-PredLabel'] = np.where(
            summary_df_avg[target_gt + '-PredLabel'] > 0.5, 1, 0)
        
        t_df_list.append(summary_df_avg)
    
    merged_df = t_df_list[0]
    for df in t_df_list[1:]:
        merged_df = merged_df.merge(df, on="PersonGroup_ID", how="left")
    
    merged_df[cols2keep] = np.nan
    for s in subjects:
        s_data_mask = embed_df['PersonGroup_ID'] == s
        s_summary_mask = (merged_df['PersonGroup_ID'] == s)

        for col in cols2keep:
            col_value = embed_df.loc[s_data_mask, col].values[0]
            merged_df.loc[s_summary_mask, col] = col_value
    
    # once predicted as disease, cannot be control
    disease_preds = ["AD-PredLabel", "LBD-PredLabel", 
                     "FTD Spectrum-PredLabel", 
                     "StrokeTIA-PredLabel"]
    mask = merged_df[disease_preds].eq(1).any(axis=1)
    merged_df.loc[mask, 'Normal Control-PredLabel'] = 0
    
    # save final predictions
    merged_df.to_csv(
        os.path.join(
            args.results_dir,
            'KShot_' + args.model + '_Predictions.csv'),
        index=False)
    
    # compare against reference results 
    ref_dict = load_pkl(args.ref_results_path)
    pred_cols = [t + '-PredLabel' for t in targets2pred]
    pred_arr = merged_df[pred_cols].values
    ref_arr = ref_dict['kshot']
    assert np.allclose(pred_arr, 
                       ref_arr,
                       equal_nan=True), "Replication Failed!"
    print("Congrats! You have replicated Fig4: K-shot (BF2)")


if __name__ == '__main__':
    main(args_parser())