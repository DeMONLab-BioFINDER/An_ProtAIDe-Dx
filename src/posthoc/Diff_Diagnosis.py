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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from src.utils.io import list2txt, load_pkl
from src.preproc.split_misc import train_val_split
from src.preproc.preproc_misc import feature_normalizing_with_embeddingPCA

import warnings
warnings.filterwarnings("ignore")


def args_parser():
    """
    Parse command-line arguments for DiffDiagnosis.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(prog='DiffDiagnosisArgs')
    parser.add_argument('--ProtAIDeDx_outputs_path', type=str, default='/')
    parser.add_argument('--output_dir', type=str, default='/')
    parser.add_argument('--ref_results_path', type=str, default='/')

    parser.add_argument('--nb_splits', type=int, default=5)
    parser.add_argument('--models', type=list, 
                        default=['M0', 'M1', 'M2', 'M3'])

    args, _ = parser.parse_known_args()
    return args


def get_four_diseases_patients(embedding_df):
    """
    Parse and select patients based on etiologies.

    Args:
        embedding_df (pd.DataFrame): 
            DataFrame containing patient embeddings and metadata

    Returns:
        pd.DataFrame: DataFrame with selected patients
    """
    df = copy.deepcopy(embedding_df)

    # select AD/PD/FTD/Stroke patients 
    df_AD = df[(df['StrokeTIA'] == 0) & (
        df['DxGroup'] == 'AD')]
    df_PD = df[(df['StrokeTIA'] == 0) & (
        df['DxGroup'] == 'LBD') & (
            df['CSF_SAA'] == 'POS'
        )]
    df_FTD = df[(df['StrokeTIA'] == 0) & (
        df['DxGroup'] == 'FTD Spectrum')]
    df_Stroke = df[(df['StrokeTIA'] == 1) & (
        df['DxGroup'] == 'Other'
    )]

    df_AD['Label'] = 0
    df_PD['Label'] = 1
    df_FTD['Label'] = 2
    df_Stroke['Label'] = 3 

    selected_df = pd.concat([df_AD, df_PD, df_FTD, df_Stroke], axis=0)

    return selected_df


def search_bestC_SVM(X_train,
                     y_train,
                     X_val,
                     y_val,
                     C_ranges=[0.001, 0.01, 0.1, 1, 10, 100]):
    """
    Search for the best regularization parameter C for an SVM.

    Args:
        X_train (np.ndarray): Training feature matrix
        y_train (np.ndarray): Training labels
        X_val (np.ndarray): Validation feature matrix
        y_val (np.ndarray): Validation labels
        C_ranges (list, optional): 
            List of regularization parameter values to search. 
            Defaults to [0.001, 0.01, 0.1, 1, 10, 100].

    Returns:
        tuple: 
            Best model, 
            best regularization parameter C, 
            and best validation AUC
    """
    best_C = -1 
    best_auc = 0 
    best_model = None 

    for C in C_ranges:
        model = SVC(kernel='linear', probability=True, 
                    class_weight='balanced',
                    C=C,
                    decision_function_shape='ovr',
                    random_state=42)
        
        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, val_probs, 
                                multi_class='ovr',
                                average='macro')
        if val_auc > best_auc:
            best_auc = val_auc
            best_C = C
            best_model = model

    return best_model, best_C, best_auc


def main(args):
    """
    Main function to run the DiffDiagnosis pipeline.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    features2norm = ['Plasma_pTau217', 'MRI_CTADSign', 'MMSE', 'Plasma_NFL',
                     'Age_at_Visit']
    z_cols = ['Z' + str(i) for i in range(32)]

    feature_sets = {
        'M0': ['Age_at_Visit', 'Sex'],
        'M1': z_cols + ['Age_at_Visit', 'Sex'], 
        'M2': ['Plasma_pTau217', 'MRI_CTADSign', 'MMSE', 'Plasma_NFL',
            'Age_at_Visit', 'Sex'], 
        'M3': z_cols + [
            'Plasma_pTau217', 'MRI_CTADSign', 'MMSE', 'Plasma_NFL',
            'Age_at_Visit', 'Sex']
    }

    # load data 
    embedding_df = pd.read_csv(args.ProtAIDeDx_outputs_path, 
                               low_memory=False)
    input_df = get_four_diseases_patients(embedding_df)

    ref_dict = load_pkl(args.ref_results_path)

    for model in args.models:
        input_df_ = input_df.copy()
        features = feature_sets[model] 
        input_df_.dropna(subset=features, inplace=True)

        subjects = input_df_['PersonGroup_ID']
        # 5-fold cross-validation 
        kf = StratifiedKFold(n_splits=args.nb_splits, 
                             shuffle=True, 
                             random_state=9)

        bca_list = []
        pred_dfs = []
        for fold, (train_val_index, test_index) in enumerate(
            kf.split(input_df_, input_df_['Label'])):
            # train/val/test split
            train_index, val_index = train_val_split(
                train_val_index, 
                input_df_['Label'],
                val_portion=0.25)
            
            train_df, val_df, test_df = \
                input_df_.iloc[train_index], input_df_.iloc[val_index], \
                    input_df_.iloc[test_index]
            X_train, y_train, X_val, y_val, X_test, y_test = \
                feature_normalizing_with_embeddingPCA(
                    train_df, val_df, test_df,
                    features, features2norm, 
                    z_cols, 
                    model,
                    n_components=5)
            
            # hyperparameter search
            best_model, _, _ = search_bestC_SVM(
                X_train, y_train, X_val, y_val)
            
            # evaluation 
            y_pred = best_model.predict(X_test)
            y_predprob = best_model.predict_proba(X_test)
            bca_list.append(balanced_accuracy_score(y_test, y_pred))

            subjs_test = subjects.iloc[test_index] 
            fold_results = pd.DataFrame({
                'PersonGroup_ID': subjs_test.values,
                'Fold': (np.ones((len(subjs_test.values))) * fold).astype(int),
                'true_label': y_test,
                'predicted_label': y_pred,
                'predicted_AD_prob': y_predprob[:, 0],
                'predicted_PD_prob': y_predprob[:, 1],
                'predicted_FTD_prob': y_predprob[:, 2],
                'predicted_Stroke_prob': y_predprob[:, 3]

            })
            pred_dfs.append(fold_results)

        all_preds_df = pd.concat(pred_dfs, axis=0) 
        # save 
        all_preds_df.to_csv(
            os.path.join(args.output_dir, model + '_pred.csv'))
        list2txt(
            bca_list, 
            os.path.join(args.output_dir, model + '_BCA.txt')
        )

        pred_array = all_preds_df[[
            'predicted_AD_prob', 'predicted_PD_prob', 
            'predicted_FTD_prob', 'predicted_Stroke_prob']].values
        
        ref_array = ref_dict['DiffDiag'][model]
        assert np.allclose(pred_array, 
                           ref_array), "Replication Failed!"
    print(f"Congrats! You have replicated Fig5: Differential Diagnosis")


if __name__ == '__main__':
    main(args_parser())