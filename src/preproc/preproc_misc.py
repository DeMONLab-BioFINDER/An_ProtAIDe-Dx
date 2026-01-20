#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import numpy as np
from sklearn import impute
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from src.preproc.gauss_rank_scaler import GaussRankScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def stand_norm(df,
               cols, 
               scaler, 
               isTrain=False):
    """
    Z normalization.

    Args:
        df (pd.DataFrame): Dataframe to normalize
        cols (list): Columns to normalize
        scaler (StandardScaler): Fitted scaler class
        isTrain (bool, optional): 
            Whether the data is training data. Defaults to False.

    Returns:
        pd.DataFrame: Normalized dataframe
        StandardScaler: Fitted scaler class
    """
    
    if isTrain:
        scaler = StandardScaler()
        scaler.fit(df[cols])
    df[cols] = scaler.transform(df[cols])
    return df, scaler


def gauss_norm(df,
               cols,
               scaler,
               isTrain=False):
    """
    Gaussian Rank normalization. 

    Args:
        df (pd.DataFrame): Dataframe to normalize
        cols (list): Columns to normalize
        scaler (GaussRankScaler): Fitted scaler class
        isTrain (bool, optional): 
            Whether the data is training data. Defaults to False.

    Returns:
        pd.DataFrame: Normalized dataframe
        GaussRankScaler: Fitted scaler class
    """
    if isTrain:
        scaler = GaussRankScaler()
        scaler.fit(df[cols])
    df[cols] = scaler.transform(df[cols])
    return df, scaler


def knn_impute(df,
               cols,
               imputer=None,
               nb_neighbor=10,
               isTrain=False):
    """
    Use K nearest neighbor to impute missing data.

    Args:
        df (pd.DataFrame): Dataframe to impute
        cols (list): Columns to normalize
        imputer (impute.KNNImputer, optional): 
            Fitted imputer class. Defaults to None.
        nb_neighbor (int, optional): 
            Number of neighbors to use. Defaults to 10.
        isTrain (bool, optional): 
            Whether the data is training data. Defaults to False.

    Returns:
        pd.DataFrame: Imputed dataframe
        impute.KNNImputer: Fitted imputer class
    """
    if isTrain:
        imputer = impute.KNNImputer(n_neighbors=nb_neighbor)
        imputer.fit(df[cols].values)
    # impute
    df[cols] = imputer.transform(df[cols].values)

    return df, imputer


def simple_impute(df,
                  cols,
                  imputer=None,
                  isTrain=False,
                  strategy='mean'):
    """
    Use simple strategy (e.g., mean) to impute missing data.

    Args:
        df (pd.DataFrame): Dataframe to impute
        cols (list): Columns to normalize
        imputer (impute.SimpleImputer, optional): 
            Fitted imputer class. Defaults to None.
        isTrain (bool, optional): 
            Whether the data is training data. Defaults to False.
        strategy (str, optional): Value to impute. Defaults to 'mean'.

    Returns:
        pd.DataFrame: Imputed dataframe
        impute.SimpleImputer: Fitted imputer class
    """
    if isTrain:
        imputer = impute.SimpleImputer(strategy=strategy)
        imputer.fit(df[cols].values)
    # impute
    df[cols] = imputer.transform(df[cols].values)
    return df, imputer


def avgProt_norm(df, prots):
    """
    Normalize proteins by average protein expression level. 

    Args:
        df (pd.DataFrame): Dataframe to normalize
        prots (list): List of proteins to normalize

    Returns:
        pd.DataFrame: Normalized dataframe
    """
    df['avgProt'] = df[prots].mean(axis=1, skipna=True)
    df[prots] = df[prots].div(df['avgProt'], axis=0)
    # drop avgProt column
    df = df.drop('avgProt', axis=1)
    df.reset_index(inplace=True, drop=True)
    return df


def construct_input_dict(ProtArray, isTrain=False):
    """
    Construct input dictionary for model input.

    Args:
        ProtArray (np.ndarray): Array of protein data
        isTrain (bool, optional): 
            Whether the data is training data. Defaults to False.
    Returns:
        dict: Input dictionary for model
    """
    input_dict = dict()
    input_dict['input'] = ProtArray
    input_dict['target'] = dict()
    input_dict['mask'] = dict()
    if isTrain:
        input_dict['mean'] = dict()
        input_dict['std'] = dict()
    return input_dict


def append_continuous_target(input_dict, 
                             df, 
                             target,
                             mean=0, 
                             std=0, 
                             isTrain=False):
    """
    Append a continuous target to input dict.

    Args:
        input_dict (dict): Input dictionary for model
        df (pd.DataFrame): Dataframe containing target data
        target (str): Target variable name
        mean (int, optional): Mean value for normalization. Defaults to 0.
        std (int, optional): 
            Standard deviation for normalization. Defaults to 0.
        isTrain (bool, optional): 
            Whether the data is training data. Defaults to False.

    Returns:
        dict: Updated input dictionary
        float: Mean value used for normalization
        float: Standard deviation used for normalization
    """
    if isTrain:
        # compute train mean and std
        mean, std = df[target].mean(), df[target].std()
        input_dict['mean'][target] = mean
        input_dict['std'][target] = std
    input_dict['target'][target] = (
        (df[target] - mean) / std).values.reshape((-1, 1))
    input_dict['mask'][target] = (~df[target].isna()).values.reshape((-1, 1))
    return input_dict, mean, std


def append_categorical_target(input_dict, 
                              df, 
                              target):
    """
    Append a categorical target to input dict.

    Args:
        input_dict (dict): Input dictionary for model
        df (pd.DataFrame): Dataframe containing target data
        target (str): Target variable name

    Returns:
        dict: Updated input dictionary
    """
    input_dict['target'][target] = df[target].values.reshape((-1, 1))
    input_dict['mask'][target] = (~df[target].isna()).values.reshape((-1, 1))
    return input_dict


def one_hot(label_vec, nb_class=2):
    """
    One-hot encode a label vector.

    Args:
        label_vec (np.ndarray): Array of labels to encode
        nb_class (int, optional): Number of classes. Defaults to 2.
    Returns:
        np.ndarray: One-hot encoded matrix
    """
    nb_samples = len(label_vec)
    label_vec = label_vec.astype(int)
    one_hot_code = np.zeros((nb_samples, nb_class))
    one_hot_code[np.arange(nb_samples), label_vec] = 1
    return one_hot_code


def impute_norm_pipeline_fit(df, prots):
    """
    Fit a pipeline for impute and normalize data. 

    Args:
        df (pd.DataFrame): Dataframe containing protein data
        prots (list): List of protein names

    Returns:
        list: List containing imputer and scaler objects
    """
    df_imputed, imputer = knn_impute(df, prots, isTrain=True)
    _, scaler = gauss_norm(df_imputed, prots, None, isTrain=True)

    return [imputer, scaler]
    

def impute_norm_pipeline_infer(df, 
                               prots, 
                               pipeline):
    """
    Infer using fitted pipeline for impute and normalize data. 

    Args:
        df (pd.DataFrame): Dataframe containing protein data
        prots (list): List of protein names
        pipeline (list): List containing imputer and scaler objects

    Returns:
        pd.DataFrame: Normalized and imputed dataframe
    """
    df_imputed, _ = knn_impute(df, prots, pipeline[0], isTrain=False)
    df_imputed_normed, _ = gauss_norm(
        df_imputed, prots, pipeline[1], isTrain=False)

    return df_imputed_normed


def train_val_test_proc(ref_df, train_df, val_df, test_df, prots):
    """
    Processing train/val/test data with:
        avgProtNorm, 
        KNN imputation, 
        GaussRank normalization.

    Args:
        ref_df (pd.DataFrame): Reference dataframe for fitting
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe
        test_df (pd.DataFrame): Test dataframe
        prots (list): List of protein names
    Returns:
        tuple: Processed training, validation, and test dataframes
    """
    # 1. normalize by avgProt
    ref_df = avgProt_norm(ref_df, prots)
    train_df = avgProt_norm(train_df, prots)
    val_df = avgProt_norm(val_df, prots)
    test_df = avgProt_norm(test_df, prots)
    # 2. fit impute and norm pipeline
    pipeline = impute_norm_pipeline_fit(ref_df, prots)
    # 3. infer
    train_df_proc = impute_norm_pipeline_infer(train_df, prots, pipeline)
    val_df_proc = impute_norm_pipeline_infer(val_df, prots, pipeline)
    test_df_proc = impute_norm_pipeline_infer(test_df, prots, pipeline)

    return train_df_proc, val_df_proc, test_df_proc



def gen_input_pkl(train_df_proc, 
                  val_df_proc, 
                  test_df_proc,
                  prots, 
                  conti_target_list, 
                  categ_target_list):
    """
    Generate input data in format of pkl.

    Args:
        train_df_proc (pd.DataFrame): Processed training dataframe
        val_df_proc (pd.DataFrame): Processed validation dataframe
        test_df_proc (pd.DataFrame): Processed test dataframe
        prots (list): List of protein names
        conti_target_list (list): List of continuous target variable names
        categ_target_list (list): List of categorical target variable names

    Returns:
        tuple: Processed training, validation, and test dataframes
    """
    # construct input dict
    train_pkl = construct_input_dict(train_df_proc[prots].values, isTrain=True)
    val_pkl = construct_input_dict(val_df_proc[prots].values, isTrain=False)
    test_pkl = construct_input_dict(test_df_proc[prots].values, isTrain=False)
    # add continuous targets
    for target_var in conti_target_list:
        train_pkl, mean, std = append_continuous_target(
            train_pkl, 
            train_df_proc, 
            target_var, isTrain=True)
        val_pkl, _, _ = append_continuous_target(
            val_pkl, 
            val_df_proc, target_var,
            mean, std, isTrain=False)
        test_pkl, _, _ = append_continuous_target(
            test_pkl, 
            test_df_proc, target_var, 
            mean, std, isTrain=False)
    # add categorical targets 
    for target_var in categ_target_list:
        train_pkl = append_categorical_target(train_pkl, 
                                              train_df_proc, 
                                              target_var)
        val_pkl = append_categorical_target(val_pkl, 
                                            val_df_proc, 
                                            target_var)
        test_pkl = append_categorical_target(test_pkl, 
                                             test_df_proc, 
                                             target_var)
    
    return train_pkl, val_pkl, test_pkl


def embeddings_PCA(train_z,
                   val_z,
                   test_z,
                   n_components=5):
    """
    PCA on embeddings of ProtAIDe-Dx.

    Args:
        train_z (np.ndarray): Training embeddings
        val_z (np.ndarray): Validation embeddings
        test_z (np.ndarray): Test embeddings
        n_components (int, optional): Number of PCA components. Defaults to 5.

    Returns:
        tuple: PCA-transformed training, validation, and test embeddings
    """

    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    train_z_scaled = scaler.fit_transform(train_z)
    val_z_scaled = scaler.transform(val_z)
    test_z_scaled = scaler.transform(test_z)

    train_z_pca = pca.fit_transform(train_z_scaled)
    val_z_pca = pca.transform(val_z_scaled)
    test_z_pca = pca.transform(test_z_scaled)

    return train_z_pca, val_z_pca, test_z_pca


def feature_normalizing_with_embeddingPCA(train_df,
                                          val_df,
                                          test_df,
                                          features,
                                          features2norm,
                                          z_cols,
                                          model,
                                          n_components=5):
    """
    Feature normalizing with embedding PCA and imbalanced data handling.

    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe
        test_df (pd.DataFrame): Test dataframe
        features (list): List of feature names
        features2norm (list): List of features to normalize
        z_cols (list): List of embedding column names
        model (str): Model identifier
        n_components (int, optional): Number of PCA components. Defaults to 5.

    Returns:
        tuple: Normalized and PCA-transformed 
            training, validation, and test features and labels
    """
    # z normalization 
    mean, std = train_df[features2norm].mean(), \
        train_df[features2norm].std()
    train_df[features2norm] = (train_df[features2norm] - mean) / std
    val_df[features2norm] = (val_df[features2norm] - mean) / std
    test_df[features2norm] = (test_df[features2norm] - mean) / std

    X_train, X_val, X_test = train_df[features].values, \
        val_df[features].values, \
            test_df[features].values
    y_train, y_val, y_test = train_df['Label'].values, \
        val_df['Label'].values, test_df['Label'].values

    if model in ['M1', 'M3']:
        # PCA on embeddings 
        train_z, val_z, test_z = X_train[:, :len(z_cols)], \
            X_val[:, :len(z_cols)], X_test[:, :len(z_cols)]
        train_z_pca, val_z_pca, test_z_pca = embeddings_PCA(
            train_z, val_z, test_z, n_components=n_components)
        
        X_train = np.concatenate(
            (train_z_pca, X_train[:, len(z_cols):]), axis=1)
        X_val = np.concatenate(
            (val_z_pca, X_val[:, len(z_cols):]), axis=1)
        X_test = np.concatenate(
            (test_z_pca, X_test[:, len(z_cols):]), axis=1)
    
    # imbalanced data handling 
    if X_train.shape[1] > 10:
        smote = SMOTE(sampling_strategy={1: 200, 2: 160, 3:160},
                      random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test
