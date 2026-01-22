#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os 
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from src.utils.io import df2csv, load_pkl
from src.utils.metrics import clf_metrics, clf_multi_target_metric
from src.ProtAIDeDx.misc.nn_loss import bce_loss, rank_loss


def load_hyperParams(hyperParams_path, args, modelOnly=False):
    """
    Load hyper parameters. 

    Args:
        hyperParams_path (str): Path to the hyperparameters CSV file
        args (argparse.Namespace): 
            Arguments namespace containing experiment details
        modelOnly (bool, optional): 
            Whether to load only model parameters. Defaults to False.

    Returns:
        argparse.Namespace or tuple: 
            Updated args namespace or model parameters tuple
    """
    HP_df_all = pd.read_csv(hyperParams_path)
    HP_df = HP_df_all[
        (HP_df_all['Experiment'] == args.suffix) & (
            HP_df_all['Split'] == args.split)]

    nb_layers = int(HP_df['nb_layers'].values[0])
    encoder_dims = []
    for i in range(nb_layers):
        encoder_dims.append(int(HP_df['n_units_l' + str(i)].values[0]))
    
    input_head_dim = int(HP_df['input_head_dim'].values[0])
    
    if modelOnly: 
        return input_head_dim, encoder_dims, float(HP_df['drop_out'].values[0]) 
    else:
        args.input_head_dim = input_head_dim
        args.optimizer_name = HP_df['optimizer_name'].values[0]
        args.lr = float(HP_df['lr'].values[0])
        args.drop_out = float(HP_df['drop_out'].values[0])
        args.lambda_rank = float(HP_df['lambda_rank'].values[0])
        args.encoder_dims = encoder_dims
        args.label_smooth = [
            float(HP_df['label_smooth_control'].values[0]), 
            float(HP_df['label_smooth_ad'].values[0]),
            float(HP_df['label_smooth_pd'].values[0]),
            float(HP_df['label_smooth_ftd'].values[0]),
            float(HP_df['label_smooth_als'].values[0]),
            float(HP_df['label_smooth_stroketia'].values[0])
        ]
        return args


def load_ProbaThresholds(ProbaThresholds_path,
                         experiment,
                         split):
    """
    Load probability thresholds.

    Args:
        ProbaThresholds_path (str): 
            Path to the probability thresholds CSV file
        experiment (str): Experiment identifier
        split (str): Data split identifier
    Returns:
        list: List of probability thresholds
    """
    df = pd.read_csv(ProbaThresholds_path)
    row = df[
        (df['Experiment'] == experiment) & (df['Split'] == split)]

    return row.iloc[0, 2:].to_list()


def load_pretrained_weights(InputHead, 
                            ProAIDeModel,
                            checkpoint_dir,
                            suffix,
                            split,
                            device):
    """
    Load pretrained weights

    Args:
        InputHead (torch.nn.Module): Input head model
        ProAIDeModel (torch.nn.Module): Main Proteomics AI model
        checkpoint_dir (str): Directory containing checkpoint files
        suffix (str): Experiment suffix identifier
        split (str): Data split identifier
        device (torch.device): Device to load the models onto

    Returns:
        tuple: Tuple containing the updated InputHead and ProAIDeModel
    """
    if suffix == 'CV':
        model_dict = load_pkl(
            os.path.join(checkpoint_dir, 
                         'ProtAIDeDx_CrossValidation_10Folds.pkl'))
    else:
        model_dict = load_pkl(
            os.path.join(checkpoint_dir, 
                        'ProtAIDeDx_LeaveOneSiteOut_14Sites.pkl'))

    InputHead_state = model_dict[split]['InputeHead']
    ProAIDeModel_state = model_dict[split]['ProtAIDe']
    InputHead.load_state_dict(InputHead_state)
    ProAIDeModel.load_state_dict(ProAIDeModel_state)

    InputHead = InputHead.to(torch.device(device))
    InputHead.eval()

    ProAIDeModel = ProAIDeModel.to(torch.device(device))
    ProAIDeModel.eval()

    return InputHead, ProAIDeModel


def eval_on_validation(args,
                       val_input,
                       val_label,
                       val_mask, 
                       InputHeadLayer,
                       ProtAIDeModel,
                       HyperSearchOnly=True,
                       OneTarget=False):
    """
    Evaluate model performance on validation set.

    Args:
        args (argparse.Namespace): Argument namespace containing configuration
        val_input (torch.Tensor): Validation input data
        val_label (torch.Tensor): Validation labels
        val_mask (torch.Tensor): Validation mask
        InputHeadLayer (torch.nn.Module): Input head model
        ProtAIDeModel (torch.nn.Module): Main Proteomics AI model
        HyperSearchOnly (bool, optional): 
            Whether to perform hyperparameter search only. Defaults to True.
        OneTarget (bool, optional): 
            Whether to evaluate on a single target. Defaults to False.

    Returns:
        float or tuple: Validation score if HyperSearchOnly is True, 
        otherwise a tuple containing validation score, validation loss, 
        balanced accuracy scores, validation metrics DataFrame, 
        and optimal thresholds
    """
    # forward pass
    input_head = InputHeadLayer(val_input)
    val_pred, _ = ProtAIDeModel(input_head)
    # compute loss
    if OneTarget:
        val_loss = bce_loss(args, val_label.float(),
                            val_pred, val_mask)
    else:
        val_loss = bce_loss(args, val_label.float(),
                            val_pred, val_mask) \
            + args.lambda_rank * rank_loss(val_label.long(),
                                           val_pred, val_mask)
    val_loss = val_loss.data.cpu().numpy().item()
    # compute validation metric
    val_label_numpy = val_label.cpu().numpy()
    val_mask_numpy = val_mask.cpu().numpy()
    val_predprob_numpy = (F.sigmoid(val_pred)).detach().cpu().numpy()
    bac_scores, opt_thresholds = clf_multi_target_metric(
        val_label_numpy, 
        val_predprob_numpy,
        val_mask_numpy,
        threshold_func='get_opt_threshold_PRC',
        metric='clf_bas')
    val_score = np.nanmean(bac_scores)
    
    if HyperSearchOnly:
        # only used for search best models
        return val_score
    else:
        # compute all relevant metrics on validation set
        val_met_dict = dict()
        for i, target in enumerate(args.target_order):
            val_met_dict[target] = clf_metrics(
                val_label_numpy[:, i][val_mask_numpy[:, i]],
                val_predprob_numpy[:, i][val_mask_numpy[:, i]],
                2, opt_thresholds[i])
        val_met_df = pd.DataFrame(val_met_dict).T.reset_index()
        val_met_df = val_met_df.rename({"index": "target"})
        return val_score, val_loss, bac_scores, val_met_df, opt_thresholds


def eval_on_test(args,
                 test_input,
                 test_label,
                 test_mask, 
                 InputHeadLayer,
                 ProtAIDeModel,
                 opt_thresholds):
    """
    Evaluation on testset.

    Args:
        args (argparse.Namespace): Argument namespace containing configuration
        test_input (torch.Tensor): Test input data
        test_label (torch.Tensor): Test labels
        test_mask (torch.Tensor): Test mask
        InputHeadLayer (torch.nn.Module): Input head model
        ProtAIDeModel (torch.nn.Module): Main Proteomics AI model
        opt_thresholds (list or np.ndarray): 
            Optimal thresholds for classification
    Returns:
        pd.DataFrame: DataFrame containing test metrics for each target
    """
    # forward pass
    input_head = InputHeadLayer(test_input)
    test_pred, _ = ProtAIDeModel(input_head)
    
    test_label_numpy = test_label.cpu().numpy()
    test_predprob_numpy = (F.sigmoid(test_pred)).detach().cpu().numpy()
    test_mask_numpy = test_mask.cpu().numpy()


    test_met_dict = dict()
    for i, target in enumerate(args.target_order):
        if np.sum(test_mask_numpy[:, i]) > 0:
            test_met_dict[target] = clf_metrics(
                test_label_numpy[:, i][test_mask_numpy[:, i]],
                test_predprob_numpy[:, i][test_mask_numpy[:, i]],
                2, opt_thresholds[i])
        else:
            test_met_dict[target] = {'acc':np.nan, 
                                     'bas':np.nan, 
                                     'sensitivity':np.nan, 
                                     'specificity':np.nan,
                                     'precision': np.nan, 
                                     'f1':np.nan, 
                                     'auc':np.nan}
    test_met_df = pd.DataFrame(test_met_dict).T.reset_index()
    test_met_df = test_met_df.rename({"index": "target"})
    return test_met_df


def model_infer(model_list, input_x):
    """
    Infer with trained model. 

    Args:
        model_list (list of torch.nn.Module): 
            List containing the input head model 
            and the main Proteomics AI model
        input_x (np.ndarray or torch.Tensor): 
            Input data for inference

    Returns:
        tuple: Tuple containing predicted probabilities 
               and latent representations
    """
    input_x = torch.tensor(input_x).float()
    input_head = model_list[0](input_x)
    pred, z = model_list[1](input_head)
    # convert pred to predicted probability 
    pred = F.sigmoid(pred)
    pred, z = pred.detach().numpy(), z.detach().numpy()
    return pred, z


def infer_one_set(model_list, 
                  input_x, 
                  raw_df,
                  target_order, 
                  threshold_list, 
                  z_dim, 
                  save_path,
                  noGTLabel=False):
    """
    Making inference on one set. 

    Args:
        model_list (list of torch.nn.Module): 
            List containing the input head model 
            and the main Proteomics AI model
        input_x (np.ndarray or torch.Tensor): 
            Input data for inference
        raw_df (pd.DataFrame): Raw input data as a DataFrame
        target_order (list of str): List of target names in order
        threshold_list (list or np.ndarray): 
            List of thresholds for classification
        z_dim (int): Dimensionality of the latent representation
        save_path (str): Path to save the inference results
        noGTLabel (bool, optional): 
            Whether ground truth labels are not available. Defaults to False.
    """
    pred, z = model_infer(model_list, input_x)
    assert pred.shape[1] == len(target_order), "Wrong #targets!"
    assert z.shape[0] == len(raw_df), "Wrong #rows!"
    
    # fill z values 
    if noGTLabel:
        result_arr = np.full([z.shape[0], len(target_order) * 2 + z_dim],
                             np.nan)
        result_arr[:, len(target_order) * 2:] = z
        for i, target in enumerate(target_order):
            pred_prob = pred[:, i].reshape((-1, ))
            pred_label = (pred_prob >= threshold_list[i]) * 1.0
            result_arr[:, 2 * i] = pred_label
            result_arr[:, 2 * i + 1] = pred_prob
    else:
        result_arr = np.full([z.shape[0], len(target_order) * 3 + z_dim],
                             np.nan)
        result_arr[:, len(target_order) * 3:] = z
        # fill gt, pred, and pred probabilities
        for i, target in enumerate(target_order):
            gt = raw_df[[target]].values.reshape((-1, ))
            pred_prob = pred[:, i].reshape((-1, ))
            pred_label = (pred_prob >= threshold_list[i]) * 1.0
            result_arr[:, 3 * i] = gt
            result_arr[:, 3 * i + 1] = pred_label
            result_arr[:, 3 * i + 2] = pred_prob
    # convert to dataframe
    cols = []
    if noGTLabel:
        for target in target_order:
            cols += [target+'-PredLabel', target+'-PredProb']
    else:   
        for target in target_order:
            cols += [target + '-GT', target+'-PredLabel', target+'-PredProb']
    cols += ['Z' + str(i) for i in range(z_dim)]
    result_df = pd.DataFrame(data=result_arr, columns=cols)
    # append columns
    cols2append = ['PersonGroup_ID', 
                   'Visit', 
                   'Contributor_Code']
    for col in cols2append:
        result_df[col] = raw_df[col]
    result_df = result_df[cols2append + cols]
    # save
    df2csv(result_df, save_path)