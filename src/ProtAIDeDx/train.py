#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import os 
import copy
import torch
import argparse
import torch.optim as optim
from src.ProtAIDeDx.misc.ProtAIDeDx_model import build_ProtAIDeDx
from src.ProtAIDeDx.misc.nn_loss import bce_loss, rank_loss
from src.ProtAIDeDx.misc.nn_init import set_random_seed
from src.ProtAIDeDx.misc.nn_data import get_train_dataloader, load_val_data
from src.ProtAIDeDx.misc.nn_logger import Logger
from src.ProtAIDeDx.misc.nn_helper import load_hyperParams
from src.ProtAIDeDx.misc.nn_helper import eval_on_validation, eval_on_test
from src.utils.io import df2csv, list2txt


def train_ProtAIDe_args_parser():
    """
    Parameters for training ProtAIDeDx models.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(prog='TrainProtAIDeDxArgs')
    # general parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_dir', type=str, default='/')
    parser.add_argument('--checkpoint_dir', type=str, default='/')
    parser.add_argument('--hyperParam_path', type=str, default='/')
    parser.add_argument('--suffix', type=str, default='CV')
    parser.add_argument('--split', type=str, default='fold_0')

    # data
    parser.add_argument('--train_data', type=str, default='train.pkl')
    parser.add_argument('--val_data', type=str, default='val.pkl')
    parser.add_argument('--test_data', type=str, default='test.pkl')
    # whether print training info, save trained model, log training info
    parser.add_argument('--HyperSearchOnly', action='store_true', 
                        default=False)
    parser.add_argument('--loadHP', action='store_true', default=False)
    # training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    # model parameters
    parser.add_argument('--in_dim', type=int, default=-1)
    parser.add_argument('--z_dim', type=int, default=-1)
    parser.add_argument('--target_order', type=list,
                        default=['CU', 'AD', 'PD', 'FTD', 'ALS', 'StrokeTIA'])
    # hyper parameters
    parser.add_argument('--input_head_dim', type=int, default=512)
    parser.add_argument('--encoder_dims', type=list, default=[16])
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--drop_out', type=float, default=0.2)
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument('--lambda_rank', type=float, default=4)
    parser.add_argument('--label_smooth', type=list, default=[
        0.1, 0.1, 0.1, 0.1, 0, 0.1])

    train_args, _ = parser.parse_known_args()
    return train_args


def train_1epoch(args,
                 train_loader, 
                 InputHeadLayer,
                 ProtAIDeModel,
                 optimizer,
                 device):
    """
    Train ProtAIDeDx for 1 epoch.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
        train_loader (torch.utils.data.DataLoader): 
            DataLoader for training data
        InputHeadLayer (torch.nn.Module): Input head layer of the model
        ProtAIDeModel (torch.nn.Module): ProtAIDeDx model
        optimizer (torch.optim.Optimizer): Optimizer for training
        device (torch.device): Device to run the training on

    Returns:
        tuple: Updated InputHeadLayer, 
            ProtAIDeModel, optimizer, and training loss
    """
    train_loss = 0
    nb_batches = len(train_loader)
    for _, (batch_x, batch_target, batch_mask) in enumerate(train_loader):
        batch_x = batch_x.float().to(device)
        batch_mask = batch_mask.bool().to(device)
        batch_target = batch_target.to(device)
        input_head = InputHeadLayer(batch_x)
        y_pred, _ = ProtAIDeModel(input_head)
        # compute loss
        loss_cat = bce_loss(args, batch_target.float(), y_pred, batch_mask)
        loss_rl = rank_loss(batch_target.long(),
                            y_pred, 
                            batch_mask)
        loss = loss_cat + args.lambda_rank * loss_rl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy().item() / nb_batches
    
    return InputHeadLayer, ProtAIDeModel, optimizer, train_loss


def train(args):
    """
    Main training function.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    # set random seed
    set_random_seed(args.seed)
    # set device
    device = torch.device(args.device)
    # check whether to load searched hyperparameters
    if args.loadHP:
        args = load_hyperParams(args.hyperParam_path,
                                args)
    args.z_dim = int(args.encoder_dims[-1])
    # get train dataloader
    train_dataloader, _ = get_train_dataloader(
        train_pkl_path=os.path.join(args.input_dir, args.train_data),
        target_order=args.target_order,
        batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    # load validation data 
    val_input, val_label, val_mask = load_val_data(
        val_pkl_path=os.path.join(args.input_dir, args.val_data),
        target_order=args.target_order,
        device=device)
    test_input, test_label, test_mask = load_val_data(
        val_pkl_path=os.path.join(args.input_dir, args.test_data),
        target_order=args.target_order,
        device=device)
    # build model, optimizer, and learning rate scheduler 
    InputHeadLayer, ProAIDeModel = build_ProtAIDeDx(args.in_dim,
                                                    args.input_head_dim,
                                                    args.encoder_dims,
                                                    args.drop_out)
    InputHeadLayer, ProAIDeModel = \
        InputHeadLayer.to(device), ProAIDeModel.to(device)
    optimizer = getattr(optim, args.optimizer_name)(
        params=list(InputHeadLayer.parameters()) + list(
            ProAIDeModel.parameters()), 
        lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2)
    # training logger
    if not args.HyperSearchOnly:
        # initialize a training logger 
        train_logger = Logger(metric_names=args.target_order)
        # begin training 
        best_models = [None, None]
        best_val_met_df = None
        best_test_met_df = None
        best_thresholds = []
    best_val_metric = 0
    for epoch in range(args.epochs):
        args.step = epoch
        # train 1 epoch
        InputHeadLayer.train()
        ProAIDeModel.train()
        InputHeadLayer, ProAIDeModel, optimizer, train_loss = \
            train_1epoch(args, train_dataloader, InputHeadLayer, 
                         ProAIDeModel, optimizer, device)
        scheduler.step()
        # validation
        InputHeadLayer.eval()
        ProAIDeModel.eval()
        if args.HyperSearchOnly:
            val_metric = eval_on_validation(args, val_input, val_label, 
                                            val_mask, 
                                            InputHeadLayer, ProAIDeModel,
                                            args.HyperSearchOnly)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
        else:
            val_metric, val_loss, val_bac_scores, \
                val_met_df, opt_thresholds = \
                eval_on_validation(args, val_input, val_label, val_mask, 
                                   InputHeadLayer, ProAIDeModel,
                                   args.HyperSearchOnly)
            # update logger
            print('Epoch:', epoch)
            train_logger.log(train_loss, val_loss, val_bac_scores)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_models = [copy.deepcopy(InputHeadLayer),
                               copy.deepcopy(ProAIDeModel)]
                best_val_met_df = val_met_df
                test_met_df = \
                    eval_on_test(args, test_input, test_label, test_mask, 
                                InputHeadLayer, ProAIDeModel,
                                opt_thresholds)
                best_test_met_df = test_met_df
                best_thresholds = opt_thresholds
                print(val_met_df)
                print(test_met_df)
                print('Optimal threshold:', opt_thresholds)
                print('=======================')   
    # end training
    if args.HyperSearchOnly:
        return best_val_metric
    else:
        # save models
        torch.save(best_models[0].state_dict(), 
                   os.path.join(
                       args.checkpoint_dir, 
                       'InputHead_' + args.suffix + '.pt'))
        torch.save(best_models[1].state_dict(), 
                   os.path.join(
                       args.checkpoint_dir, 
                       'ProtAIDe_' + args.suffix + '.pt'))
        list2txt(best_thresholds, os.path.join(
            args.checkpoint_dir, 'thresholds.txt'))
        # save logger
        train_logger.save(os.path.join(args.checkpoint_dir, 'log.txt'))
        train_logger.plot(os.path.join(args.checkpoint_dir, 'curve.png'),
                          'Balanced Accuracy')
        # save evaluation results
        df2csv(best_val_met_df, os.path.join(
            args.checkpoint_dir, 'val_metrics.csv'))
        df2csv(best_test_met_df, os.path.join(
            args.checkpoint_dir, 'test_metrics.csv'))


        
if __name__ == '__main__':
    train(train_ProtAIDe_args_parser())