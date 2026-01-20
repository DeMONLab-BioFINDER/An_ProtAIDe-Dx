#!/bin/sh
# Written by Lijun An and DeMON Lab under MIT license https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE

set -euo pipefail

## Environment variables setup
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd $ROOT_DIR
DATA_DIR=$ROOT_DIR"/data/replica"
CKPT_DIR=$ROOT_DIR'/checkpoints/ProtAIDeDx/'
RESULTS_DIR=$ROOT_DIR'/results/replica/Fig3'
hyper_params_path=$ROOT_DIR'/checkpoints/ProtAIDeDx/ProtAIDeDx_HyperParams.csv' 
aptamers_info_path=$ROOT_DIR'/data/Selected_SomaLogic7k_Aptamers.csv'
ref_results_path=$ROOT_DIR'/replication/ref_results/Fig3_GNPC_ref.pkl'

EXP="CV"
input_dir_exp=$DATA_DIR'/'$EXP'/deep_input'
input_features_dir_exp=$ROOT_DIR"/checkpoints/data_proc/"$EXP

# permFIT to get feature importance for each fold
folds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9") 
for fold in "${folds[@]}";
do
    fold_dir='fold_'$fold
    input_features=$input_features_dir_exp'/'$fold_dir'/input_aptamers.txt'
    python -m src.XAI.permFIT \
        --input_dir $input_dir_exp'/'$fold_dir \
        --checkpoint_dir $CKPT_DIR \
        --features_path $input_features \
        --hyperParam_path $hyper_params_path \
        --PermFIT_results_dir $RESULTS_DIR'/'$fold_dir \
        --suffix $EXP \
        --split $fold_dir
done

# plot protein importance
python -m src.XAI.get_important_proteins \
   --results_dir $RESULTS_DIR \
   --aptamers_info_path $aptamers_info_path \
   --ref_results_path $ref_results_path