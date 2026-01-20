#!/bin/sh
# Written by Lijun An and DeMON Lab under MIT license https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE

set -euo pipefail

## Environment variables setup
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd $ROOT_DIR
DATA_DIR=$ROOT_DIR"/data/replica"
CKPT_DIR=$ROOT_DIR'/checkpoints/ProtAIDeDx/'
REF_DIR=$ROOT_DIR"/replication/ref_results"
RESULTS_DIR=$ROOT_DIR'/results/replica/Fig4_GNPC'
hyper_params_path=$ROOT_DIR'/checkpoints/ProtAIDeDx/ProtAIDeDx_HyperParams.csv' 
probaThresholds_path=$ROOT_DIR'/checkpoints/ProtAIDeDx/ProtAIDeDx_ProbaThresholds.csv' 
targets2pred_path=$ROOT_DIR'/data/Targets2Pred_LeaveOneSiteOut.csv' 

EXP="LOSO"
splits_dir_exp=$DATA_DIR'/'$EXP'/splits'
input_dir_exp=$DATA_DIR'/'$EXP'/deep_input'
results_dir_exp=$RESULTS_DIR'/ProtAIDeDx'
input_features_dir_exp=$ROOT_DIR"/checkpoints/data_proc/"$EXP

# replicate ProtAIDeDx's results for leave-one-site-out
sites=("A" "C" "D" "E" "F" "G" "I" "J" "L" "M" "N" "P" "Q" "R")
for site in "${sites[@]}"; do
    site_dir='site_'$site 
    input_features=$input_features_dir_exp'/'$site_dir'/input_aptamers.txt'
    python -m src.ProtAIDeDx.main \
        --input_dir $input_dir_exp'/'$site_dir \
        --checkpoint_dir $CKPT_DIR \
        --results_dir $results_dir_exp'/'$site_dir \
        --splits_dir $splits_dir_exp'/'$site_dir \
        --hyperParam_path $hyper_params_path \
        --probaThresholds_path $probaThresholds_path \
        --features_path $input_features \
        --suffix $EXP \
        --split $site_dir
done

# print out results summary as reported in paper
python -m src.utils.print_ProtAIDeDx_results \
    --results_dir $results_dir_exp \
    --targets2pred_path $targets2pred_path \
    --suffix $EXP

# check whether it is successfully replicated
python -m src.utils.check_ProtAIDeDx_replication \
    --results_dir $results_dir_exp \
    --ref_dir $REF_DIR \
    --suffix $EXP