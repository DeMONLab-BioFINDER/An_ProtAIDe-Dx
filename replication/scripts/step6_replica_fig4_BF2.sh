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
RESULTS_DIR=$ROOT_DIR'/results/replica/Fig4_BF2'
hyper_params_path=$ROOT_DIR'/checkpoints/ProtAIDeDx/ProtAIDeDx_HyperParams.csv' 
probaThresholds_path=$ROOT_DIR'/checkpoints/ProtAIDeDx/ProtAIDeDx_ProbaThresholds.csv' 


# generate input
cp $DATA_DIR'/raw/BF2_Soma7k_Baseline.csv' $DATA_DIR'/BF2/splits'
python -m src.preproc.input_gen --BF2

# # ProtAIDeDx 
input_features=$ROOT_DIR"/checkpoints/data_proc/LOSO/site_C/input_aptamers.txt"
python -m src.ProtAIDeDx.main \
    --input_dir $DATA_DIR'/BF2/deep_input' \
    --checkpoint_dir $CKPT_DIR \
    --results_dir $RESULTS_DIR'/ProtAIDeDx' \
    --splits_dir $DATA_DIR'/BF2/splits' \
    --test_raw 'BF2_Soma7k_Baseline.csv' \
    --hyperParam_path $hyper_params_path \
    --probaThresholds_path $probaThresholds_path \
    --features_path $input_features \
    --suffix 'LOSO' \
    --split 'site_C' \
    --new \
    --NoEval

# postprocessing ProtAIDeDx's outputs by appending biomarker & diagnosis info
python -m src.utils.append_columns \
    --input_csv_path $RESULTS_DIR'/ProtAIDeDx/test_results.csv' \
    --src_csv_path $DATA_DIR'/raw/BF2_Soma7k_Baseline.csv' \
    --output_csv_path $RESULTS_DIR'/ProtAIDeDx/ProtAIDeDx_outputs.csv'

# kshot 
python -m src.k_shot.wrapper \
    --model 'LR' \
    --suffix 'LOSO' \
    --split 'site_C' \
    --input_dir $RESULTS_DIR'/ProtAIDeDx' \
    --results_dir $RESULTS_DIR'/kshot' \
    --probaThresholds_path $probaThresholds_path \
    --ref_results_path $REF_DIR'/Fig4_BF2_ref.pkl'

python -m src.utils.print_kshot_results 

# Embedding-Biomarker Correlation Analysis
python -m src.posthoc.Embed_Biomarker_Corr \
    --ref_results_path $REF_DIR'/Fig4_BF2_ref.pkl'