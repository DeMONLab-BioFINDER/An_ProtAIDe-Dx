#!/bin/sh
# Written by Lijun An and DeMON Lab under MIT license https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE

set -euo pipefail

## Environment variables setup
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd $ROOT_DIR
DATA_DIR=$ROOT_DIR"/data/replica"
CKPT_DIR=$ROOT_DIR'/checkpoints/replica'
RESULTS_DIR=$ROOT_DIR'/results/replica/Fig5_GNPC'
ProtAIDeDx_prediction_dir=$ROOT_DIR'/results/replica/Fig1/ProtAIDeDx'
multiVisit_data_path=$ROOT_DIR'/data/replica/raw/GNPC_Soma7k_MultiVisit.csv'
ref_results_path=$ROOT_DIR'/replication/refer_results/Fig5_GNPC_ref.pkl'


python -m src.posthoc.MMSE_LME_GNPC \
    --ProtAIDeDx_prediction_dir $ProtAIDeDx_prediction_dir \
    --multiVisit_data_path $multiVisit_data_path \
    --output_dir $RESULTS_DIR \
    --ref_results_path $ref_results_path


