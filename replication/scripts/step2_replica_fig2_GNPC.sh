#!/bin/sh
# Written by Lijun An and DeMON Lab under MIT license https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE

set -euo pipefail

## Environment variables setup
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd $ROOT_DIR
PROTAIDEDX_RESULTS_DIR=$ROOT_DIR'/results/replica/Fig1/ProtAIDeDx'
OUPUT_DIR=$ROOT_DIR'/results/replica/Fig2'
SPLITS_DIR_CV=$ROOT_DIR'/data/replica/CV/splits'
REF_DIR=$ROOT_DIR'/replication/ref_results'
probaThresholds_path=$ROOT_DIR'/checkpoints/ProtAIDeDx/ProtAIDeDx_ProbaThresholds.csv' 



# replicate tSNE's results
python -m src.posthoc.tSNE \
    --ProtAIDeDx_results_dir $PROTAIDEDX_RESULTS_DIR \
    --splits_dir $SPLITS_DIR_CV \
    --output_dir $OUPUT_DIR \
    --ref_dir $REF_DIR \
    --probaThresholds_path $probaThresholds_path