#!/bin/sh
# Written by Lijun An and DeMON Lab under MIT license https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE

set -euo pipefail

## Environment variables setup
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd $ROOT_DIR
RESULTS_DIR=$ROOT_DIR'/results/replica'
REF_DIR=$ROOT_DIR"/replication/ref_results"


# Differential Diagnosis
python -m src.posthoc.Diff_Diagnosis \
    --ProtAIDeDx_outputs_path $RESULTS_DIR'/Fig4_BF2//ProtAIDeDx/ProtAIDeDx_outputs.csv' \
    --output_dir $RESULTS_DIR'/Fig5_BF2/DifferentialDiagnosis' \
    --ref_results_path $REF_DIR'/Fig5_BF2_ref.pkl'

python -m src.posthoc.Diff_Diagnosis_plot

# Longitudinal MMSE trajectory by baseline prediction
python -m src.posthoc.MMSE_LME_BF2 \
    --ref_results_path $REF_DIR'/Fig5_BF2_ref.pkl'