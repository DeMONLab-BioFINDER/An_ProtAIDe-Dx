#!/bin/sh
# Written by Lijun An and DeMON Lab under MIT license https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE

set -euo pipefail

# Get root dir
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_DIR=$ROOT_DIR'/data/replica'
RESULTS_DIR=$ROOT_DIR'/results/replica'
cd $ROOT_DIR

# data folders
mkdir -p $DATA_DIR'/raw'
mkdir -p $DATA_DIR'/BF2/splits'
mkdir -p $DATA_DIR'/BF2/deep_input'

# results folders
mkdir -p $RESULTS_DIR'/Fig4_BF2/ProtAIDeDx'
mkdir -p $RESULTS_DIR'/Fig4_BF2/kshot'
mkdir -p $RESULTS_DIR'/Fig4_BF2/EmbBiomarkerCorr'
mkdir -p $RESULTS_DIR'/Fig5_BF2/MMSE_Trajectory'
mkdir -p $RESULTS_DIR'/Fig5_BF2/DifferentialDiagnosis'

