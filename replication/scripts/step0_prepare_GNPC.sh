#!/bin/sh
# Written by Lijun An and DeMON Lab under MIT license https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE

set -euo pipefail

make_fold_dirs() {
  local base_dir="$1"
  local n_folds="${2:-10}"

  [[ -z "$base_dir" ]] && return 1
  mkdir -p "$base_dir" || return 1

  for ((i=0; i<n_folds; i++)); do
    local d="${base_dir}/fold_${i}"
    [[ -d "$d" ]] || mkdir "$d"
  done
}

make_site_dirs() {
  local base_dir="$1"
  [[ -z "$base_dir" ]] && return 1
  mkdir -p "$base_dir" || return 1

  local sites=(A C D E F G I J L M N P Q R)

  local s d
  for s in "${sites[@]}"; do
    d="${base_dir}/site_${s}"
    [[ -d "$d" ]] || mkdir "$d"
  done
}

# Get root dir
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_DIR=$ROOT_DIR'/data/replica'
RESULTS_DIR=$ROOT_DIR'/results/replica'
cd $ROOT_DIR

# data folders
mkdir -p $DATA_DIR'/raw'
make_fold_dirs $DATA_DIR'/CV/splits'
make_fold_dirs $DATA_DIR'/CV/deep_input'
make_site_dirs $DATA_DIR'/LOSO/splits'
make_site_dirs $DATA_DIR'/LOSO/deep_input'

# results folders
make_fold_dirs $RESULTS_DIR'/Fig1/ProtAIDeDx'
mkdir -p $RESULTS_DIR'/Fig2'
make_fold_dirs $RESULTS_DIR'/Fig3'
make_site_dirs $RESULTS_DIR'/Fig4_GNPC/ProtAIDeDx'
mkdir -p $RESULTS_DIR'/Fig5_GNPC'

# copy data for replication 
cp '/files/NMED-A141173/replication_data/GNPC_Soma7k_Baseline.csv' $DATA_DIR'/raw'
cp '/files/NMED-A141173/replication_data/GNPC_Soma7k_MultiVisit.csv' $DATA_DIR'/raw'

# data split and generate input files 
python -m src.preproc.data_split
python -m src.preproc.input_gen --GNPC