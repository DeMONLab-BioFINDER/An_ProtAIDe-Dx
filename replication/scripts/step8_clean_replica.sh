#!/bin/sh
# Written by Lijun An and DeMON Lab under MIT license https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE

set -euo pipefail

## Environment variables setup
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR=$ROOT_DIR'/results'
DATA_DIR=$ROOT_DIR'/data/replica'

cd $ROOT_DIR

# clean replication results and data
rm -rf $RESULTS_DIR
rm -rf $DATA_DIR

# remove hidden files 
find "$ROOT_DIR" -type f \( \
  -name "*.pyc" -o \
  -name ".vscode-upload.json" -o \
  -name ".DS_Store" \
\) -print -delete