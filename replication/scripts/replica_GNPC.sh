#!/bin/sh
# Written by Lijun An and DeMON Lab under MIT license https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE

set -euo pipefail

# Get root dir
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd $ROOT_DIR

# replicate GNPC main results 
bash replication/scripts/step0_prepare_GNPC.sh
bash replication/scripts/step1_replica_fig1_GNPC.sh
bash replication/scripts/step2_replica_fig2_GNPC.sh
bash replication/scripts/step3_replica_fig3_GNPC.sh
bash replication/scripts/step4_replica_fig4_GNPC.sh
bash replication/scripts/step5_replica_fig5_GNPC.sh