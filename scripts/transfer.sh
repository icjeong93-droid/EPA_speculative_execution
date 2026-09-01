#!/usr/bin/env bash
# Move the offline bundle AND the dataset to the HPC in one go.
#
#   bash scripts/transfer.sh user@hpc:/scratch/$USER/EPA_incoming
#
# The dataset is sent straight from data/ rather than being copied into the
# bundle first, so it is never duplicated on the local disk.
set -euo pipefail
DEST="${1:?usage: transfer.sh user@host:/remote/path}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RS="rsync -avhP --partial --info=progress2"
echo "==> bundle (wheels + hf_cache + repo + scripts/configs)"
$RS "$ROOT/offline_bundle/" "$DEST/bundle/"
echo "==> dataset (~28 GB, resumable)"
$RS "$ROOT/data/SpokenWOZ/" "$DEST/data/SpokenWOZ/"
cat <<EOF

==> transferred. on the HPC:
  cd $DEST/bundle && bash setup_offline.sh /scratch/\$USER/EPA
  mkdir -p /scratch/\$USER/EPA/data && mv $DEST/data/SpokenWOZ /scratch/\$USER/EPA/data/
EOF
