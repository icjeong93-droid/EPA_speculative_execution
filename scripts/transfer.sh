#!/usr/bin/env bash
# Move the offline bundle AND the dataset to the HPC in one go.
#
#   bash scripts/transfer.sh user@hpc:/scratch/$USER/EPA_incoming
#   bash scripts/transfer.sh user@hpc:/scratch/$USER/EPA_incoming \
#        --data-dest /home/sr5/SR_AISolution_ACU/database/EPA
#   bash scripts/transfer.sh user@hpc:/path --bundle-only
#   bash scripts/transfer.sh user@hpc:/path --data-only
#
# --data-dest puts SpokenWOZ/ somewhere other than <remote>/data. On this cluster
# the dataset and the dump live in the shared database area, NOT in the scratch
# workdir (README section 4) -- scratch is budgeted for ~15 GB and the dataset is 28.
# Mirrors transfer.ps1's -DataDest.
#
# The dataset is sent straight from data/ rather than being copied into the bundle
# first, so it is never duplicated on the local disk.
set -euo pipefail

DEST=""; DATA_DEST=""; DATA_ONLY=0; BUNDLE_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dest)   DATA_DEST="${2:?--data-dest needs a remote path}"; shift 2 ;;
    --data-only)   DATA_ONLY=1; shift ;;
    --bundle-only) BUNDLE_ONLY=1; shift ;;
    -h|--help)     sed -n '2,18p' "$0"; exit 0 ;;
    -*)            echo "unknown option: $1" >&2; exit 1 ;;
    *)             [ -z "$DEST" ] || { echo "unexpected argument: $1" >&2; exit 1; }
                   DEST="$1"; shift ;;
  esac
done
[ -n "$DEST" ] || { echo "usage: transfer.sh user@host:/remote/path [--data-dest PATH]" >&2; exit 1; }
case "$DEST" in *:*) ;; *) echo "!! DEST must look like user@host:/remote/path" >&2; exit 1 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${DEST%%:*}"
REMOTE_ROOT="${DEST#*:}"; REMOTE_ROOT="${REMOTE_ROOT%/}"
DATA_ROOT="${DATA_DEST:-$REMOTE_ROOT/data}"; DATA_ROOT="${DATA_ROOT%/}"

command -v rsync >/dev/null || {
  echo "!! rsync not found. On Windows use scripts/transfer.ps1 instead (scp-based)." >&2
  exit 1; }

[ -d "$ROOT/offline_bundle" ] || [ "$DATA_ONLY" = 1 ] || {
  echo "!! $ROOT/offline_bundle not found. Build it first:" >&2
  echo "   python scripts/build_offline_bundle.py --root . --out offline_bundle" >&2
  exit 1; }

RS="rsync -avhP --partial --info=progress2"

if [ "$DATA_ONLY" != 1 ]; then
  echo "==> bundle (wheels + hf_cache + repo + scripts/configs + setup_offline.sh)"
  # setup_offline.sh must land at the bundle ROOT -- that is where README section 3
  # tells the operator to run it from.
  [ -f "$ROOT/offline_bundle/setup_offline.sh" ] || {
    echo "!! offline_bundle/setup_offline.sh is missing -- rebuild the bundle." >&2
    exit 1; }
  $RS "$ROOT/offline_bundle/" "$HOST:$REMOTE_ROOT/bundle/"
fi

if [ "$BUNDLE_ONLY" != 1 ]; then
  echo "==> dataset (~28 GB, resumable) -> $DATA_ROOT/SpokenWOZ/"
  ssh "$HOST" "mkdir -p '$DATA_ROOT'"
  $RS "$ROOT/data/SpokenWOZ/" "$HOST:$DATA_ROOT/SpokenWOZ/"
  echo "==> verifying remote counts"
  ssh "$HOST" "
    printf 'train_dev wav : %s (expect 4700)\n' \"\$(ls '$DATA_ROOT/SpokenWOZ/audio_5700_train_dev'/*.wav 2>/dev/null | wc -l)\"
    printf 'test wav      : %s (expect 1000)\n' \"\$(ls '$DATA_ROOT/SpokenWOZ/audio_5700_test'/*.wav 2>/dev/null | wc -l)\"
  "
fi

cat <<EOF

==> transferred. on the HPC:

  cd $REMOTE_ROOT/bundle && bash setup_offline.sh /scratch/\$USER/EPA
  source /scratch/\$USER/EPA/env.sh

  # The dataset stays where it was sent -- do NOT move it into the workdir
  # (scratch is budgeted for the bundle + checkpoints only, README section 4).
  # Point the configs at it, and put the ~110 GB dump on the same filesystem:
  \$VENV/bin/python scripts/make_configs.py --root /scratch/\$USER/EPA --num-workers 32 \\
      --spokenwoz $DATA_ROOT/SpokenWOZ \\
      --dump $DATA_ROOT/dump

  bash scripts/preflight.sh /scratch/\$USER/EPA
EOF
