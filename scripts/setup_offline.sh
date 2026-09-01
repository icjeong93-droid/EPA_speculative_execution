#!/usr/bin/env bash
# Install EPA on an OFFLINE Linux HPC from the bundle built by build_offline_bundle.py.
# No internet, no DNS, no uv required -- stdlib venv + pip --no-index only.
#
#   bash setup_offline.sh /scratch/$USER/EPA
#
# Run this from inside the unpacked bundle directory.
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?usage: setup_offline.sh /path/to/EPA_workdir}"
PYBIN="${PYTHON:-python3.12}"

command -v "$PYBIN" >/dev/null || { echo "!! $PYBIN not found. module load python/3.12, or set PYTHON=..."; exit 1; }
echo "python : $($PYBIN --version)  ($(command -v $PYBIN))"
echo "bundle : $BUNDLE"
echo "target : $TARGET"

mkdir -p "$TARGET"
# our scripts/configs + the upstream repo (patch already applied; the raw diff is
# kept alongside as repo_local_changes.patch so the change stays auditable)
cp -r "$BUNDLE/epa/." "$TARGET/"
mkdir -p "$TARGET/EndpointAnticipation"
cp -r "$BUNDLE/repo/." "$TARGET/EndpointAnticipation/"
cp "$BUNDLE/repo_local_changes.patch" "$TARGET/" 2>/dev/null || true
mkdir -p "$TARGET"/{data,dump,checkpoints,logs}

# --- venv from local wheels only -----------------------------------------
echo "==> venv"
"$PYBIN" -m venv "$TARGET/.venv"
PY="$TARGET/.venv/bin/python"
"$PY" -m pip install --no-index --find-links "$BUNDLE/wheels" --upgrade pip setuptools wheel

echo "==> torch first (moshi pins torch<2.10; installing it after moshi would let"
echo "    moshi's resolver replace the cu128 build with a CPU one)"
"$PY" -m pip install --no-index --find-links "$BUNDLE/wheels" torch==2.9.1 torchaudio==2.9.1

echo "==> remaining deps"
"$PY" -m pip install --no-index --find-links "$BUNDLE/wheels" \
  numpy matplotlib PyYAML huggingface_hub tqdm moshi librosa scikit-learn \
  easydict coloredlogs soundfile silero-vad transformers wandb

# --- offline environment --------------------------------------------------
# HF_HUB_CACHE points at the snapshots we shipped; the OFFLINE flags stop
# transformers/hub from even attempting a network call (which on a DNS-blocked
# node otherwise stalls until timeout rather than failing fast).
HFC="$TARGET/hf_cache"
cp -r "$BUNDLE/hf_cache/." "$HFC/" 2>/dev/null || { mkdir -p "$HFC"; cp -r "$BUNDLE/hf_cache/." "$HFC/"; }

cat > "$TARGET/env.sh" <<EOF
# source this before any run on the HPC
export HF_HUB_CACHE="$HFC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
# dataloader workers per run; 4 concurrent single-GPU runs x this = total procs
export EPA_NUM_WORKERS=\${EPA_NUM_WORKERS:-8}
export VENV="$TARGET/.venv"
EOF
echo "wrote $TARGET/env.sh"

# --- verify ---------------------------------------------------------------
echo "==> verifying (offline)"
set +u; source "$TARGET/env.sh"; set -u
"$PY" - <<'EOF'
import os, torch, torchaudio
from moshi.modules.transformer import StreamingTransformer
from transformers import MimiModel, AutoFeatureExtractor
from silero_vad import load_silero_vad
import wandb  # noqa

print("torch      ", torch.__version__)
print("torchaudio ", torchaudio.__version__)
assert torch.__version__.startswith("2.9"), "moshi needs torch<2.10 -- wrong torch installed"
assert "+cu" in torch.__version__, "CPU-only torch installed -- cu128 wheel was replaced"
print("cuda       ", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if torch.cuda.is_available():
    print("arch list  ", torch.cuda.get_arch_list())
    assert "sm_90" in torch.cuda.get_arch_list(), "H100 (sm_90) not in this build"

# the two things that would otherwise hit the network at runtime
load_silero_vad(); print("silero     OK (bundled in package, no network)")
MimiModel.from_pretrained("kyutai/mimi"); AutoFeatureExtractor.from_pretrained("kyutai/mimi")
print("mimi       OK (loaded from HF_HUB_CACHE with HF_HUB_OFFLINE=1)")

try:
    import torchcodec._core.ops  # noqa
    print("torchcodec OK -> torchaudio.load native")
except Exception as e:
    print("torchcodec unusable ->", type(e).__name__, "-- installing soundfile shim")
    import site, shutil, pathlib
    sp = pathlib.Path(site.getsitepackages()[0])
    src = pathlib.Path(os.environ["PWD"]) / "scripts" / "sitecustomize.py"
    if src.exists():
        shutil.copy(src, sp / "sitecustomize.py"); print("  installed", sp / "sitecustomize.py")
EOF

cat <<EOF

==> done.

next, on the HPC:
  source $TARGET/env.sh
  # dataset must already be at $TARGET/data/SpokenWOZ  (transferred separately)
  \$VENV/bin/python scripts/make_configs.py --root $TARGET --num-workers 32
  # 1) preprocessing is CPU-bound (1-2h) -- run it as a CPU job first so the
  #    H100s are not idling through resampling:
  #    EPA_NUM_WORKERS=0 \$VENV/bin/python -c "..."   (see README)
  # 2) then: sbatch scripts/train.sbatch table1
EOF
