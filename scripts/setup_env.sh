#!/usr/bin/env bash
# Reproduce the EPA environment. Verified on Windows (RTX 5080, sm_120) and
# intended for Linux HPC (H100, sm_90). Both are covered by the cu128 wheels.
#
#   bash scripts/setup_env.sh [/path/to/venv]
set -euo pipefail

VENV="${1:-.venv}"
echo "==> creating venv at $VENV"
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY="$VENV/Scripts/python.exe"

# --- torch first, pinned ------------------------------------------------
# moshi requires torch<2.10 (it ships the StreamingTransformer this model uses),
# while RTX 50xx (sm_120) needs a cu128 build. torch 2.9.1+cu128 is the only
# point that satisfies both, and it also covers H100 (sm_90).
# Installing torch AFTER moshi lets moshi's pin silently downgrade you to a
# CPU-only build -- install it first, and re-check at the end.
echo "==> torch 2.9.1+cu128"
uv pip install --python "$PY" torch==2.9.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

# --- everything else ----------------------------------------------------
# requirements.txt upstream is incomplete: silero-vad (data prep VAD),
# transformers (Mimi loading) and wandb (imported unconditionally by
# src/utils/wandb_logger.py even when use_wandb: false) are all missing.
echo "==> remaining deps"
uv pip install --python "$PY" \
  numpy matplotlib PyYAML huggingface_hub tqdm moshi librosa scikit-learn \
  easydict coloredlogs soundfile silero-vad transformers wandb torchcodec

echo "==> verifying"
"$PY" - <<'EOF'
import torch, torchaudio
from moshi.modules.transformer import StreamingTransformer
from transformers import MimiModel
import silero_vad, wandb
print("torch     ", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("cuda      ", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("arch list ", torch.cuda.get_arch_list() if torch.cuda.is_available() else "")
assert torch.__version__.startswith("2.9"), "moshi needs torch<2.10 -- torch got overwritten"
try:
    import torchcodec._core.ops  # noqa
    print("torchcodec OK (torchaudio.load native)")
except Exception as e:
    print("torchcodec unusable ->", type(e).__name__)
    print("  install the soundfile shim: cp scripts/sitecustomize.py $VENV/lib/*/site-packages/")
EOF
echo "==> done"
