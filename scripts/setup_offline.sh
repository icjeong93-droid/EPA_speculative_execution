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
# torchcodec goes in with torch: torchaudio 2.9 sends every load/save through it and
# raises ImportError without it, so omitting it yields an install that passes every
# import check and then dies on the first wav of preprocessing.
"$PY" -m pip install --no-index --find-links "$BUNDLE/wheels" \
  torch==2.9.1 torchaudio==2.9.1 torchcodec==0.16.0

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

# The shim lives in our scripts/, which the bundle carries under epa/. The old code
# looked for it at $PWD/scripts/sitecustomize.py -- a path that exists in the git
# checkout but never in the bundle -- so the fallback silently did nothing and the
# installer still exited 0. Hand both candidates to python explicitly.
export EPA_SHIM_SRC="$BUNDLE/epa/scripts/sitecustomize.py"
export EPA_SHIM_SRC_ALT="$TARGET/scripts/sitecustomize.py"

"$PY" - <<'EOF'
import os, sys, torch, torchaudio
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

# --- audio I/O: the check that actually matters ---------------------------
# Every wav in preprocessing goes through torchaudio.load/save. torchaudio 2.9
# dispatches both to TorchCodec unconditionally, and TorchCodec additionally needs
# FFmpeg shared libs at runtime. Importing the module is not proof; do a real
# round-trip, and install the soundfile shim if it fails.
import pathlib, shutil, site, tempfile


def roundtrip():
    """Write a wav and read it back the way the pipeline does. Returns None on success."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "probe.wav")
        # 1-D input on purpose: data_processing.handle_channels saves single channels
        # as multichannel_audio[ch_idx], which is 1-D.
        wav = torch.sin(torch.arange(8000, dtype=torch.float32) * 0.05) * 0.5
        try:
            torchaudio.save(p, wav, 8000)
            # frame_offset/num_frames are what load_audio_segment uses
            back, sr = torchaudio.load(p, frame_offset=100, num_frames=4000)
        except Exception as e:                       # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        if sr != 8000 or back.shape[-1] != 4000:
            return f"bad round-trip: sr={sr} shape={tuple(back.shape)}"
    return None


err = roundtrip()
if err is None:
    print("audio i/o  OK (torchaudio.load/save round-trip, TorchCodec native)")
else:
    print("audio i/o  FAILED ->", err)
    print("           installing the soundfile shim (scripts/sitecustomize.py)")
    src = None
    for cand in (os.environ.get("EPA_SHIM_SRC"), os.environ.get("EPA_SHIM_SRC_ALT")):
        if cand and pathlib.Path(cand).is_file():
            src = pathlib.Path(cand)
            break
    if src is None:
        raise SystemExit(
            "!! sitecustomize.py not found in the bundle -- audio I/O is broken and\n"
            "   cannot be repaired offline. Rebuild the bundle with\n"
            "   scripts/build_offline_bundle.py and re-transfer.")
    dst = pathlib.Path(site.getsitepackages()[0]) / "sitecustomize.py"
    shutil.copy(src, dst)
    print("           installed", dst)
    # sitecustomize only runs at interpreter start, so re-check in a fresh process
    import subprocess
    r = subprocess.run(
        [sys.executable, "-c",
         "import torch,torchaudio,tempfile,os\n"
         "d=tempfile.mkdtemp(); p=os.path.join(d,'p.wav')\n"
         "torchaudio.save(p, torch.zeros(8000), 8000)\n"
         "b,s=torchaudio.load(p, frame_offset=100, num_frames=4000)\n"
         "assert s==8000 and b.shape[-1]==4000, (s, b.shape)\n"
         "print('audio i/o  OK (soundfile shim active)')"],
        capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        raise SystemExit("!! audio I/O still broken after installing the shim -- stop here.")
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
