#!/usr/bin/env python3
"""
Build a transferable bundle for an OFFLINE Linux HPC (no internet, no DNS).

Run this on a machine WITH internet. It collects everything the HPC needs:

  offline_bundle/
    wheels/          Linux cp312 wheels for every dependency (incl. torch cu128)
    hf_cache/        HuggingFace snapshots (kyutai/mimi is required for training)
    repo/            upstream EndpointAnticipation at a pinned commit + our patch
    epa/             our scripts/ and configs/
    MANIFEST.txt
    setup_offline.sh install script to run on the HPC

The dataset (~28 GB) is deliberately NOT copied into the bundle -- it is
transferred straight from data/ by transfer.sh, to avoid duplicating it on disk.

  python scripts/build_offline_bundle.py --root . --out offline_bundle

What actually needs the network at runtime (verified by reading the code):
  * training : transformers MimiModel/AutoFeatureExtractor.from_pretrained("kyutai/mimi")
  * infer.py : hf_hub_download("viks66/endpoint-anticipation") + Mimi from "kyutai/stt-1b-en_fr"
  * silero-vad: NONE -- the package ships silero_vad/data/*.jit and loads it via
    importlib.resources, so VAD preprocessing is already offline-safe.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# cp312 / linux x86_64. Several tags because not every project publishes the newest.
# manylinux_2_27 is NOT optional: the CUDA 12.8 nvidia-* wheels (cublas, cudnn,
# curand, cusolver) publish ONLY that tag. Without it pip reports the pinned
# version as nonexistent and the fallback below used to fetch the host wheel --
# which on a Windows build machine means win_amd64 wheels in a Linux bundle.
# Keep every tag <= the HPC's glibc (checked: 2.34).
PLATFORMS = ["manylinux_2_28_x86_64", "manylinux_2_27_x86_64",
             "manylinux2014_x86_64", "manylinux_2_17_x86_64"]
PY_VERSION = "3.12"
ABI = "cp312"

TORCH = ["torch==2.9.1", "torchaudio==2.9.1"]

# torch's CUDA runtime deps are all guarded by `; platform_system == "Linux"`.
# pip's --platform flag changes WHEEL TAG SELECTION ONLY -- environment markers are
# still evaluated against the machine running pip. Downloading from Windows therefore
# silently skips every one of these, producing a bundle that cannot import torch on
# the HPC. Pin them explicitly. Versions are read from the cu128 wheel's METADATA.
LINUX_ONLY = [
    "nvidia-cuda-nvrtc-cu12==12.8.93", "nvidia-cuda-runtime-cu12==12.8.90",
    "nvidia-cuda-cupti-cu12==12.8.90", "nvidia-cudnn-cu12==9.10.2.21",
    "nvidia-cublas-cu12==12.8.4.1", "nvidia-cufft-cu12==11.3.3.83",
    "nvidia-curand-cu12==10.3.9.90", "nvidia-cusolver-cu12==11.7.3.90",
    "nvidia-cusparse-cu12==12.5.8.93", "nvidia-cusparselt-cu12==0.7.1",
    "nvidia-nccl-cu12==2.27.5", "nvidia-nvshmem-cu12==3.3.20",
    "nvidia-nvtx-cu12==12.8.90", "nvidia-nvjitlink-cu12==12.8.93",
    "nvidia-cufile-cu12==1.13.1.3", "triton==3.5.1",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

PYPI = [
    "numpy", "matplotlib", "PyYAML", "huggingface_hub", "tqdm", "moshi",
    "librosa", "scikit-learn", "easydict", "coloredlogs", "soundfile",
    "silero-vad", "transformers", "wandb",
    # bootstrap so the offline install can build any sdist that slips through
    "pip", "setuptools", "wheel",
]

# repo_id -> why we need it
HF_MODELS = {
    "kyutai/mimi": "training feature extractor (REQUIRED)",
    "viks66/endpoint-anticipation": "released checkpoint, for infer.py / sanity check",
}
HF_MODELS_OPTIONAL = {
    "kyutai/stt-1b-en_fr": "infer.py's Mimi source (~GBs); only needed to run infer.py on the HPC",
}


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=False, **kw)


def dl_wheels(py, dest, pkgs, index=None):
    dest.mkdir(parents=True, exist_ok=True)
    base = [py, "-m", "pip", "download", "--dest", str(dest),
            "--python-version", PY_VERSION, "--implementation", "cp", "--abi", ABI]
    for p in PLATFORMS:
        base += ["--platform", p]
    if index:
        # torch lives on the pytorch index but its nvidia-*/numpy/jinja2 deps are
        # resolved from PyPI, so both indexes must be visible or the pass fails.
        base += ["--index-url", index, "--extra-index-url", "https://pypi.org/simple"]
    # pass 1: wheels only (what we want)
    r = run(base + ["--only-binary=:all:"] + pkgs)
    if r.returncode == 0:
        return []
    # pass 2: retry individually, allowing sdists for the stragglers
    print("  ! binary-only pass failed; retrying package by package", flush=True)
    failed = []
    for pkg in pkgs:
        r1 = run(base + ["--only-binary=:all:", pkg])
        if r1.returncode != 0:
            # Retry WITHOUT --deps but WITH the platform flags. Dropping them (as an
            # earlier version did) makes pip fall back to the host platform, which
            # silently puts win_amd64 wheels in a Linux bundle -- they install fine
            # here and are unusable on the HPC.
            r2 = run(base + ["--only-binary=:all:", "--no-deps", pkg])
            if r2.returncode != 0:
                failed.append(pkg)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="offline_bundle")
    ap.add_argument("--python", default=sys.executable, help="python with pip, used to download")
    ap.add_argument("--with-stt", action="store_true", help="also fetch kyutai/stt-1b-en_fr (large)")
    ap.add_argument("--skip-wheels", action="store_true")
    ap.add_argument("--skip-hf", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"bundle -> {out}\n")

    failed = []
    if not args.skip_wheels:
        print("[1/4] linux wheels")
        failed += dl_wheels(args.python, out / "wheels", TORCH, index=TORCH_INDEX)
        print("  -- CUDA runtime deps (skipped by env markers on non-Linux hosts)")
        failed += dl_wheels(args.python, out / "wheels", LINUX_ONLY, index=TORCH_INDEX)
        failed += dl_wheels(args.python, out / "wheels", PYPI)

        # The PyPI pass resolves moshi/silero deps and drags in the CPU torch build
        # and a newer torchaudio. Leaving them in find-links is a footgun, so drop
        # anything that is not the +cu128 build we pinned.
        wd = out / "wheels"
        for whl in list(wd.glob("torch-*.whl")) + list(wd.glob("torchaudio-*.whl")):
            if "+cu128" not in whl.name:
                print(f"  removing non-cu128 duplicate: {whl.name}")
                whl.unlink()

        # Nothing platform-specific may be non-Linux. A win_amd64/macos wheel here
        # installs cleanly on the build machine and breaks only on the HPC, where
        # there is no network to fix it.
        bad = [w.name for w in wd.glob("*.whl")
               if ("win32" in w.name or "win_amd64" in w.name
                   or "macosx" in w.name or "-linux_" in w.name)]
        if bad:
            print()
            print("!! non-Linux wheels in the bundle -- these will NOT install on the HPC:")
            for b in bad:
                print(f"     {b}")
            failed.append("non-linux-wheels")

    if not args.skip_hf:
        print("\n[2/4] huggingface snapshots")
        from huggingface_hub import snapshot_download
        cache = out / "hf_cache"
        cache.mkdir(parents=True, exist_ok=True)
        models = dict(HF_MODELS)
        if args.with_stt:
            models.update(HF_MODELS_OPTIONAL)
        for repo, why in models.items():
            print(f"  {repo}  ({why})", flush=True)
            try:
                snapshot_download(repo_id=repo, cache_dir=str(cache))
            except Exception as e:
                print(f"  ! {repo} failed: {type(e).__name__}: {e}")
                failed.append(repo)

    print("\n[3/4] repo + our files")
    src_repo = root / "EndpointAnticipation"
    commit = subprocess.run(["git", "-C", str(src_repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    repo_dst = out / "repo"
    if repo_dst.exists():
        shutil.rmtree(repo_dst)
    shutil.copytree(src_repo, repo_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    # keep our modification visible as a patch rather than a silent edit
    diff = subprocess.run(["git", "-C", str(src_repo), "diff"], capture_output=True, text=True).stdout
    (out / "repo_local_changes.patch").write_text(diff, encoding="utf-8")

    epa = out / "epa"
    epa.mkdir(exist_ok=True)
    for d in ("scripts", "configs"):
        dst = epa / d
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(root / d, dst, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("README.md", "requirements-freeze.txt"):
        if (root / f).exists():
            shutil.copy2(root / f, epa / f)

    print("\n[4/4] manifest")

    def du(p):
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

    data_dir = root / "data" / "SpokenWOZ"
    parts = {
        "wheels": du(out / "wheels"),
        "hf_cache": du(out / "hf_cache"),
        "repo": du(repo_dst),
        "epa": du(epa),
    }
    lines = [
        "EPA offline bundle",
        f"built on            : {sys.platform}",
        f"upstream commit     : {commit}",
        f"local patch         : repo_local_changes.patch ({len(diff.splitlines())} lines)",
        f"target              : linux x86_64, cp312, torch 2.9.1+cu128",
        "",
        "contents:",
    ]
    for k, v in parts.items():
        lines.append(f"  {k:10s} {v/1e9:8.2f} GB")
    lines += [
        f"  {'TOTAL':10s} {sum(parts.values())/1e9:8.2f} GB",
        "",
        f"dataset (transferred separately from data/SpokenWOZ): {du(data_dir)/1e9:.2f} GB",
        f"grand total to move : {(sum(parts.values()) + du(data_dir))/1e9:.2f} GB",
        "",
        "on the HPC:",
        "  bash setup_offline.sh /path/to/EPA",
    ]
    if failed:
        lines += ["", "!! FAILED (fetch these manually):"] + [f"  - {f}" for f in failed]
    text = "\n".join(lines)
    (out / "MANIFEST.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
