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
import re
import zipfile
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
             "manylinux_2_24_x86_64", "manylinux2014_x86_64",
             "manylinux_2_17_x86_64"]
PY_VERSION = "3.12"
# hf-xet ships only cp38-abi3 wheels; offering just cp312 rejects them and the
# bundle silently loses the dependency again. Pure-python (abi "none") wheels
# are matched regardless of what is listed here.
ABIS = ["cp312", "abi3"]

# torchcodec is NOT optional. torchaudio 2.9 routes torchaudio.load/save through
# it unconditionally (torchaudio/__init__.py:86,178) and raises ImportError when it
# is missing -- which is every audio read in preprocessing. Leaving it out produced
# a bundle that installed cleanly and then died on the first torchaudio.load, two
# hours into a CPU job, after preflight had said PASS.
# It is belt-and-braces with scripts/sitecustomize.py: torchcodec additionally needs
# FFmpeg shared libraries at runtime, which an HPC image may not carry, and the shim
# takes over when the import fails for that reason.
TORCH = ["torch==2.9.1", "torchaudio==2.9.1", "torchcodec==0.16.0"]

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
    # Same trap, no CUDA involved: huggingface_hub 0.36 requires hf-xet under
    # the marker platform_machine == "x86_64" or "amd64" or ... -- Windows
    # reports "AMD64" and marker comparison is case-sensitive, so pip on a
    # Windows build host skips it. The HPC then dies at pip install with
    # "Could not find a version that satisfies the requirement hf-xet".
    # Unpinned: the constraint (>=1.1.3,<2) comes from huggingface_hub itself.
    "hf-xet",
    # moshi requires this under sys_platform == "linux" -- false on a Windows
    # build host, so pip drops it and the HPC fails at install. Version range
    # is moshi's own; leave it unpinned inside that range.
    "bitsandbytes>=0.45,<0.50",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

# The upstream clone is NOT tracked by this repo (.gitignore: /EndpointAnticipation/)
# because it carries its own .git. A fresh checkout on a new machine therefore does
# not have it, and the bundle build used to discover that only at step [3/4] -- after
# downloading 4.8 GB -- as a bare OS "path not found".
UPSTREAM_URL = "https://github.com/bloodraven66/EndpointAnticipation.git"
UPSTREAM_COMMIT = "531e1d7e70751e980b82088a3a358a2e75fa8a12"
# marker proving patches/epa-num-workers.patch is applied to the clone
PATCH_MARKER = ("anticipation-model/src/data/__init__.py", "EPA_NUM_WORKERS")

PYPI = [
    "numpy", "matplotlib", "PyYAML", "huggingface_hub", "tqdm", "moshi",
    "librosa", "scikit-learn", "easydict", "coloredlogs", "soundfile",
    "silero-vad", "transformers", "wandb",
    # experiment tracking: logs to a local file store, no server.
    # skinny is enough to WRITE runs -- the UI runs on a laptop.
    "mlflow-skinny",
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


# The bundle is built on Windows and unpacked on Linux. git's eol=lf only governs
# what is COMMITTED -- with core.autocrlf=true the working tree holds CRLF, and this
# script copies the working tree. A shell script copied verbatim then arrives on the
# HPC as `#!/usr/bin/env bash\r`, and the first command in the runbook
# (bash preflight.sh) dies with a syntax error on every line. Normalise on copy so
# the bundle cannot depend on how the build machine happens to be checked out.
TEXT_SUFFIXES = {".sh", ".sbatch", ".py", ".yaml", ".yml", ".md", ".txt",
                 ".patch", ".ps1", ".cfg", ".toml", ".json", ".in"}
TEXT_NAMES = {".gitignore", ".gitattributes"}


def copy_lf(src, dst, *, follow_symlinks=True):
    """shutil.copy2 that rewrites CRLF -> LF in text files."""
    p = Path(src)
    if p.suffix.lower() in TEXT_SUFFIXES or p.name in TEXT_NAMES:
        data = p.read_bytes()
        if b"\x00" not in data:                      # not actually binary
            Path(dst).write_bytes(data.replace(b"\r\n", b"\n"))
            shutil.copystat(src, dst)
            return dst
    return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)


def crlf_offenders(root_dir):
    """Text files under root_dir that still carry CRLF. Should always be empty."""
    out = []
    for f in Path(root_dir).rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in TEXT_SUFFIXES and f.name not in TEXT_NAMES:
            continue
        try:
            data = f.read_bytes()
        except OSError:
            continue
        if b"\x00" not in data and b"\r\n" in data:
            out.append(f)
    return out


def check_upstream(root, allow_drift=False):
    """Validate the upstream clone BEFORE anything is downloaded.

    Three ways this used to go wrong silently or late:
      * missing clone   -> bare "path not found" at [3/4], after 4.8 GB of downloads
      * wrong commit    -> a bundle that does not reproduce the pinned code
      * patch not applied -> the shipped repo keeps num_workers=8 hardcoded, so
        EPA_NUM_WORKERS is ignored on the HPC (preprocess.sbatch relies on =0)
    """
    src = root / "EndpointAnticipation"
    how = (f"  git clone {UPSTREAM_URL} \"{src}\"\n"
           f"  git -C \"{src}\" checkout {UPSTREAM_COMMIT}\n"
           f"  git -C \"{src}\" apply \"{root / 'patches' / 'epa-num-workers.patch'}\"")

    if not src.is_dir():
        raise SystemExit(
            f"!! upstream clone not found: {src}\n\n"
            f"   It is deliberately untracked (.gitignore: /EndpointAnticipation/) because it\n"
            f"   is a separate repository, so a fresh checkout of this repo does not have it.\n"
            f"   Clone it at the pinned commit and apply our one patch:\n\n{how}")

    if not (src / ".git").exists():
        raise SystemExit(
            f"!! {src} exists but is not a git clone.\n"
            f"   The bundle records the upstream commit for reproducibility.\n"
            f"   Remove it and redo:\n\n{how}")

    r = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    head = r.stdout.strip()
    if r.returncode != 0 or not head:
        raise SystemExit(f"!! cannot read HEAD of {src}: {r.stderr.strip()}\n\n{how}")

    if head != UPSTREAM_COMMIT:
        msg = (f"!! upstream is at {head[:12]}, expected {UPSTREAM_COMMIT[:12]}.\n"
               f"   The bundle would ship code the reproduction was not verified against.\n"
               f"   Fix:  git -C \"{src}\" checkout {UPSTREAM_COMMIT}\n"
               f"   (then re-apply patches/epa-num-workers.patch)\n"
               f"   Override with --allow-upstream-drift if this is deliberate.")
        if not allow_drift:
            raise SystemExit(msg)
        print(msg.replace("!!", "warning:"))

    marker_file, marker = PATCH_MARKER
    p = src / marker_file
    if not p.is_file():
        raise SystemExit(f"!! upstream clone looks wrong -- {p} is missing.\n\n{how}")
    if marker not in p.read_text(encoding="utf-8", errors="replace"):
        raise SystemExit(
            f"!! patches/epa-num-workers.patch is NOT applied to the clone\n"
            f"   ({marker} not found in {marker_file}).\n"
            f"   Without it num_workers stays hardcoded at 8 and EPA_NUM_WORKERS is\n"
            f"   ignored on the HPC -- preprocess.sbatch depends on EPA_NUM_WORKERS=0.\n"
            f"   Fix:  git -C \"{src}\" apply \"{root / 'patches' / 'epa-num-workers.patch'}\"")

    print(f"upstream: {src}")
    print(f"          commit {head[:12]} (pinned), epa-num-workers.patch applied")
    return head


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=False, **kw)


# A dependency guarded by one of these is invisible to pip on a Windows build
# host: markers are evaluated against the machine running pip, not against
# --platform. hf-xet and bitsandbytes both reached the HPC as an install-time
# ERROR after 4.8 GB had already been moved. Catch the whole class here instead.
LINUX_SYS = re.compile(r"""sys_platform\s*==\s*['"]linux['"]""")
MACHINE_X86 = re.compile(r"""platform_machine\s*==\s*['"](x86_64|amd64|AMD64)['"]""")


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def linux_only_gaps(wheel_dir):
    """Requirements that apply on linux/x86_64 but have no wheel in the bundle."""
    have = {canon(w.name.split("-")[0]) for w in wheel_dir.glob("*.whl")}
    missing = {}
    for w in sorted(wheel_dir.glob("*.whl")):
        try:
            with zipfile.ZipFile(w) as z:
                meta = next((n for n in z.namelist()
                             if n.endswith(".dist-info/METADATA")), None)
                if meta is None:
                    continue
                text = z.read(meta).decode("utf-8", "replace")
        except (zipfile.BadZipFile, OSError, KeyError):
            continue
        for line in text.splitlines():
            if not line.startswith("Requires-Dist:"):
                continue
            req = line.split(":", 1)[1].strip()
            if ";" not in req:
                continue
            name, marker = req.split(";", 1)
            # extras are not installed unless requested, so they are not gaps
            if "extra" in marker:
                continue
            if not (LINUX_SYS.search(marker) or MACHINE_X86.search(marker)):
                continue
            dist = canon(re.split(r"[<>=!~\[ (]", name.strip())[0])
            if dist and dist not in have:
                missing.setdefault(dist, w.name)
    return missing


def dl_wheels(py, dest, pkgs, index=None):
    dest.mkdir(parents=True, exist_ok=True)
    base = [py, "-m", "pip", "download", "--dest", str(dest),
            "--python-version", PY_VERSION, "--implementation", "cp"]
    for a in ABIS:
        base += ["--abi", a]
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
    ap.add_argument("--allow-upstream-drift", action="store_true",
                    help="build even if the upstream clone is not at the pinned commit")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"bundle -> {out}")

    # Check the cheap precondition first: a missing or wrong upstream clone used to
    # surface only at [3/4], after gigabytes had already been fetched.
    commit = check_upstream(root, allow_drift=args.allow_upstream_drift)
    print()

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

        # Presence check for the wheels whose absence only shows up at runtime on a
        # machine with no network. torchcodec/soundfile are the audio I/O path;
        # without one of them nothing can read a wav.
        REQUIRED = {
            "torch (cu128)": "torch-*+cu128*.whl",
            "torchaudio (cu128)": "torchaudio-*+cu128*.whl",
            "torchcodec": "torchcodec-*.whl",
            "soundfile": "soundfile-*.whl",
            "hf-xet": "hf_xet-*.whl",
            "bitsandbytes": "bitsandbytes-*.whl",
            "mlflow-skinny": "mlflow_skinny-*.whl",
        }
        for label, pat in REQUIRED.items():
            if not list(wd.glob(pat)):
                print(f"!! required wheel missing from bundle: {label}  ({pat})")
                failed.append(f"missing-wheel:{label}")

        # Generalisation of the list above: anything a downloaded wheel requires
        # on linux/x86_64 must itself be in the bundle. Named packages can be
        # forgotten; this cannot.
        gaps = linux_only_gaps(wd)
        if gaps:
            print()
            print("!! linux-only dependencies missing from the bundle "
                  "(pip skipped them: this build host is not Linux)")
            for dist, src in sorted(gaps.items()):
                print(f"     {dist}   <- required by {src}")
            print("   Add each to LINUX_ONLY and rebuild, or the HPC fails at")
            print("   pip install with 'No matching distribution found'.")
            failed.append("linux-marker-gaps")
        else:
            print("  linux-only deps: all present")

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
    src_repo = root / "EndpointAnticipation"   # validated by check_upstream() above
    repo_dst = out / "repo"
    if repo_dst.exists():
        shutil.rmtree(repo_dst)
    shutil.copytree(src_repo, repo_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
                    copy_function=copy_lf)
    # keep our modification visible as a patch rather than a silent edit
    diff = subprocess.run(["git", "-C", str(src_repo), "diff"], capture_output=True, text=True).stdout
    # newline="\n": written on Windows, applied on Linux. A CRLF patch does not apply.
    (out / "repo_local_changes.patch").write_text(diff, encoding="utf-8", newline="\n")
    if not diff.strip():
        # check_upstream already proved the change is present in the files, so this
        # only means it was committed into the clone rather than left in the worktree.
        print("  note: repo_local_changes.patch is empty -- the num_workers change is"
              " committed in the clone, not a working-tree diff. repo/ still carries it.")

    epa = out / "epa"
    epa.mkdir(exist_ok=True)
    # patches/ travels too: it is the auditable record of what we changed in the
    # upstream clone, and the HPC needs the files to re-apply one by hand.
    for d in ("scripts", "configs", "patches"):
        dst = epa / d
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(root / d, dst, ignore=shutil.ignore_patterns("__pycache__"),
                        copy_function=copy_lf)
    for f in ("README.md", "requirements-freeze.txt"):
        if (root / f).exists():
            copy_lf(root / f, epa / f)

    # The installer has to sit at the BUNDLE ROOT: README section 3 and transfer.sh
    # both say `cd bundle && bash setup_offline.sh`. It used to be copied only into
    # epa/scripts/, so a freshly built bundle failed that command with
    # "No such file or directory" -- the copy at the root was a manual leftover.
    installer = root / "scripts" / "setup_offline.sh"
    if not installer.is_file():
        raise SystemExit(f"installer not found: {installer}")
    copy_lf(installer, out / "setup_offline.sh")
    os.chmod(out / "setup_offline.sh", 0o755)
    print(f"  setup_offline.sh -> {out / 'setup_offline.sh'}")

    # The bundle must be byte-clean for Linux before it leaves this machine --
    # there is no way to fix a CRLF shell script on a host with no network.
    offenders = crlf_offenders(epa) + crlf_offenders(repo_dst)
    if (out / "setup_offline.sh").read_bytes().count(b"\r\n"):
        offenders.append(out / "setup_offline.sh")
    if offenders:
        print("!! CRLF survived into the bundle (would break on Linux):")
        for o in offenders[:20]:
            print(f"     {o}")
        failed.append("crlf-in-bundle")
    else:
        print("  line endings: LF throughout (checked epa/, repo/, setup_offline.sh)")

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
    (out / "MANIFEST.txt").write_text(text, encoding="utf-8", newline="\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
