#!/usr/bin/env python3
"""
Download SpokenWOZ and arrange it into the layout src/data/spokenwoz_dataset.py expects.

STANDALONE: no dependency on the rest of this repo. Copy this single file to any
machine that has internet and run it. Only `huggingface_hub` is required.

    pip install huggingface_hub
    python download_spokenwoz.py --root /somewhere            # download + arrange
    python download_spokenwoz.py --root /somewhere --verify   # check an existing copy
    python download_spokenwoz.py --root /somewhere --skip-test  # train/dev only

Behind a corporate proxy or an internal mirror:
    export HTTPS_PROXY=http://proxy.corp:8080
    export HF_ENDPOINT=https://your-hf-mirror

Resumable: rerun after an interruption. Completed stages are skipped, and partial
HF downloads continue from where they stopped.

Target layout (the data config's raw_path points at <root>/data/SpokenWOZ):

  SpokenWOZ/
    audio_5700_train_dev/      4700 wav   (2-channel, 8 kHz; ch0=user ch1=system)
    audio_5700_test/           1000 wav
    text_5700_train_dev/       data.json, valListFile.json
    text_5700_test/            data.json, testListFile.json

Source: https://spokenwoz.github.io/  (CC BY-NC 4.0)
Size: ~12.5 GB compressed, ~29 GB on disk.
"""
import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise SystemExit("huggingface_hub is required:\n    pip install huggingface_hub")

REPOS = {
    "train_audio": ("ssz1111/SpokenWOZ-Train-Audio", ["audio_5700_train_dev.tar.gz"]),
    "train_text": ("ssz1111/SpokenWOZ-Train-Text", ["data.json", "valListFile.json"]),
    "test_audio": ("ssz1111/SpokenWOZ-Test-Audio-Fixed", ["audio_5700_test.zip"]),
    "test_text": ("ssz1111/SpokenWOZ-Test-Text-Fixed", ["data.json", "testListFile.json"]),
}

# archives produced on macOS carry these siblings; they break the
# "single top-level dir" heuristic and leave AppleDouble sidecars behind
JUNK_DIRS = {"__MACOSX"}


def fetch(repo, fname, cache):
    print(f"  fetching {repo}/{fname} ...", flush=True)
    return hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", cache_dir=cache)


def extract(archive: Path, dest_parent: Path, expect_dir: str):
    """Extract archive under dest_parent so that dest_parent/expect_dir holds the files."""
    target = dest_parent / expect_dir
    if target.is_dir() and any(target.iterdir()):
        print(f"  {expect_dir}/ already populated ({len(list(target.iterdir()))} entries) - skip")
        return
    staging = dest_parent / f".staging_{expect_dir}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    print(f"  extracting {archive.name} ...", flush=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(staging)
    else:
        with tarfile.open(archive) as t:
            t.extractall(staging)

    for j in list(staging.iterdir()):
        if j.name in JUNK_DIRS:
            shutil.rmtree(j, ignore_errors=True)

    entries = [p for p in staging.iterdir() if not p.name.startswith(".")]
    src = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging

    target.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name.startswith("._"):          # AppleDouble sidecars
            continue
        shutil.move(str(p), str(target / p.name))
    shutil.rmtree(staging, ignore_errors=True)
    print(f"  -> {target} ({len(list(target.iterdir()))} files)")


def verify(out: Path, skip_test: bool) -> bool:
    exact = {"audio_5700_train_dev": 4700}
    files = ["text_5700_train_dev/data.json", "text_5700_train_dev/valListFile.json"]
    if not skip_test:
        exact["audio_5700_test"] = 1000
        files.append("text_5700_test/data.json")

    ok = True
    for rel, want in exact.items():
        n = len(list((out / rel).glob("*.wav"))) if (out / rel).is_dir() else 0
        good = n == want
        ok &= good
        print(f"  [{'OK ' if good else 'BAD'}] {rel}: {n} wav (expected {want})")
    for rel in files:
        good = (out / rel).is_file()
        ok &= good
        print(f"  [{'OK ' if good else 'BAD'}] {rel}")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n  total on disk: {total/1e9:.2f} GB")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset lands in <root>/data/SpokenWOZ")
    ap.add_argument("--cache", default=None, help="HF cache dir (default: HF default)")
    ap.add_argument("--skip-test", action="store_true", help="train/dev only (enough to train)")
    ap.add_argument("--verify", action="store_true", help="check an existing copy; download nothing")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = root / "data" / "SpokenWOZ"
    print(f"SpokenWOZ -> {out}\n")

    if args.verify:
        if not out.is_dir():
            raise SystemExit(f"not found: {out}")
        raise SystemExit(0 if verify(out, args.skip_test) else 1)

    out.mkdir(parents=True, exist_ok=True)

    # --- text (small) ---
    repo, files = REPOS["train_text"]
    d = out / "text_5700_train_dev"
    d.mkdir(exist_ok=True)
    for f in files:
        shutil.copy(fetch(repo, f, args.cache), d / f)
    print(f"  -> {d}")

    if not args.skip_test:
        repo, files = REPOS["test_text"]
        d = out / "text_5700_test"
        d.mkdir(exist_ok=True)
        for f in files:
            shutil.copy(fetch(repo, f, args.cache), d / f)
        print(f"  -> {d}")

    # --- audio (large) ---
    repo, files = REPOS["train_audio"]
    extract(Path(fetch(repo, files[0], args.cache)), out, "audio_5700_train_dev")

    if not args.skip_test:
        repo, files = REPOS["test_audio"]
        extract(Path(fetch(repo, files[0], args.cache)), out, "audio_5700_test")

    print("\nverification:")
    if verify(out, args.skip_test):
        print("\nOK - set raw_path to:")
        print(f"  {out.as_posix()}")
    else:
        raise SystemExit("\nINCOMPLETE - rerun this script (it resumes)")


if __name__ == "__main__":
    main()
