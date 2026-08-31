#!/usr/bin/env python3
"""
Download SpokenWOZ and arrange it into the layout src/data/spokenwoz_dataset.py expects.

Target layout (raw_path in the data config points here):

  SpokenWOZ/
    audio_5700_train_dev/      *.wav   (2-channel, 8 kHz; ch0=user ch1=system)
    audio_5700_test/           *.wav
    text_5700_train_dev/       data.json, valListFile.json
    text_5700_test/            data.json, testListFile.json

Source: https://spokenwoz.github.io/  (CC BY-NC 4.0)
Download size ~12.5 GB compressed, ~30 GB extracted.

Usage:
  python scripts/download_spokenwoz.py --root /path/to/EPA
  python scripts/download_spokenwoz.py --root /path/to/EPA --skip-test
"""
import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOS = {
    "train_audio": ("ssz1111/SpokenWOZ-Train-Audio", ["audio_5700_train_dev.tar.gz"]),
    "train_text": ("ssz1111/SpokenWOZ-Train-Text", ["data.json", "valListFile.json"]),
    "test_audio": ("ssz1111/SpokenWOZ-Test-Audio-Fixed", ["audio_5700_test.zip"]),
    "test_text": ("ssz1111/SpokenWOZ-Test-Text-Fixed", ["data.json", "testListFile.json"]),
}


def fetch(repo, fname, cache):
    print(f"  fetching {repo}/{fname} ...", flush=True)
    return hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", cache_dir=cache)


def extract(archive: Path, dest_parent: Path, expect_dir: str):
    """Extract archive under dest_parent so that dest_parent/expect_dir exists."""
    target = dest_parent / expect_dir
    if target.is_dir() and any(target.iterdir()):
        print(f"  {expect_dir}/ already populated ({len(list(target.iterdir()))} entries) - skip")
        return
    staging = dest_parent / f".staging_{expect_dir}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    print(f"  extracting {archive.name} -> {staging} ...", flush=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(staging)
    else:
        with tarfile.open(archive) as t:
            t.extractall(staging)
    # the archive may or may not carry its own top-level folder
    JUNK = {"__MACOSX"}
    entries = [p for p in staging.iterdir()
               if not p.name.startswith(".") and p.name not in JUNK]
    for j in staging.iterdir():
        if j.name in JUNK:
            shutil.rmtree(j, ignore_errors=True)
    if len(entries) == 1 and entries[0].is_dir():
        src = entries[0]
    else:
        src = staging
    target.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name.startswith("._"):
            continue
        shutil.move(str(p), str(target / p.name))
    shutil.rmtree(staging)
    print(f"  -> {target} ({len(list(target.iterdir()))} files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", default=None, help="HF cache dir (default: HF default)")
    ap.add_argument("--skip-test", action="store_true", help="train/dev only (enough to train)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = root / "data" / "SpokenWOZ"
    out.mkdir(parents=True, exist_ok=True)
    print(f"SpokenWOZ -> {out}\n")

    # --- text (small) ---
    repo, files = REPOS["train_text"]
    d = out / "text_5700_train_dev"; d.mkdir(exist_ok=True)
    for f in files:
        p = fetch(repo, f, args.cache)
        shutil.copy(p, d / f)
    print(f"  -> {d}")

    if not args.skip_test:
        repo, files = REPOS["test_text"]
        d = out / "text_5700_test"; d.mkdir(exist_ok=True)
        for f in files:
            p = fetch(repo, f, args.cache)
            shutil.copy(p, d / f)
        print(f"  -> {d}")

    # --- audio (large) ---
    repo, files = REPOS["train_audio"]
    p = Path(fetch(repo, files[0], args.cache))
    extract(p, out, "audio_5700_train_dev")

    if not args.skip_test:
        repo, files = REPOS["test_audio"]
        p = Path(fetch(repo, files[0], args.cache))
        extract(p, out, "audio_5700_test")

    # --- verify against what spokenwoz_dataset.py reads ---
    print("\nverification:")
    ok = True
    checks = [
        ("audio_5700_train_dev", "dir"),
        ("text_5700_train_dev/data.json", "file"),
        ("text_5700_train_dev/valListFile.json", "file"),
    ]
    if not args.skip_test:
        checks += [("audio_5700_test", "dir"), ("text_5700_test/data.json", "file")]
    for rel, kind in checks:
        p = out / rel
        good = p.is_dir() if kind == "dir" else p.is_file()
        extra = f" ({len(list(p.iterdir()))} files)" if good and kind == "dir" else ""
        print(f"  [{'OK ' if good else 'MISS'}] {rel}{extra}")
        ok &= good
    print("\nOK - set raw_path to:" if ok else "\nINCOMPLETE - check errors above")
    if ok:
        print(f"  {out.as_posix()}")


if __name__ == "__main__":
    main()
