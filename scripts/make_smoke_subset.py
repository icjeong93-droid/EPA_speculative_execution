#!/usr/bin/env python3
"""
Carve a small SpokenWOZ subset so the full pipeline (channel split -> 24 kHz
resample -> Silero VAD turn annotation -> training) can be validated in minutes
instead of hours.

Full preprocessing of SpokenWOZ inflates to ~110 GB of dump (8 kHz channel-split
plus 24 kHz resampled copies) and takes 1-2 h. That is worth doing once, on the
machine that will train -- not twice. Use this to prove the pipeline works, then
point the real config at the full dataset.

  python scripts/make_smoke_subset.py --root . --train 32 --val 8
  # then: --data-config swoz_smoke_v1

Selection is deterministic (sorted, then head) so runs are comparable.
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--train", type=int, default=32, help="train dialogues to keep")
    ap.add_argument("--val", type=int, default=8, help="val dialogues to keep")
    ap.add_argument("--name", default="SpokenWOZ_smoke")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    src = root / "data" / "SpokenWOZ"
    dst = root / "data" / args.name
    if not src.is_dir():
        raise SystemExit(f"full dataset not found at {src} -- run download_spokenwoz.py first")

    data = json.loads((src / "text_5700_train_dev" / "data.json").read_text(encoding="utf-8"))
    val_ids = [l.strip() for l in (src / "text_5700_train_dev" / "valListFile.json")
               .read_text(encoding="utf-8").splitlines() if l.strip()]
    val_set = set(val_ids)

    all_ids = sorted(data)
    train_pick = [k for k in all_ids if k not in val_set][: args.train]
    val_pick = [k for k in all_ids if k in val_set][: args.val]
    keep = train_pick + val_pick
    print(f"keeping {len(train_pick)} train + {len(val_pick)} val = {len(keep)} dialogues")

    (dst / "text_5700_train_dev").mkdir(parents=True, exist_ok=True)
    audio_dst = dst / "audio_5700_train_dev"
    audio_dst.mkdir(parents=True, exist_ok=True)

    (dst / "text_5700_train_dev" / "data.json").write_text(
        json.dumps({k: data[k] for k in keep}, ensure_ascii=False), encoding="utf-8")
    (dst / "text_5700_train_dev" / "valListFile.json").write_text(
        "\n".join(val_pick) + "\n", encoding="utf-8")

    audio_src = src / "audio_5700_train_dev"
    total = 0
    for k in keep:
        s = audio_src / f"{k}.wav"
        if not s.is_file():
            print(f"  !! missing audio for {k}")
            continue
        d = audio_dst / s.name
        if not d.exists():
            shutil.copy2(s, d)
        total += d.stat().st_size

    # the loader also opens the test split; give it a valid but tiny one
    (dst / "text_5700_test").mkdir(exist_ok=True)
    (dst / "text_5700_test" / "data.json").write_text("{}", encoding="utf-8")
    (dst / "audio_5700_test").mkdir(exist_ok=True)

    print(f"audio copied: {total/1e6:.0f} MB -> {audio_dst}")
    print(f"\nsubset at {dst.as_posix()}")
    print("generate its config with:")
    print(f'  python scripts/make_configs.py --root "{root.as_posix()}" '
          f'--spokenwoz "{dst.as_posix()}" --out "{(root/"configs_smoke").as_posix()}" --num-workers 8')


if __name__ == "__main__":
    main()
