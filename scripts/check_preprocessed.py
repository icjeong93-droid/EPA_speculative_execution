#!/usr/bin/env python3
"""
Decide whether a split is REALLY preprocessed, not just partly.

Why this exists: the obvious check -- "does filtered_<mode>_context_*.json exist?" --
is wrong, and wrong in the expensive direction. In endpointing_dataset.__init__ the
order is

    handle_length_filtering()   -> writes filtered_<mode>_....json
    ...
    get_audio_feature()
        -> handle_resampling()  -> writes resampled_audios_<mode>_target_sr_<sr>/

so a preprocessing job killed during resampling leaves the filtered json in place.
Both preflight.sh and evaluate.sh used to report "done" for that state, and the next
GPU job would restart the resampling while holding four H100s -- exactly what the
two-stage (CPU preprocess / GPU train) split exists to prevent.

The resampling stage's own completeness rule is in handle_resampling:

    num_resampled >= sum(len(data_json[key]) for key in keys)

i.e. one file per (dialogue, channel) = 2 x dialogues for SpokenWOZ. This script
applies that same rule from the outside.

  python scripts/check_preprocessed.py --dump /path/to/dump --modes train val
  python scripts/check_preprocessed.py --config configs/data/swoz_v1.yaml --modes test

Exit 0 only if every requested mode is complete. Stdlib only, so it runs before the
venv exists.
"""
import argparse
import glob
import json
import os
import re
import sys

STAGES = ("preprocessed", "vad_processed", "processed")


def dump_from_config(path):
    """Pull save_paths.dump out of a data config without needing PyYAML."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^\s*dump:\s*(.+?)\s*$", line)
            if m:
                return m.group(1)
    return None


def count_keys(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return len(json.load(fh))
    except Exception:
        return None


def channels_of(path):
    """Files per dialogue = number of channels (SpokenWOZ separates into 2)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        first = next(iter(data.values()))
        return len(first) if isinstance(first, dict) else 1
    except Exception:
        return None


def check_mode(sw, mode):
    """Returns (verdict, detail-dict). verdict in {OK, MISSING, PARTIAL}."""
    d = {"stages": [], "filtered": None, "dialogues": None,
         "resampled": None, "expected": None}

    for st in STAGES:
        if not os.path.isfile(os.path.join(sw, f"{st}_{mode}.json")):
            d["stages"].append(f"{st}_{mode}.json")

    filt = sorted(glob.glob(os.path.join(sw, f"filtered_{mode}_context_*.json")))
    if not filt:
        return ("MISSING" if d["stages"] else "PARTIAL"), d
    d["filtered"] = os.path.basename(filt[0])
    d["dialogues"] = count_keys(filt[0])
    nch = channels_of(filt[0]) or 1

    rs = sorted(glob.glob(os.path.join(sw, f"resampled_audios_{mode}_target_sr_*")))
    if not rs:
        d["resampled"], d["expected"] = 0, (d["dialogues"] or 0) * nch
        return "PARTIAL", d

    d["resampled"] = len([f for f in os.listdir(rs[0])
                          if os.path.isfile(os.path.join(rs[0], f))])
    d["expected"] = (d["dialogues"] or 0) * nch

    if d["stages"]:
        return "PARTIAL", d
    # handle_resampling itself accepts >=, so match it exactly
    return ("OK" if d["resampled"] >= d["expected"] else "PARTIAL"), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None, help="dump root (contains <dataset>/)")
    ap.add_argument("--config", default=None,
                    help="data config to read save_paths.dump from instead")
    ap.add_argument("--dataset", default="spokenwoz")
    ap.add_argument("--modes", nargs="+", default=["train", "val"],
                    choices=["train", "val", "test"])
    ap.add_argument("--quiet", action="store_true", help="verdict only, no table")
    args = ap.parse_args()

    dump = args.dump or (dump_from_config(args.config) if args.config else None)
    if not dump:
        raise SystemExit("need --dump or --config (with save_paths.dump)")
    # forward slashes so the remedy commands below are copy-pasteable everywhere
    sw = os.path.join(dump, args.dataset).replace("\\", "/")
    if not os.path.isdir(sw):
        if not args.quiet:
            print(f"dump not found: {sw}   -> nothing preprocessed yet")
        return 1

    rows, bad = [], []
    for mode in args.modes:
        verdict, d = check_mode(sw, mode)
        rows.append((mode, verdict, d))
        if verdict != "OK":
            bad.append((mode, d))

    if not args.quiet:
        print(f"dump: {sw}")
        print(f"{'mode':6s} {'verdict':8s} {'dialogues':>9s} {'resampled/expected':>20s}  missing stages")
        for mode, verdict, d in rows:
            rs = "-" if d["resampled"] is None else f"{d['resampled']}/{d['expected']}"
            dl = "-" if d["dialogues"] is None else str(d["dialogues"])
            print(f"{mode:6s} {verdict:8s} {dl:>9s} {rs:>20s}  "
                  f"{' '.join(d['stages']) if d['stages'] else '-'}")

    # The one interrupted state that is NOT fixed by re-running: upstream's
    # handle_and_add_turns returns (not continues) when the first mode's output
    # already exists, so processed_train.json without processed_val.json makes every
    # rerun a silent no-op and training dies later on a missing processed_val.json.
    if (os.path.isfile(os.path.join(sw, "processed_train.json"))
            and not os.path.isfile(os.path.join(sw, "processed_val.json"))):
        print()
        print("!! preprocessing was interrupted BETWEEN train and val.")
        print("   Re-running will NOT fix it (src/data/data_processing.py:29 returns")
        print("   instead of continuing). Delete and redo:")
        print(f"     rm {sw}/processed_*.json")
        print("     sbatch scripts/preprocess.sbatch")
        return 1

    if bad:
        print()
        for mode, d in bad:
            if d["resampled"] is not None and d["expected"] and d["resampled"] < d["expected"]:
                print(f"!! [{mode}] resampling is INCOMPLETE "
                      f"({d['resampled']}/{d['expected']} files).")
                print(f"   filtered_{mode}_... exists, so a filename-only check would say")
                print("   'done' and the GPU job would resample while holding the GPUs.")
            else:
                print(f"!! [{mode}] not preprocessed.")
        arg = "" if set(args.modes) <= {"train", "val"} else " " + " ".join(
            m for m in args.modes if m == "test")
        print(f"   Re-run (CPU job; finished stages are skipped):")
        print(f"     sbatch scripts/preprocess.sbatch{arg}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
