#!/usr/bin/env python3
"""
Turn the inference sweep into the paper's Table 1.

ForecastingTrainer.infer_loop sweeps 40 decision thresholds and saves the whole
curve to infer_results.pt. The paper does not report a curve -- it reports one
row per (model, horizon), read at a threshold tuned to a chosen ERC operating
point (~33.8% on SpokenWOZ). This script does that selection and prints the row,
so "reproduced or not" is a number and not an eyeball on a plot.

  python scripts/report_table1.py --root /scratch/$USER/EPA
  python scripts/report_table1.py --root . --target-erc 33.8 --csv sweep.csv
  python scripts/report_table1.py --root . --threshold 0.35     # fixed threshold instead

Metric names in infer_results.pt -> paper (src/training/forecasting_trainer.py):
  median_forecast              -> MRA (ms)   median of (t_EOT - t_pred)
  accuracies_with_collar       -> HEA (frac) fired inside the horizon + collar
  ep_cutoff                    -> PAR (%)    turns with >=1 premature anticipation
  total_cutoff_proportions_mean-> ERC (frac) premature / max possible, per turn
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch

# Table 1, SpokenWOZ block (arXiv 2606.13450). MRA ms, HEA %, PAR %, ERC %.
PAPER = {
    ("EPA-S", 640):  (640, 66.3, 66.5, 33.9),
    ("EPA-M", 640):  (640, 67.0, 66.2, 33.8),
    ("EPA-S", 1280): (1200, 50.3, 53.9, 33.7),
    ("EPA-M", 1280): (1120, 49.7, 52.8, 33.2),
}
# the reproduction gate (README section 1)
GATE = ("EPA-M", 640)
GATE_TOL = {"MRA": 100.0, "HEA": 3.0, "ERC": 3.0}


def variant_of(run_name: str) -> str:
    """fcall carries every horizon in one model -- that is EPA-M. Anything else is EPA-S."""
    return "EPA-M" if re.search(r"__fcall[_.]", run_name + "_") else "EPA-S"


def load_runs(root: Path, ckpt_dir: Path):
    runs = []
    for f in sorted(ckpt_dir.glob("*/infer_results.pt")):
        m = torch.load(f, map_location="cpu", weights_only=False)
        folder = f.parent.name                      # <run_name>__<dataset>
        run_name, _, dataset = folder.rpartition("__")
        runs.append({"path": f, "folder": folder, "run": run_name,
                     "dataset": dataset, "metrics": m})
    return runs


def rows_for(entry, target_erc, fixed_threshold):
    """One row per horizon, at the threshold whose ERC lands on the operating point."""
    m = entry["metrics"]
    horizons = list(m["forecast_intervals"])
    thresholds = sorted(m["ep_cutoff"].keys())
    erc = np.array([np.asarray(m["total_cutoff_proportions_mean"][t], dtype=float) for t in thresholds]) * 100
    hea = np.array([np.asarray(m["accuracies_with_collar"][t], dtype=float) for t in thresholds]) * 100
    par = np.array([np.asarray(m["ep_cutoff"][t], dtype=float) for t in thresholds])
    mra = np.array([np.asarray(m["median_forecast"][t], dtype=float) for t in thresholds])

    variant = variant_of(entry["run"])
    out = []
    for hi, h in enumerate(horizons):
        # the paper reads each row at its own ERC, so default to that row's value
        # instead of forcing every horizon onto a single operating point
        tgt = target_erc if target_erc is not None else PAPER.get((variant, int(h)), (0, 0, 0, 33.8))[3]
        if fixed_threshold is not None:
            ti = int(np.argmin(np.abs(np.array(thresholds) - fixed_threshold)))
        else:
            ti = int(np.argmin(np.abs(erc[:, hi] - tgt)))
        out.append({
            "run": entry["run"], "dataset": entry["dataset"],
            "variant": variant, "h": int(h),
            "threshold": thresholds[ti], "target_erc": tgt,
            "MRA": mra[ti, hi], "HEA": hea[ti, hi], "PAR": par[ti, hi], "ERC": erc[ti, hi],
        })
    return out, thresholds, {"ERC": erc, "HEA": hea, "PAR": par, "MRA": mra}, horizons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="EPA working root")
    ap.add_argument("--checkpoints", default=None, help="default <root>/checkpoints")
    ap.add_argument("--target-erc", type=float, default=None,
                    help="force one ERC operating point for every row. Default: each row uses its own paper ERC (33.8/33.9/33.7/33.2).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="read at this fixed threshold instead of matching ERC")
    ap.add_argument("--csv", default=None, help="also dump the full threshold sweep here")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ckpt_dir = Path(args.checkpoints) if args.checkpoints else root / "checkpoints"
    runs = load_runs(root, ckpt_dir)
    if not runs:
        raise SystemExit(
            f"no infer_results.pt under {ckpt_dir}/*/\n"
            f"run the evaluation first:  bash scripts/evaluate.sh")

    if args.threshold is not None:
        print(f"reading every row at fixed threshold {args.threshold}\n")
    elif args.target_erc is not None:
        print(f"reading each row at the threshold whose ERC is closest to {args.target_erc}%\n")
    else:
        print("reading each row at the threshold whose ERC is closest to that row's paper ERC")
        print()

    all_rows, sweep_rows, dead = [], [], []
    for e in runs:
        rows, thresholds, curves, horizons = rows_for(e, args.target_erc, args.threshold)
        all_rows += rows
        # A model whose probabilities never reach the sweep's lower bound produces
        # an all-zero table that looks like a perfect ERC of 0%. Say so explicitly --
        # zeros here mean "never fired", not "no redundant computation".
        for hi, h in enumerate(horizons):
            if curves["PAR"][:, hi].max() == 0 and curves["MRA"][:, hi].max() == 0:
                dead.append((e["run"], int(h)))
        for ti, t in enumerate(thresholds):
            for hi, h in enumerate(horizons):
                sweep_rows.append({
                    "run": e["run"], "dataset": e["dataset"], "h": int(h), "threshold": t,
                    "MRA_ms": round(float(curves["MRA"][ti, hi]), 1),
                    "HEA_pct": round(float(curves["HEA"][ti, hi]), 2),
                    "PAR_pct": round(float(curves["PAR"][ti, hi]), 2),
                    "ERC_pct": round(float(curves["ERC"][ti, hi]), 2),
                })

    hdr = f"{'variant':7s} {'h(ms)':>6s} {'thr':>5s} {'MRA(ms)':>8s} {'HEA(%)':>7s} {'PAR(%)':>7s} {'ERC(%)':>7s} {'@ERC':>6s}   paper (MRA/HEA/PAR/ERC)"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(all_rows, key=lambda x: (x["variant"], x["h"])):
        ref = PAPER.get((r["variant"], r["h"])) if r["dataset"] == "spokenwoz" else None
        ref_s = f"   {ref[0]:.0f} / {ref[1]:.1f} / {ref[2]:.1f} / {ref[3]:.1f}" if ref else ""
        print(f"{r['variant']:7s} {r['h']:6d} {r['threshold']:5.2f} "
              f"{r['MRA']:8.0f} {r['HEA']:7.1f} {r['PAR']:7.1f} {r['ERC']:7.1f} {r['target_erc']:6.1f}{ref_s}")

    if dead:
        lo = min(min(m["ep_cutoff"].keys()) for m in (e["metrics"] for e in runs))
        print()
        print("!! never fired at ANY threshold (all metrics are 0, which is NOT a good ERC):")
        for run, h in dead:
            print(f"     {run}  h={h}")
        print(f"   The sweep starts at {lo:.2f}. Either the model is undertrained, or its")
        print( "   probabilities sit below that. Check the max sigmoid output before believing")
        print( "   any row above; lower infer_params.threshold_range[0] in configs/infer.yaml")
        print( "   to see the curve.")

    # ---- the gate -------------------------------------------------------
    print()
    gate = [r for r in all_rows
            if (r["variant"], r["h"]) == GATE and r["dataset"] == "spokenwoz"]
    if not gate:
        print(f"gate row {GATE[0]} h={GATE[1]} not present -- train 'fcall' to evaluate the gate.")
    else:
        r = gate[0]
        ref = PAPER[GATE]
        d = {"MRA": r["MRA"] - ref[0], "HEA": r["HEA"] - ref[1], "ERC": r["ERC"] - ref[3]}
        ok = all(abs(d[k]) <= GATE_TOL[k] for k in d)
        print(f"GATE  {GATE[0]} h={GATE[1]}:  " +
              "  ".join(f"{k} {r[k]:.1f} vs {ref[i]:.1f} (delta {d[k]:+.1f})"
                        for i, k in zip((0, 1, 3), ("MRA", "HEA", "ERC"))))
        print("      => " + ("PASS" if ok else "OUT OF TOLERANCE") +
              f"  (tolerance MRA +-100ms, HEA +-3pp, ERC +-3pp)")
    print("\nNOTE: the paper trained on SpokenWOZ + Switchboard; this run is SpokenWOZ only.")
    print("      Report that alongside any comparison -- a miss here is a condition")
    print("      difference, not necessarily a failed reproduction.")

    if args.csv:
        out = Path(args.csv)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"\nfull sweep ({len(sweep_rows)} rows) -> {out}")


if __name__ == "__main__":
    main()
