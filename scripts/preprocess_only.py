#!/usr/bin/env python3
"""
Run ONLY the data pipeline (channel split -> resample -> Silero VAD -> filter)
and exit, so it can be submitted as a CPU job.

Preprocessing SpokenWOZ is CPU-bound and takes 1-2 h. It currently happens inside
run.py, which means a GPU job sits on 4x H100 doing resampling. Doing it once here
populates the dump; every later training run then hits the cache and starts on the
GPU immediately.

  python scripts/preprocess_only.py --config configs/forecasting/mimi/fc640_....yaml

Forces device=cpu: the Mimi extractor is constructed during setup but is only used
inside __getitem__, so preprocessing never needs the GPU.
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="any model config; only its data_config matters")
    ap.add_argument("--model-dir", default=None, help="path to anticipation-model (default: sibling of this repo)")
    ap.add_argument("--modes", nargs="+", default=None, choices=["train", "val", "test"],
                    help="override cfg.data.modes. The data config fixes [train, val]; pass "
                         "'test' to prepare the evaluation split ahead of time, so inference "
                         "does not start by resampling 1000 dialogues on a GPU node.")
    args = ap.parse_args()

    # resolve BEFORE chdir: a relative --config (what preprocess.sbatch passes from
    # the submit dir) would otherwise be looked up under the model dir and fail.
    config = Path(args.config).resolve()
    if not config.is_file():
        raise SystemExit(f"config not found: {config}")

    model_dir = Path(args.model_dir) if args.model_dir else \
        Path(__file__).resolve().parent.parent / "EndpointAnticipation" / "anticipation-model"
    if not (model_dir / "src").is_dir():
        raise SystemExit(f"anticipation-model not found at {model_dir}")
    os.chdir(model_dir)
    sys.path.insert(0, str(model_dir))

    from src.utils.common import load_config, load_run
    from src.data import load_data

    cfg = load_config([str(config)])
    cfg.run_params.device = "cpu"
    if args.modes:
        cfg.data.modes = list(args.modes)
    print(f"run_name : {cfg.run_name}")
    print(f"dump     : {cfg.data.save_paths.dump}")
    print(f"datasets : {list(cfg.data.datasets.keys())}")
    print(f"modes    : {list(cfg.data.modes)}")
    print("device   : cpu (preprocessing only)\n")

    _model, cfg, _trainer, feat_extractor = load_run(cfg)
    loaders = load_data(cfg, feat_extractor)

    print("\npreprocessing complete.")
    for mode, ld in loaders.items():
        if mode == "test":
            for name, l in ld:
                print(f"  test/{name:12s} {len(l.dataset):7d} samples")
        else:
            print(f"  {mode:17s} {len(ld.dataset):7d} samples")
    print("\ncached under the dump path above; training runs will now skip straight to the GPU.")


if __name__ == "__main__":
    main()
