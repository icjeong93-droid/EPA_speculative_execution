#!/usr/bin/env python3
"""
Generate path-corrected EPA configs from the upstream templates.

Upstream configs hardcode the authors' cluster paths (/mnt/matylda4/...), so they
cannot be used as-is. This rewrites only the path fields and leaves everything
else — including comments and hyperparameters — byte-identical to upstream, so we
stay honest to the reference implementation and easy to re-sync.

It also emits the fc640 config that upstream is missing but the paper's Table 1
headline rows require.

Usage (local):
  python scripts/make_configs.py --root "C:/Users/user/Desktop/0.WORK/Samsung/EPA"

Usage (HPC):
  python scripts/make_configs.py --root /scratch/$USER/EPA --device cuda

NOTE on config naming: src/utils/common.py builds run_name with
`basename(data_yaml).rstrip(".yaml")`. rstrip strips a CHARACTER SET, not a
suffix, so a data-config name ending in any of . y a m l gets mangled
("spokenwoz_only.yaml" -> "spokenwoz_on"). Our data-config names therefore end
in a digit, which is safe.
"""
import argparse
import re
from pathlib import Path

UPSTREAM = "EndpointAnticipation/anticipation-model/configs"

# data config stem -> which datasets it enables
DATA_CONFIGS = {
    "swoz_v1": ["spokenwoz"],                  # no LDC licence needed
    "swoz_swbd_v1": ["spokenwoz", "switchboard"],  # what the released ckpt used
}


def build_infer_config(datasets, spokenwoz, switchboard, ckpt):
    """
    The evaluation config. Table 1's four metrics (MRA/HEA/PAR/ERC) are computed
    only in ForecastingTrainer.infer_loop, which runs only when cfg.infer_params
    exists -- training itself reports nothing but val accuracy. So without this
    file the HPC run produces checkpoints and no paper numbers.

    src/utils/common.py:load_run resolves the checkpoint as
        join(infer_params.root_path, infer_params.checkpoint_folder, wandb.run_name)
    and run.py's --infer flag supplies run_name, so root_path/checkpoint_folder
    are split from the checkpoint dir here.

    data.datasets here OVERRIDES the training data config (load_run keeps
    infer_datasets), so the raw paths must be right in this file too.
    """
    blocks = []
    if "spokenwoz" in datasets:
        blocks.append(f"""    spokenwoz:
        raw_path: {spokenwoz}
        sr: 8000
        channels:
          separate: true
          preserve: all
          save_path: "{{dump_path}}/spokenwoz_separate_channels/{{fname}}"
""")
    if "switchboard" in datasets:
        blocks.append(f"""    switchboard:
        raw_path: {switchboard}
        sr: 8000
        channels:
          separate: true
          preserve: all
          save_path: "{{dump_path}}/switchboard_separate_channels/{{fname}}"
        filter_out_keyword: FISHER
        filter_keyword: sw
        min_words_per_turn: 4
""")
    ckpt = Path(ckpt)
    return f"""data:
  modes: [test] #keep this fixed -- evaluation runs on the held-out test split

  datasets:
{"".join(blocks)}
infer_params:
  device: cuda
  batch_size: 1          # full-dialogue inference; the loader is not padded
  root_path: {ckpt.parent.as_posix()}
  checkpoint_folder: {ckpt.name}
  infer_single_sample: false
  score_turns: [user]
  threshold_range: [0.05, 0.7, 40]   # swept; report_table1.py picks the operating point
  infer_accuracy_collar_frames: 1
  reset_resample: False
  num_infer_pred_imgs: 1             # per-run trace figure; 0 disables
  infer_checkpoint_name: best_val_acc.pt
  min_turn_length: 1

wandb:
  run_name: PLACEHOLDER   # overridden per run by: run.py --infer <run_name>
  wandb_project:
  use_wandb: False
"""


def build_data_config(datasets, spokenwoz, switchboard, dump):
    blocks = []
    if "spokenwoz" in datasets:
        blocks.append(f"""    spokenwoz:
        raw_path: {spokenwoz}
        sr: 8000
        channels:
          separate: true
          preserve: all
          save_path: "{{dump_path}}/spokenwoz_separate_channels/{{fname}}"
""")
    if "switchboard" in datasets:
        blocks.append(f"""    switchboard:
        raw_path: {switchboard}
        sr: 8000
        channels:
          separate: true
          preserve: all
          save_path: "{{dump_path}}/switchboard_separate_channels/{{fname}}"
        filter_out_keyword: FISHER
        filter_keyword: sw
        min_words_per_turn: 4
""")
    return f"""data:
  modes: [train, val] #keep this fixed

  datasets:
{"".join(blocks)}
  save_paths:
    dump: {dump}
    preprocessed_data_path: "{{dataset}}/preprocessed_{{mode}}.json"
    vad_data_path: "{{dataset}}/vad_processed_{{mode}}.json"
    processed_data_path: "{{dataset}}/processed_{{mode}}.json"
    filtered_data_path: "{{dataset}}/filtered_{{mode}}_context_{{context_in_sec}}_offset_{{extra_offset}}.json"
    resampled_audios_path: "{{dataset}}/resampled_audios_{{mode}}_target_sr_{{target_sr}}"

  override_preprocessed_data: []
  override_vad_data: []
  override_processed_data: []
  override_filtered_data: []

  num_vad_workers: {{num_workers}}
  num_resample_workers: {{num_workers}}

  special_tokens:
    system_end: <system_end>
    user_end: <user_end>
    system: system
    user: user

  text_delim: "_##_"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="EPA working root")
    ap.add_argument("--spokenwoz", default=None, help="SpokenWOZ raw path (default <root>/data/SpokenWOZ)")
    ap.add_argument("--switchboard", default=None, help="Switchboard kaldi data dir (LDC)")
    ap.add_argument("--dump", default=None, help="intermediate dump dir (default <root>/dump)")
    ap.add_argument("--checkpoints", default=None, help="checkpoint dir (default <root>/checkpoints)")
    ap.add_argument("--data-config", default="swoz_v1", choices=list(DATA_CONFIGS),
                    help="which data config the model configs point at")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--out", default=None, help="output configs dir (default <root>/configs)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    spokenwoz = args.spokenwoz or (root / "data" / "SpokenWOZ")
    switchboard = args.switchboard or "/PATH/TO/SWITCHBOARD_KALDI_DATA  # LDC required"
    dump = args.dump or (root / "dump")
    ckpt = args.checkpoints or (root / "checkpoints")
    out = (Path(args.out).resolve() if args.out else (root / "configs"))
    upstream = root / UPSTREAM

    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "forecasting" / "mimi").mkdir(parents=True, exist_ok=True)

    # ---- data configs -------------------------------------------------
    for stem, datasets in DATA_CONFIGS.items():
        text = build_data_config(datasets, spokenwoz, switchboard, dump)
        text = text.replace("{num_workers}", str(args.num_workers))
        (out / "data" / f"{stem}.yaml").write_text(text, encoding="utf-8")
        print(f"  data/{stem}.yaml")

    target_data_cfg = (out / "data" / f"{args.data_config}.yaml").as_posix()

    # ---- model configs: copy upstream, rewrite only the path lines ----
    src_dir = upstream / "forecasting" / "mimi"
    if not src_dir.is_dir():
        raise SystemExit(f"upstream configs not found at {src_dir}")

    made = []
    for src in sorted(src_dir.glob("*.yaml")):
        text = src.read_text(encoding="utf-8")
        text = re.sub(r"^data_config:.*$", f"data_config: {target_data_cfg}", text, flags=re.M)
        text = re.sub(r"^(\s*save_folder:).*$", rf"\1 {Path(ckpt).as_posix()}", text, flags=re.M)
        (out / "forecasting" / "mimi" / src.name).write_text(text, encoding="utf-8")
        made.append(src.name)

    # ---- fc640: upstream is missing it, Table 1 headline needs it -----
    base = (src_dir / "fc960_transformer_mimi_12.5hz_loss1-01_m3.yaml").read_text(encoding="utf-8")
    base = re.sub(r"forecast_intervals_ms:\s*\[960\]", "forecast_intervals_ms: [640]", base)
    base = re.sub(r"^data_config:.*$", f"data_config: {target_data_cfg}", base, flags=re.M)
    base = re.sub(r"^(\s*save_folder:).*$", rf"\1 {Path(ckpt).as_posix()}", base, flags=re.M)
    name640 = "fc640_transformer_mimi_12.5hz_loss1-01_m3.yaml"
    (out / "forecasting" / "mimi" / name640).write_text(base, encoding="utf-8")
    made.append(name640 + "   <- CREATED (missing upstream)")

    # ---- infer config: upstream ships one with the authors' paths baked in ----
    infer_text = build_infer_config(DATA_CONFIGS[args.data_config], spokenwoz, switchboard, ckpt)
    (out / "infer.yaml").write_text(infer_text, encoding="utf-8")
    print("  infer.yaml   <- evaluation (MRA/HEA/PAR/ERC)")

    for m in sorted(made):
        print(f"  forecasting/mimi/{m}")
    print(f"\nconfigs written to {out}")
    print(f"model configs point at: {target_data_cfg}")
    print(f"checkpoints -> {Path(ckpt).as_posix()}")
    print(f"evaluate with : $VENV/bin/python run.py --config {(out / 'infer.yaml').as_posix()} --infer <run_name>")


if __name__ == "__main__":
    main()
