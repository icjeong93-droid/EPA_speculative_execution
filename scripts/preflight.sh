#!/usr/bin/env bash
# Preflight for the offline HPC. Run this FIRST, and after each major step.
# It never touches the network and never modifies anything.
#
#   bash scripts/preflight.sh [WORKDIR]        # default: current directory
#
# Prints one line per check: [ OK ] / [FAIL] / [ -- ] (not applicable yet),
# then a REMEDY block for anything that failed. Exit 0 only if nothing failed.

WORK="${1:-$PWD}"
SELFDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# any python is enough for the stdlib-only helpers; prefer the venv when it exists
ANYPY="$WORK/.venv/bin/python"
[ -x "$ANYPY" ] || ANYPY="$(command -v python3 2>/dev/null)"
[ -n "$ANYPY" ] || ANYPY="$(command -v python 2>/dev/null)"
# on the HPC the bundle sits next to the workdir; on the build machine it is
# offline_bundle/ inside it. try both so this script works in either place.
BUNDLE="${BUNDLE:-}"
if [ -z "$BUNDLE" ]; then
  for c in "$WORK/../bundle" "$WORK/bundle" "$WORK/offline_bundle"; do
    [ -d "$c/wheels" ] && { BUNDLE="$c"; break; }
  done
  [ -n "$BUNDLE" ] || BUNDLE="$WORK/bundle"
fi

fails=(); warns=()
ok(){   printf '[ OK ] %s\n' "$1"; }
no(){   printf '[FAIL] %s\n' "$1"; fails+=("$2"); }
skip(){ printf '[ -- ] %s\n' "$1"; }
warn(){ printf '[WARN] %s\n' "$1"; warns+=("$2"); }

echo "=== EPA preflight ==="
echo "workdir : $WORK"
echo "bundle  : $BUNDLE"
echo

# ---------------------------------------------------------------- host
echo "-- host --"
if command -v python3.12 >/dev/null 2>&1; then
  ok "python3.12 present ($(python3.12 --version 2>&1))"
else
  no "python3.12 NOT found" "module avail python  → 3.12 계열을 load. 없으면 사용자에게 알릴 것."
fi

GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
if [ -n "$GLIBC" ]; then
  MAJ=${GLIBC%%.*}; MIN=${GLIBC##*.}
  if [ "$MAJ" -gt 2 ] || { [ "$MAJ" -eq 2 ] && [ "$MIN" -ge 28 ]; }; then
    ok "glibc $GLIBC (>= 2.28, torch wheel 태그와 호환)"
  else
    no "glibc $GLIBC < 2.28" "번들의 torch가 manylinux_2_28 태그다. 이 노드에서는 설치 불가.
     인터넷 없이 해결할 수 없으므로 사용자에게 알리고, manylinux2014 태그로 번들 재빌드를 요청할 것."
  fi
else
  warn "glibc 버전을 읽지 못함" "ldd --version 을 직접 확인할 것"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
  NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  [ "$NGPU" -ge 1 ] && ok "GPU ${NGPU}장 ($NAME)" || no "GPU 없음" "GPU 노드에서 실행할 것"
else
  skip "nvidia-smi 없음 (로그인 노드일 수 있음 — 학습은 GPU 노드에서)"
fi

# dump can live on another filesystem (here: the shared database area), so read
# it from the config rather than assuming <workdir>/dump.
DUMP="${EPA_DUMP:-}"
if [ -z "$DUMP" ] && [ -f "$WORK/configs/data/swoz_v1.yaml" ]; then
  DUMP=$(grep -m1 "dump:" "$WORK/configs/data/swoz_v1.yaml" | sed "s/.*dump:[[:space:]]*//; s/[[:space:]]*$//")
fi
[ -n "$DUMP" ] || DUMP="$WORK/dump"

check_space() {  # dir needed_G label
  d="$1"; need="$2"; lbl="$3"
  probe="$d"; while [ ! -d "$probe" ] && [ "$probe" != "/" ]; do probe=$(dirname "$probe"); done
  a=$(df -BG "$probe" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')
  [ -n "$a" ] || return 0
  if [ "$a" -ge "$need" ]; then ok "$lbl ${a}G 여유 (>= ${need}G) — $probe"
  else no "$lbl ${a}G 여유 (${need}G 필요) — $probe" "공간 부족. dump를 옮기려면 make_configs.py --dump <경로> 로 config를 다시 생성할 것."; fi
}
check_space "$DUMP" 120 "dump(전처리)"
check_space "$WORK" 15 "작업 디렉터리(번들+체크포인트)"
echo

# ---------------------------------------------------------------- bundle
echo "-- bundle (오프라인 설치 재료) --"
if [ -d "$BUNDLE/wheels" ]; then
  NW=$(ls "$BUNDLE/wheels"/*.whl 2>/dev/null | wc -l)
  NV=$(ls "$BUNDLE/wheels" 2>/dev/null | grep -i nvidia | grep -c manylinux)
  TR=$(ls "$BUNDLE/wheels" 2>/dev/null | grep -ci '^triton')
  CU=$(ls "$BUNDLE/wheels"/torch-*+cu128*.whl 2>/dev/null | wc -l)
  BAD=$(ls "$BUNDLE/wheels"/torch-*.whl 2>/dev/null | grep -vc cu128)
  [ "$NW" -ge 80 ] && ok "wheel ${NW}개" || no "wheel ${NW}개뿐" "번들 전송이 불완전"
  [ "$NV" -ge 16 ] && ok "nvidia CUDA wheel ${NV}개" \
    || no "nvidia CUDA wheel ${NV}개 (16 필요)" "torch가 CUDA 없이 뜬다. 번들 재빌드 필요 — 사용자에게 알릴 것."
  NONLINUX=$(ls "$BUNDLE/wheels" 2>/dev/null | grep -cE "win_amd64|win32|macosx")
  if [ "$NONLINUX" -eq 0 ]; then ok "전부 리눅스 wheel";
  else no "비-리눅스 wheel ${NONLINUX}개" "빌드 머신(Windows/mac)용 wheel이 섞였다. 이 노드에서는 설치되지 않고 오프라인 복구도 불가 — 사용자에게 번들 재빌드를 요청할 것: ls \$BUNDLE/wheels | grep -E win_amd64\|macosx"; fi
  [ "$TR" -ge 1 ] && ok "triton 있음" || warn "triton 없음" "NO_TORCH_COMPILE=1 로 우회 가능(성능만 손해)"
  # torchaudio 2.9 does every load/save through TorchCodec. A bundle without this
  # wheel installs cleanly and then dies on the first wav of preprocessing (§8-F).
  TC=$(ls "$BUNDLE/wheels"/torchcodec-*.whl 2>/dev/null | wc -l)
  SF=$(ls "$BUNDLE/wheels"/soundfile-*.whl 2>/dev/null | wc -l)
  if [ "$TC" -ge 1 ]; then ok "torchcodec wheel 있음 (오디오 I/O)"
  elif [ "$SF" -ge 1 ]; then warn "torchcodec wheel 없음 (soundfile 폴백만 있음)" "setup_offline.sh 가 sitecustomize.py shim 을 깔아 우회한다. §8-F"
  else no "torchcodec·soundfile wheel 둘 다 없음" "torchaudio.load 가 동작하지 않는다 — 전처리 첫 wav에서 죽는다.
     오프라인으로 복구 불가. 번들 재빌드를 사용자에게 요청할 것 (build_offline_bundle.py 가 이제 둘 다 담는다)."; fi
  [ "$CU" -ge 1 ] && ok "torch cu128 wheel 있음" || no "torch cu128 wheel 없음" "번들 재빌드 필요"
  [ "$BAD" -eq 0 ] && ok "CPU torch 중복 없음" \
    || warn "CPU torch wheel이 섞여 있음" "pip이 잘못 집을 수 있다. cu128 아닌 torch-*.whl 삭제 권장"
else
  no "bundle/wheels 없음" "인터넷 되는 곳에서: python scripts/build_offline_bundle.py --root . --out offline_bundle
     그다음:              bash scripts/transfer.sh user@hpc:/scratch/\$USER/EPA_incoming"
fi

MIMI="$BUNDLE/hf_cache/models--kyutai--mimi"
if [ -d "$MIMI" ]; then
  if find "$MIMI" -name config.json | grep -q .; then
    ok "Mimi 스냅샷 (config.json 확인)"
  else
    no "Mimi 스냅샷에 config.json 없음" "from_pretrained가 실패한다. 스냅샷이 불완전 — 사용자에게 재다운로드 요청."
  fi
else
  no "bundle/hf_cache/models--kyutai--mimi 없음" "학습 필수. 번들 재빌드 필요."
fi

[ -d "$BUNDLE/repo/anticipation-model/src" ] && ok "상류 repo 포함" \
  || no "bundle/repo 없음" "번들 전송이 불완전"
echo

# ---------------------------------------------------------------- dataset
echo "-- dataset --"
# The dataset need not live under the workdir (on this cluster it sits in a shared
# database area). configs/data/swoz_v1.yaml is the authority once make_configs.py
# has run; SPOKENWOZ=<path> overrides for a check before that.
D="${SPOKENWOZ:-}"
if [ -z "$D" ] && [ -f "$WORK/configs/data/swoz_v1.yaml" ]; then
  D=$(grep -m1 "raw_path:" "$WORK/configs/data/swoz_v1.yaml" | sed "s/.*raw_path:[[:space:]]*//; s/[[:space:]]*$//")
fi
if [ -z "$D" ]; then
  D="$WORK/data/SpokenWOZ"
  [ -d "$D" ] || D="$WORK/../data/SpokenWOZ"
fi
echo "경로: $D"
if [ -d "$D" ]; then
  NTR=$(ls "$D/audio_5700_train_dev"/*.wav 2>/dev/null | wc -l)
  NTE=$(ls "$D/audio_5700_test"/*.wav 2>/dev/null | wc -l)
  [ "$NTR" -eq 4700 ] && ok "train_dev wav 4700개" \
    || no "train_dev wav ${NTR}개 (4700 필요)" "rsync 재실행 (--partial 이라 이어받는다)"
  [ "$NTE" -eq 1000 ] && ok "test wav 1000개" || warn "test wav ${NTE}개 (1000 기대)" "평가에만 쓰이므로 학습은 가능"
  [ -f "$D/text_5700_train_dev/data.json" ] && ok "data.json 있음" || no "data.json 없음" "rsync 재실행"
  [ -f "$D/text_5700_train_dev/valListFile.json" ] && ok "valListFile.json 있음" || no "valListFile.json 없음" "rsync 재실행"
else
  no "SpokenWOZ 데이터셋 없음 (찾은 경로: $D)" "★ HPC에서는 다운로드할 수 없다. 인터넷 되는 곳에서 받아 전송해야 한다:
       python scripts/download_spokenwoz.py --root .        (12.5GB 압축 / 29GB 해제)
       bash scripts/transfer.sh user@hpc:/scratch/\$USER/EPA_incoming
     사용자에게 위 두 명령을 로컬에서 실행해달라고 요청할 것."
fi
echo

# ---------------------------------------------------------------- install
echo "-- 설치 상태 --"
PY="$WORK/.venv/bin/python"
if [ -x "$PY" ]; then
  ok "venv 있음"
  OUT=$("$PY" - <<'EOF' 2>&1
try:
    import torch
    v = torch.__version__
    print("TORCH", v)
    print("CU", "+cu" in v)
    print("CUDA_AVAIL", torch.cuda.is_available())
    print("ARCH", ",".join(torch.cuda.get_arch_list()) if torch.cuda.is_available() else "")
except Exception as e:
    print("IMPORTFAIL", type(e).__name__, e)
EOF
)
  if echo "$OUT" | grep -q IMPORTFAIL; then
    no "torch import 실패: $(echo "$OUT" | head -1)" "§8-B. nvidia wheel 누락 가능."
  else
    TV=$(echo "$OUT" | awk '/^TORCH/{print $2}')
    echo "$OUT" | grep -q "^CU True" && ok "torch $TV (cu 빌드)" \
      || no "torch $TV — CPU 빌드" "§8-A. moshi가 cu128을 덮어썼다. torch를 먼저 재설치할 것:
     \$VENV/bin/pip install --no-index --find-links $BUNDLE/wheels --force-reinstall torch==2.9.1 torchaudio==2.9.1"
    echo "$OUT" | grep -q "^CUDA_AVAIL True" && ok "CUDA 사용 가능" || warn "CUDA 사용 불가" "로그인 노드면 정상"
    echo "$OUT" | grep -q "sm_90" && ok "sm_90 (H100) 포함" || warn "sm_90 없음" "H100에서 못 돈다"
  fi
  "$PY" -c "import moshi, transformers, silero_vad, wandb" 2>/dev/null && ok "moshi/transformers/silero/wandb import" \
    || no "의존성 import 실패" "setup_offline.sh 를 다시 실행"

  # Importing modules is not proof that audio works. torchaudio 2.9 sends every
  # load/save through TorchCodec, which also needs FFmpeg shared libs at runtime.
  # Every wav in preprocessing goes through here, so a real round-trip is the only
  # honest check -- otherwise preflight says PASS and the 1-2 h CPU job dies on the
  # first file.
  AIO=$("$PY" - <<'EOF' 2>&1
import os, tempfile
try:
    import torch, torchaudio
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "probe.wav")
        torchaudio.save(p, torch.zeros(8000), 8000)   # 1-D, as handle_channels does
        b, sr = torchaudio.load(p, frame_offset=100, num_frames=4000)
        assert sr == 8000 and b.shape[-1] == 4000, (sr, tuple(b.shape))
    via = "soundfile shim" if getattr(torchaudio, "_EPA_SOUNDFILE_SHIM", False) else "torchcodec"
    print("AUDIO_OK", via)
except Exception as e:
    print("AUDIO_FAIL", type(e).__name__, e)
EOF
)
  case "$AIO" in
    AUDIO_OK*) ok "오디오 I/O 왕복 정상 (${AIO#AUDIO_OK })" ;;
    *) no "torchaudio.load/save 실패: $AIO" "★ 전처리의 모든 wav 읽기·쓰기가 여기를 지난다. torchaudio 2.9는 TorchCodec으로만 I/O를 한다.
     1) torchcodec 설치 확인: \$VENV/bin/pip install --no-index --find-links $BUNDLE/wheels torchcodec==0.16.0
     2) 그래도 안 되면(FFmpeg 미설치 등) soundfile 폴백을 넣을 것:
        cp $WORK/scripts/sitecustomize.py \$(\$VENV/bin/python -c 'import site;print(site.getsitepackages()[0])')/
     3) 둘 다 불가하면 번들 재빌드가 필요하므로 사용자에게 알릴 것." ;;
  esac
else
  skip "venv 없음 — 아직 설치 전 (README §3)"
fi

if [ -f "$WORK/env.sh" ]; then
  ok "env.sh 있음"
  [ -n "${HF_HUB_OFFLINE:-}" ] && ok "env.sh가 source 됨 (HF_HUB_OFFLINE=$HF_HUB_OFFLINE)" \
    || warn "env.sh를 source 하지 않음" "source $WORK/env.sh — 안 하면 DNS 차단 노드에서 timeout까지 매달린다(§8-G)"
else
  skip "env.sh 없음 — 아직 설치 전"
fi
echo

# ---------------------------------------------------------------- progress
echo "-- 진행 상태 --"
[ -d "$WORK/configs/forecasting/mimi" ] \
  && { N=$(ls "$WORK/configs/forecasting/mimi"/*.yaml 2>/dev/null | wc -l)
       ls "$WORK/configs/forecasting/mimi"/fc640_*.yaml >/dev/null 2>&1 \
         && ok "config ${N}개 (fc640 포함)" || no "fc640 config 없음" "make_configs.py 재실행 (README §5)"; } \
  || skip "config 미생성 (README §5)"

if [ -f "$WORK/configs/infer.yaml" ]; then ok "infer.yaml 있음 (평가용)";
elif [ -d "$WORK/configs/forecasting/mimi" ]; then no "configs/infer.yaml 없음" "평가(MRA/HEA/PAR/ERC)를 돌릴 수 없다. make_configs.py 재실행 (README §5)";
else skip "infer.yaml 미생성 (README §5)"; fi

SW="$DUMP/spokenwoz"
# check_preprocessed.py is the single authority for "is this split really done".
# A filename-only check is not enough: filtered_<mode>_....json is written BEFORE
# resampling starts, so a job killed midway looks complete and the next GPU job
# resamples while holding four H100s.
if [ ! -d "$SW" ]; then
  skip "전처리 전 (README §6)"
elif [ -z "$ANYPY" ]; then
  warn "python이 없어 전처리 상태를 확인하지 못함" "python3 를 load 한 뒤 preflight 를 다시 실행할 것"
else
  PREP=$("$ANYPY" "$SELFDIR/check_preprocessed.py" --dump "$DUMP" --modes train val 2>&1); PREPRC=$?
  echo "$PREP" | sed 's/^/       /'
  if [ "$PREPRC" -eq 0 ]; then
    ok "전처리 완료 train/val ($(du -sh "$DUMP" 2>/dev/null | cut -f1))"
  else
    no "전처리 미완 또는 중단" "위 check_preprocessed.py 출력의 대응 명령을 그대로 실행할 것.
     리샘플링만 덜 끝난 상태는 파일 이름으로는 '완료'로 보이므로 이 검사를 건너뛰지 말 것."
  fi
fi

# test split: only needed before evaluation, so not a hard failure here
if [ -n "$ANYPY" ] && [ -d "$SW" ]; then
  if "$ANYPY" "$SELFDIR/check_preprocessed.py" --dump "$DUMP" --modes test --quiet >/dev/null 2>&1; then
    ok "test 분할 전처리 완료 (평가 준비됨)"
  else
    skip "test 분할 미전처리 — 평가 전에: sbatch scripts/preprocess.sbatch test (README §7.5)"
  fi
fi

if [ -d "$WORK/checkpoints" ] && [ -n "$(find "$WORK/checkpoints" -name 'best_val_acc.pt' 2>/dev/null)" ]; then
  ok "체크포인트 $(find "$WORK/checkpoints" -name 'best_val_acc.pt' 2>/dev/null | wc -l)개"
  if [ -n "$ANYPY" ]; then
    # Surface the two training properties that decide whether a checkpoint is worth
    # evaluating (README 부록 A): epochs actually completed, and how much the val
    # metric moves between epochs. Both come free from train.json.
    TRS=$("$ANYPY" - "$WORK/checkpoints" <<'EOF' 2>&1
import glob, json, os, sys
root = sys.argv[1]
rows = []
for tj in sorted(glob.glob(os.path.join(root, "*", "train.json"))):
    run = os.path.basename(os.path.dirname(tj))
    try:
        with open(tj, encoding="utf-8") as fh:
            r = json.load(fh)
    except Exception:
        continue
    va = [e["val_accuracy"] for e in r if isinstance(e, dict) and "val_accuracy" in e]
    if not va:
        continue
    best = max(va)
    rows.append((run, len(r), best, va.index(best), min(va), max(va)))
if not rows:
    sys.exit(0)
def short(name):
    # the model-config stem at the END is the identifying part, so trim the head
    return name if len(name) <= 44 else "..." + name[-41:]
print("%-44s %6s %8s %7s %s" % ("run", "epochs", "best_val", "@epoch", "val range"))
for run, n, best, bi, lo, hi in rows:
    print("%-44s %6d %8.4f %7d  %.3f..%.3f" % (short(run), n, best, bi, lo, hi))
sus = [r for r in rows if r[1] >= 3 and r[3] == 0]
short = [r for r in rows if r[1] <= 2]
if sus:
    print()
    print("best val accuracy is still epoch 0 for: " + ", ".join(r[0] for r in sus))
    print("early stopping counts every later epoch as 'no improvement', so the run can")
    print("end at patience=6 having saved epoch-0 weights. The val window is re-drawn")
    print("at random every epoch (README appendix A), so this can be pure noise.")
if short:
    print()
    print("only 1-2 epochs recorded for: " + ", ".join(r[0] for r in short))
    print("that is a crashed run, not a finished one -- check logs/train/<run>.log")
sys.exit(2 if (sus or short) else 0)
EOF
)
    TRRC=$?
    [ -n "$TRS" ] && echo "$TRS" | sed 's/^/       /'
    [ "$TRRC" -eq 2 ] && warn "학습 진행 상태에 의심스러운 런이 있음" "위 표를 볼 것. 평가 전에 해당 런의 로그를 확인할 것 (README §7)"
  fi
else
  skip "학습 전 (README §7)"
fi

NEVAL=$(ls "$WORK"/checkpoints/*/infer_results.pt 2>/dev/null | wc -l)
if [ "$NEVAL" -gt 0 ]; then ok "평가 결과 ${NEVAL}개 — 표: scripts/report_table1.py --root $WORK";
else skip "평가 전 (README §7.5)"; fi
echo

# ---------------------------------------------------------------- verdict
if [ ${#fails[@]} -eq 0 ]; then
  echo "=== PASS — 다음 단계로 진행 가능 ==="
  [ ${#warns[@]} -gt 0 ] && { echo; echo "경고:"; for w in "${warns[@]}"; do echo "  - $w"; done; }
  exit 0
else
  echo "=== FAIL (${#fails[@]}건) ==="
  echo
  echo "REMEDY:"
  for f in "${fails[@]}"; do echo "  - $f"; echo; done
  exit 1
fi
