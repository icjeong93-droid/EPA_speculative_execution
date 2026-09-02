# EPA 재현 — 현재 상태와 남은 절차

작성: 2026-09-02 · 대상 클러스터: `SR_AISolution_ACU` · 계정: `ic.jeong`

> 이 문서는 **이번 실행의 상태 스냅샷**이다. 절차의 근거와 배경은 `README.md`(런북)에,
> 실패 모드 표는 README §8에 있다. 여기에는 **지금 어디까지 왔고 다음에 무엇을 치는지**만 적는다.
>
> 모든 명령 블록에 실행 위치를 **[HPC]** / **[로컬]** 로 표시했다.

## ⚠️ 이 문서는 수정하지 말 것 — 회사 로컬에서도, HPC에서도

**갱신은 Claude가 저장소(`main`)에서만 한다.** 다른 두 곳의 파일은 사본이다.

- **HPC 사본**은 `scp` 로 덮어써지는 읽기 전용 복사본이다. 거기서 고친 내용은 다음 전송 때 사라진다.
- **로컬 사본**(`D:\VScode\EPA\STATUS.md`)은 `git pull` 로 덮인다. 커밋하지 않은 수정은 충돌하거나 사라진다.

어느 쪽에서 고쳐도 **조용히 유실되고, 그동안 문서와 실제 상태가 어긋난다.** 그것이 이 문서의
존재 이유를 정확히 무너뜨린다.

내용이 틀렸거나 진행이 바뀌었으면 **고치지 말고 Claude에게 말할 것.** 갱신한 뒤 이렇게 받는다:

```powershell
# [로컬]
git pull
scp D:\VScode\EPA\STATUS.md ic.jeong@<HOST>:/home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA/
```

---

## 0. 한 줄 요약

**설치까지 끝났다.** 남은 것은 config 생성 → (MLflow 연결) → 전처리 → 학습 → 평가.
**계산은 전부 HPC**이고, 회사 로컬은 두 곳에서만 등장한다 — MLflow wheel을 받아 보낼 때(4-1b)와
끝나고 결과를 가져와 UI로 볼 때(4-5).

---

## 1. 확정된 경로

| 무엇 | 경로 |
|---|---|
| 작업 디렉터리 (HPC) | `/home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA` |
| 데이터셋 (HPC) | `/home/sr5/SR_AISolution_ACU/database/EPA/SpokenWOZ` |
| dump — 전처리 산출물, ~110GB (HPC) | `/home/sr5/SR_AISolution_ACU/database/EPA/dump` |
| 번들 — 설치 후 삭제 가능 (HPC) | `~/bundle` |
| 로컬 작업 복사본 | `D:\VScode\EPA` |

작업과 데이터가 다른 파일시스템에 나뉘어 있다. **이 분리는 의도된 것이다** — 데이터·dump는
공용 database 영역, venv·체크포인트·로그는 개인 작업 공간.

**[HPC]** 모든 명령 전에:

```bash
cd /home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA
source ./env.sh
```

---

## 2. 어디서 무엇을 하는가

**HPC는 인터넷·DNS가 차단되어 있다.** 그래서 역할이 갈린다.

| | 회사 로컬 (`D:\VScode\EPA`) | HPC |
|---|---|---|
| 성격 | 인터넷 되는 쪽. **바깥에서 뭔가 가져와야 할 때만** | 실제 계산이 도는 쪽. 남은 절차 전부 |
| 하는 일 | 오프라인 번들 빌드, 빠진 wheel 받기, HF 캐시 확보, 파일 전송(`scp`), 결과 회수 | config 생성, 전처리, 학습, 평가 |
| 명령 모양 | `.ps1`, `scp`, `dir`, `python scripts\...` | `bash`, `sbatch`, `ls`, `$VENV/bin/python` |

**구분 규칙:** `.ps1`·`scp`·`dir`이면 로컬, `bash`·`sbatch`·`ls`면 HPC.

### 로컬에서만 할 수 있는 일 (HPC에서 시도하지 말 것)

HPC에서 다운로드를 시도하는 것은 README §9-1 위반이다. 아래는 전부 로컬 몫이다.

```powershell
# 번들 재빌드 (인터넷 필요)
python scripts\build_offline_bundle.py --root . --out offline_bundle

# 빠진 wheel 하나만 받기 — 태그는 PyPI에서 먼저 확인할 것 (§5 교훈)
python -m pip download <패키지> --dest offline_bundle\wheels --no-deps --only-binary=:all: `
    --python-version 3.12 --implementation <cp|py> --abi <cp312|abi3|none> --platform <manylinux 태그>

# HPC로 보내기 (콜론 뒤를 비우면 원격 홈으로 간다)
scp <파일> ic.jeong@<HOST>:bundle/wheels/
```

---

## 3. 완료된 것

- [x] **[로컬]** 오프라인 번들 빌드 및 전송 (4.6GB)
- [x] **[로컬]** `hf-xet`, `bitsandbytes` wheel 수동 보충 — §5 참조
- [x] **[HPC]** `hf_cache` 중첩(`hf_cache/hf_cache`) 해소
- [x] **[HPC]** `setup_offline.sh` 통과 — `torch 2.9.1+cu128` / `cuda True` / `sm_90` / `silero OK` / `mimi OK`
- [x] **[HPC]** 데이터 배치 확인 (train_dev 4700 wav, test 1000 wav)

---

## 4. 남은 절차

### 4-1. Config 생성  ← **지금 여기** · [HPC] · 즉시

```bash
# dump가 database 영역에 ~110GB 생긴다. 공간과 쓰기 권한 먼저.
df -h /home/sr5/SR_AISolution_ACU/database/EPA
touch /home/sr5/SR_AISolution_ACU/database/EPA/.wtest \
  && rm /home/sr5/SR_AISolution_ACU/database/EPA/.wtest \
  && echo "쓰기 가능"

$VENV/bin/python scripts/make_configs.py --root $PWD --num-workers 32 \
    --spokenwoz /home/sr5/SR_AISolution_ACU/database/EPA/SpokenWOZ \
    --dump      /home/sr5/SR_AISolution_ACU/database/EPA/dump
```

**쓰기 권한이 없으면** dump만 작업 공간으로 돌린다 (그 파일시스템에 110GB 여유 필요):

```bash
--dump /home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA/dump
```

**검증**

```bash
bash scripts/preflight.sh $PWD
```

`=== PASS ===` 가 나올 때까지 다음으로 넘어가지 않는다. 실패 항목마다 대응 명령을 직접 출력한다.
`configs/forecasting/mimi/` 에 fc320·fc640·fc960·fc1280·fc1600·fc1920·fc2240·fc2560·fcall,
`configs/infer.yaml`(평가용)이 있어야 한다.

---

### 4-1b. MLflow 연결 · 1회, 학습 전에 · [로컬] + [HPC]

상류에는 쓸 수 있는 실험 추적기가 없다. 유일한 것이 wandb인데 모드가 `"online"` 으로
하드코딩돼 있어 네트워크 없는 노드에서는 켤 수가 없다. 그래서 **MLflow를 붙였다** —
`patches/epa-mlflow-logger.patch` 가 같은 로거 객체에 MLflow 기록을 추가한다.
백엔드는 **파일 스토어**(`logs/mlruns`)라 서버도 네트워크도 필요 없다.

**[로컬] ① mlflow-skinny wheel 받아서 보내기** (UI는 노트북에서 돌리므로 skinny로 충분)

```powershell
cd D:\VScode\EPA
git pull
mkdir mlflow_wheels
python -m pip download mlflow-skinny --dest mlflow_wheels --only-binary=:all: `
    --python-version 3.12 --implementation cp --abi cp312 --abi abi3 `
    --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 `
    --platform manylinux_2_24_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64

scp mlflow_wheels\*.whl                  ic.jeong@<HOST>:bundle/wheels/
scp patches\epa-mlflow-logger.patch      ic.jeong@<HOST>:/home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA/
scp scripts\make_configs.py              ic.jeong@<HOST>:/home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA/scripts/
```

**[HPC] ② 설치 · 패치 적용 · 환경변수**

```bash
cd /home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA && source ./env.sh

$VENV/bin/python -m pip install --no-index --find-links ~/bundle/wheels mlflow-skinny
$VENV/bin/python -c "import mlflow; print(mlflow.__version__)"

cd EndpointAnticipation && patch -p1 < ../epa-mlflow-logger.patch && cd ..
grep -c mlflow EndpointAnticipation/anticipation-model/src/utils/wandb_logger.py   # 0 이 아니어야 함

echo 'export MLFLOW_ALLOW_FILE_STORE=true' >> ./env.sh && source ./env.sh
```

`MLFLOW_ALLOW_FILE_STORE` 는 MLflow ≥ 3.5 가 파일 스토어를 거부하기 때문에 필요하다.
파일 스토어를 쓰는 이유는 **런 5개가 동시에 기록**하기 때문이다 — 파일 스토어는 런마다
디렉터리가 갈리지만 sqlite 단일 파일은 직렬화되며 `database is locked` 를 낸다.

**③ config 재생성** — 4-1을 이미 했더라도 `mlflow:` 블록을 넣기 위해 다시 돌린다.
명령은 4-1과 같다. 생성된 config 끝에 이것이 붙어야 한다:

```yaml
mlflow:
  use_mlflow: True
  tracking_uri: file:///home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA/logs/mlruns
  experiment_name: "EPA reproduction (SpokenWOZ)"
```

> **로깅 실패는 학습을 죽이지 않는다.** MLflow 호출은 전부 try/except 안에 있고,
> mlflow가 없거나 3회 이상 실패하면 한 줄 찍고 조용히 꺼진 채 학습은 계속된다.
> 즉 이 단계를 건너뛰어도 4-3은 그대로 돌아간다 — 지표가 텍스트 로그에만 남을 뿐이다.

---

### 4-2. 전처리 · [HPC] · CPU 잡, 1~2시간

```bash
sbatch scripts/preprocess.sbatch          # train + val — 학습에 필요
sbatch scripts/preprocess.sbatch test     # test 분할 — 평가에 필요, 지금 같이 걸어둘 것
```

**이 단계를 건너뛰고 학습을 제출하지 말 것.** 전처리는 `run.py` 안에서도 돌기 때문에,
바로 학습을 걸면 H100 4장이 리샘플링이 끝날 때까지 논다.

**검증**

```bash
bash scripts/preflight.sh $PWD          # 리샘플 파일 개수까지 세어 완료를 판정한다
du -sh /home/sr5/SR_AISolution_ACU/database/EPA/dump    # ~110GB
```

`filtered_*.json` 이 있다고 완료가 아니다 — 그 파일은 리샘플링 **시작 전에** 쓰인다.
`check_preprocessed.py` 가 그 구분을 하고, preflight·evaluate·학습 제출이 모두 그것을 기준으로 삼는다.

**중간에 끊겼다면** preflight가 두 경우를 구분해 알려준다.

- `[WARN] 전처리 미완` → **그냥 재실행.** 완료된 단계는 건너뛴다.
- `[FAIL] train과 val 사이에서 끊김` → **지우고 재실행.**
  `rm /home/sr5/SR_AISolution_ACU/database/EPA/dump/spokenwoz/processed_*.json`

---

### 4-3. 학습 · [HPC] · GPU 잡, 8~12시간

```bash
sbatch scripts/train.sbatch fig2      # fc640, fc1280, fcall, fc960, fc2560 (권장)
```

`fcall` 이 EPA-M(전 horizon 한 모델)이다. GPU 1장당 1런씩 채우고 wave 단위로 대기한다.
`fig2` 를 쓰는 이유: GPU가 4장인데 `table1`(3런)은 1장이 놀고, `fc960` 은 공개 체크포인트와
같은 horizon이라 외부 대조가 가능한 유일한 런이다.

**진행 확인**

```bash
tail -f logs/train/fc640.log
```

```
Number of trainable parameters: 25.19M      ← 다르면 config가 잘못된 것. 중단하고 4-1 재확인
Mode: train, Loss: 0.xxxx, Acc: 0.xxxx
val epoch: {'val_total': ..., 'val_accuracy': ...}
Saving model at epoch N to .../best_val_acc.pt
```

**val 곡선이 에폭마다 튀는 것은 정상이다** — 상류가 validation 창을 매 에폭 무작위로 다시 뽑는다.
다만 `save_best_from_val_acc` 와 early stopping이 그 위에서 도니, **조기 종료가 10에폭 이전에
걸리면** README 부록 A의 val 시드 고정을 검토할 것.

재제출은 안전하다 — `best_val_acc.pt` 가 이미 있는 런은 건너뛰고 요약에 이름이 찍힌다
(강제 재학습은 `FORCE=1`).

**MLflow로 보기** (4-1b를 했다면). 런이 시작되면 바로 쌓인다:

```bash
ls logs/mlruns/                                   # 실험 디렉터리가 생겼는지
find logs/mlruns -name "*.yaml" -path "*meta*" | wc -l    # 런 개수 = 제출한 런 수
```

기록되는 것:

| 종류 | 이름 |
|---|---|
| 지표 (에폭마다) | `train/total`, `train/accuracy`, `val/total`, `val/accuracy` |
| 파라미터 | config 전체를 평탄화한 값 + `num_params` |
| 아티팩트 | val 예측 시각화 `pred_epochNNN.png` |

런 이름은 config 이름(`fc640_...`)이라 UI에서 horizon별로 바로 갈린다.
**UI는 HPC에서 띄우지 않는다** — 브라우저도 서버도 없다. §4-5에서 로컬로 가져와서 본다.

---

### 4-4. 평가 — Table 1 뽑기 · [HPC] · 1~2시간

```bash
sbatch scripts/evaluate.sbatch           # 학습된 런 전부 → 추론 → 표 출력
# 특정 런만: bash scripts/evaluate.sh fc640 fcall
```

**이 단계를 해야 숫자가 나온다.** 학습은 val accuracy만 남기고, MRA/HEA/PAR/ERC는
`infer_params` 가 있는 config(=`configs/infer.yaml`)로 도는 추론 경로에서만 계산된다.
여기까지 안 하면 체크포인트만 남고 논문과 대조할 값은 하나도 없다.

---

### 4-5. 결과 회수 · [로컬] · 마지막

표와 로그만 가져오면 된다. 체크포인트는 HPC에 두는 편이 낫다(런당 ~0.3GB).

```powershell
$W = "ic.jeong@<HOST>:/home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA"
scp -r "${W}/logs/eval"      .\results\
scp    "${W}/logs/train/*.log" .\results\
scp -r "${W}/logs/mlruns"    .\results\      # MLflow 런 (4-1b를 했다면)
```

**MLflow UI는 여기서 띄운다** (HPC에는 브라우저가 없다):

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri "file:///$($PWD.Path -replace '\\','/')/results/mlruns"
```

학습이 도는 중에도 `logs/mlruns` 만 다시 받아오면 그 시점까지의 곡선을 볼 수 있다.
UI를 띄우려면 로컬에 **full mlflow**가 필요하다(HPC에 깐 skinny는 기록 전용).

---

## 5. 결과를 보고할 때

관문: **SpokenWOZ EPA-M h=640에서 HEA 67.0% ± 3%p, MRA 640ms ± 100ms, ERC 33.8% ± 3%p.**

반드시 함께 적을 것:

1. **학습 데이터가 논문과 다르다** — 논문은 SpokenWOZ + Switchboard, 우리는 SpokenWOZ 단독
   (Switchboard는 LDC 라이선스 미확보). **관문 미달은 재현 실패가 아니라 조건 차이다.**
2. 논문 본문과 저자 코드가 어긋나는 4건(배포 구현을 따랐음) — README 부록 A.

---

## 6. 이번 설치에서 걸렸던 것 (재발 시 대응)

| 증상 | 어디서 | 원인 | 대응 |
|---|---|---|---|
| `No matching distribution found for hf-xet` / `bitsandbytes` | HPC 설치 중 | pip은 환경 마커를 **빌드 호스트** 기준으로 평가한다. Windows에서 번들을 만들면 `sys_platform == "linux"` / `platform_machine == "x86_64"` 로 걸린 의존성이 통째로 빠진다 | **[로컬]** 해당 wheel만 받아 **[HPC]** `bundle/wheels/` 에 넣고 `setup_offline.sh` 재실행. 빌드 스크립트에 검사가 추가되어 이제는 빌드 시점에 잡힌다(`linux_only_gaps`) |
| `OSError: We couldn't connect to https://huggingface.co ... couldn't find them in the cached files` | HPC 검증 중 | `hf_cache/hf_cache` 중첩. `scp -r hf_cache remote:bundle/` 인데 원격에 이미 `bundle/hf_cache` 가 있으면 그 **안으로** 들어간다 | **[HPC]** `mv $H/hf_cache/* $H/ && rmdir $H/hf_cache` 를 번들과 대상 양쪽에 |
| `mkdir: Permission denied` | 로컬에서 전송 시 | 존재하지 않는 상위 디렉터리를 만들려 한 것 (경로 오타) | **[HPC]** `pwd` 로 실제 경로 확인. scp는 `host:` 뒤를 비우면 원격 홈으로 간다 |
| `transfer.ps1` 이 매번 비밀번호를 물음 | 로컬 | 스크립트가 ssh를 30회 이상 호출한다. Windows OpenSSH는 접속 재사용(ControlMaster) 미지원 | **[로컬]** 공개키를 `~/.ssh/authorized_keys` 에 등록. 홈이 그룹 쓰기 가능이면 sshd가 키를 무시하므로 **[HPC]** `chmod go-w ~` 도 함께 |
| 파일이 탐색기에는 보이는데 `find` 는 없다고 함 | — | 로컬 셸과 HPC 셸을 혼동했거나, 변수(`$T` 등)가 그 셸에 없어 경로가 `/hf_cache/...` 로 평가된 것 | `hostname` 으로 어느 기계인지 먼저 확인. 변수 대신 전체 경로를 쓸 것 |

**교훈 하나** — wheel을 수동으로 받을 때는 **플랫폼·ABI 태그를 PyPI에서 먼저 확인**할 것.
번들의 다른 wheel과 같은 태그일 것이라 짐작하면 틀린다. 실제로 `hf-xet` 은 `cp38-abi3` +
`manylinux_2_17`, `bitsandbytes` 는 `py3-none` + `manylinux_2_24` 로 서로 달랐다.

---

## 7. 참고

- 절차의 근거·배경: `README.md`
- 실패 모드 표: `README.md` §8
- 금지 사항(어떤 경우에도 위반 금지): `README.md` §9
- 상류 동작 중 고치지 않은 것들과 그 이유: `patches/README.md`
