# EPA 재현 — HPC 실행 런북

> **이 문서를 읽는 에이전트에게:** 이것은 설명서가 아니라 **실행 절차서**다.
>
> **먼저 이것부터 실행할 것:**
> ```bash
> bash scripts/preflight.sh /scratch/$USER/EPA
> ```
> 전제 조건·번들·데이터·설치 상태·진행 단계를 한 번에 점검하고, 실패한 항목마다 **정확한 대응 명령**을 출력한다.
> PASS가 나올 때까지 다음으로 넘어가지 말 것. 각 단계 이후에도 다시 돌려 확인할 것.
>
> 그다음 §2부터 순서대로. 막히면 §8(실패 모드 표)을 먼저 보고, §9(금지 사항)는 어떤 경우에도 위반하지 말 것.
>
> **HPC는 인터넷·DNS가 차단돼 있다.** 무엇이든 다운로드를 시도하지 말 것.

---

## 0. 데이터와 패키지는 어디서 오는가 — 중요

**HPC에서는 데이터셋도 패키지도 받을 수 없다.** 전부 인터넷 되는 로컬 머신에서 받아 전송된다.

```
[로컬 - 인터넷 O]                              [HPC - 오프라인]
 download_spokenwoz.py    ──┐
 build_offline_bundle.py  ──┼── transfer    ──►  bundle/ (작업 FS)
                            │                    SpokenWOZ/ (공용 database)
                            │    (rsync)          └─ setup_offline.sh 로 설치
```

`preflight.sh`가 `SpokenWOZ 없음` 또는 `bundle/wheels 없음`을 보고하면, **HPC에서 해결하려 하지 말 것.**
사용자에게 **회사 로컬 머신에서** 아래를 실행해달라고 요청할 것.

### 회사 로컬에서 (인터넷 필요, 1회)

```bash
pip install huggingface_hub                                    # 데이터 스크립트의 유일한 의존성

python scripts/download_spokenwoz.py --root .                  # 12.5GB 압축 → 29GB
python scripts/download_spokenwoz.py --root . --verify          # 4700/1000 wav 확인

# ★ 상류 클론 — 이 repo에 들어있지 않다. 새 머신이면 반드시 먼저 (아래 설명)
git clone https://github.com/bloodraven66/EndpointAnticipation.git
git -C EndpointAnticipation checkout 531e1d7e70751e980b82088a3a358a2e75fa8a12
git -C EndpointAnticipation apply ../patches/epa-num-workers.patch

python scripts/build_offline_bundle.py --root . --out offline_bundle   # wheel+HF, 4.8GB
bash scripts/transfer.sh user@hpc:/scratch/$USER/EPA_incoming \
    --data-dest /home/sr5/SR_AISolution_ACU/database/EPA     # 번들 + 데이터셋
```

`--data-dest` 가 SpokenWOZ를 **작업 디렉터리 밖 공용 database 영역**으로 보낸다 (§4).
빼면 `<remote>/data` 로 가는데, scratch는 15GB 예산이고 데이터셋은 28GB다.
번들만/데이터만 나눠 보내려면 `--bundle-only` / `--data-only`.

**`download_spokenwoz.py`는 단독 실행 파일이다.** 이 repo의 나머지에 의존하지 않으므로
파일 하나만 복사해 가도 된다. 중단되면 그냥 다시 실행하면 이어받는다.

> **`EndpointAnticipation/` 은 이 repo에 포함돼 있지 않다.** 자체 `.git` 을 가진 별도
> 저장소라 `.gitignore` 로 제외돼 있다 — **이 repo를 새로 clone한 머신에는 그 디렉터리가
> 없다.** 위 3줄(clone → 고정 커밋 checkout → 패치 적용)을 건너뛰면 번들 빌드가
> `[3/4] repo + our files` 에서 `지정된 경로를 찾을 수 없습니다` 로 멈춘다.
> `build_offline_bundle.py` 는 이제 **다운로드를 시작하기 전에** 이 세 가지를 확인한다 —
> 클론 존재 / 고정 커밋(`531e1d7e`) 일치 / `epa-num-workers.patch` 적용 여부. 하나라도
> 어긋나면 정확한 복구 명령을 출력하고 멈추므로, 4.8GB를 받은 뒤에 실패하지 않는다.
> (커밋이 다른 채로 굳이 빌드하려면 `--allow-upstream-drift`.)

> **디스크는 29GB가 아니라 ~42GB가 필요하다.** HF 캐시가 압축 아카이브(tar.gz 10.3GB + zip 2.2GB)를
> 해제 후에도 들고 있기 때문이다. `--verify` 통과 후 캐시는 지워도 된다.

사내망 프록시나 미러가 있다면:
```bash
export HTTPS_PROXY=http://proxy.corp:8080
export HF_ENDPOINT=https://your-hf-mirror
```

### 로컬이 Windows인 경우

다운로드 명령은 cmd/PowerShell에서 **그대로 동작한다**. 다만 두 가지가 다르다.

**환경변수** — bash의 `export` 대신:
```powershell
$env:HTTPS_PROXY = "http://proxy.corp:8080"
$env:HF_ENDPOINT = "https://your-hf-mirror"
```

**전송** — `transfer.sh`는 **rsync를 쓰므로 Windows에서 실행할 수 없다**
(Git Bash도 rsync를 포함하지 않는다). 대신 PowerShell 판을 쓴다.

```powershell
.\scripts\transfer.ps1 -Dest user@hpc:/scratch/$env:USERNAME/EPA_incoming `
    -DataDest /home/sr5/SR_AISolution_ACU/database/EPA
```

Windows 10+ 내장 OpenSSH의 `scp`/`ssh`만 사용하며, **보내기 전에 원격에 이미 있는 파일 목록을
받아 없는 것만 전송**하므로 중간에 끊겨도 다시 실행하면 이어진다. 끝나면 원격 파일 수(4700/1000)를
직접 세어 검증한다. `-DataOnly` / `-BundleOnly` 로 나눠 보낼 수도 있다.

`build_offline_bundle.py`는 리눅스용 wheel을 받는다 — **로컬이 Windows여도 무방하다**(`--platform` 지정).
다만 `pip`가 있는 python으로 실행해야 하고, `huggingface_hub`도 필요하다.

전송이 끝나면 HPC로 돌아와 §3부터 진행한다.

---

## 1. 무엇을 하는가

[Endpoint Anticipation for Low-Latency Spoken Dialogue](https://arxiv.org/abs/2606.13450) (Interspeech 2026) 재현.
턴 종료를 최대 2.56초 앞서 예측해 LLM·TTS를 투기 실행하는 모델이며, Unmute 통합에서 지연 1195ms → 690ms를 낭비 연산 +28.4%에 샀다.

**HPC에서의 목표는 하나 — 논문 Table 1을 재현하는 학습.**

| 데이터 | 모델 | h(ms) | MRA(ms) | HEA(%) | PAR(%) | ERC(%) |
|---|---|---|---|---|---|---|
| SpokenWOZ | EPA-S | 640 | 640 | 66.3 | 66.5 | 33.9 |
| SpokenWOZ | **EPA-M** | 640 | **640** | **67.0** | 66.2 | 33.8 |
| SpokenWOZ | EPA-S | 1280 | 1200 | 50.3 | 53.9 | 33.7 |
| SpokenWOZ | EPA-M | 1280 | 1120 | 49.7 | 52.8 | 33.2 |

**관문:** SpokenWOZ EPA-M h=640에서 **HEA 67.0% ± 3%p, MRA 640ms ± 100ms, ERC 33.8% ± 3%p**.

> ⚠️ **논문은 SpokenWOZ + Switchboard로 학습했고 우리는 SpokenWOZ 단독이다** (Switchboard는 LDC 라이선스 미확보).
> 학습 데이터가 다르므로 관문을 못 넘어도 그것은 **재현 실패가 아니라 조건 차이**다.
> 결과를 보고할 때 이 사실을 반드시 함께 적을 것.

**왜 학습이 필요한가:** 공개 체크포인트 `viks66/endpoint-anticipation`의 config는 `forecast_intervals_ms: [960]` —
**h=960 단일 horizon EPA-S**다. Table 1의 h=640·1280 행은 어느 것도 평가만으로는 나오지 않는다.

---

## 2. 사전 조건 확인

```bash
python3.12 --version          # 없으면: module avail python → 3.12 계열 load
ldd --version | head -1       # ★ glibc ≥ 2.28 이어야 함 (아래 참조)
nvidia-smi                    # H100 4장
df -h /home/sr5/SR_AISolution_ACU/database/EPA         # 데이터+dump: >= 150 GB
df -h /scratch                                    # 작업 디렉터리: >= 15 GB
ls bundle/                            # 전송된 번들
ls /home/sr5/SR_AISolution_ACU/database/EPA/SpokenWOZ/   # 데이터셋
```

**검증:** python 3.12.x / **glibc ≥ 2.28** / GPU 4장 / 디스크(위 두 곳) /
`bundle/wheels`·`bundle/hf_cache`·`/home/sr5/SR_AISolution_ACU/database/EPA/SpokenWOZ` 존재. 하나라도 없으면 **중단하고 사용자에게 알릴 것.**

> ✅ **확인됨 — 이 HPC의 glibc는 2.34다.** 번들에서 가장 높은 태그가 `manylinux_2_28`이고
> CUDA wheel 4개(cublas/cudnn/curand/cusolver)가 `manylinux_2_27`이므로 전부 호환된다.
>
> ⚠️ 다른 노드로 옮길 경우: **glibc가 2.28 미만이면 (예: CentOS 7 = 2.17) 설치가 실패한다.** 이 경우 **인터넷 없이는 해결할 수 없다** —
> 번들을 `manylinux2014`(glibc 2.17) 태그로 다시 빌드해야 하므로 사용자에게 알릴 것.
> 확인 명령: `ls bundle/wheels/torch-*.whl` 의 파일명에 찍힌 태그와 `ldd --version` 을 대조.

디스크 내역 — **두 파일시스템에 나뉜다.**
- `/home/sr5/SR_AISolution_ACU/database/EPA` : 원본 28GB + dump ~110GB = **~140GB**
- 작업 디렉터리(`/scratch/$USER/EPA`) : 번들 ~6GB + 체크포인트 런당 ~0.3GB

---

## 3. 설치 (오프라인)

```bash
cd bundle
bash setup_offline.sh /scratch/$USER/EPA
```

표준 `venv` + `pip --no-index --find-links`만 쓴다. `uv`는 필요 없다.

**검증** — 스크립트 끝의 자동 검증이 아래를 모두 출력해야 한다.

```
torch       2.9.1+cu128         ← "+cu"가 없으면 CPU 빌드가 깔린 것 → §8-A
cuda        True NVIDIA H100 ...
arch list   [... 'sm_90' ...]   ← sm_90 필수
silero      OK (bundled ...)
mimi        OK (loaded from HF_HUB_CACHE ...)
audio i/o   OK (torchaudio.load/save round-trip, TorchCodec native)
```

하나라도 빠지면 진행하지 말고 §8을 볼 것.

> **`audio i/o` 줄이 가장 중요하다.** torchaudio 2.9는 `load`/`save`를 전부 TorchCodec으로
> 보내고 없으면 `ImportError`다 — 전처리의 모든 wav가 여기를 지난다. import만 되는지 보는
> 검사로는 잡히지 않아서, 실제로 wav를 쓰고 다시 읽는다. 실패하면 스크립트가 soundfile
> 폴백(`sitecustomize.py`)을 깔고 **새 프로세스에서 다시 검증한 뒤**, 그래도 안 되면 멈춘다.
> `audio i/o   OK (soundfile shim active)` 로 나와도 정상이다 — §8-F.

---

## 4. 데이터 배치

**이 클러스터에서 데이터셋은 작업 디렉터리 밖, 공용 database 영역에 있다.**

```
/home/sr5/SR_AISolution_ACU/database/EPA/SpokenWOZ/
```

`transfer.ps1 -DataDest /home/sr5/SR_AISolution_ACU/database/EPA` 로 보냈다면 이미 여기 있다. **옮기지 말 것.**

```bash
cd /scratch/$USER/EPA
source ./env.sh                 # 모든 실행 전에 반드시 source
```

**검증**

```bash
D=/home/sr5/SR_AISolution_ACU/database/EPA/SpokenWOZ
ls $D/audio_5700_train_dev/*.wav | wc -l   # 4700
ls $D/audio_5700_test/*.wav | wc -l        # 1000
ls $D/text_5700_train_dev/                 # data.json, valListFile.json
```

숫자가 다르면 전송이 불완전한 것이다. rsync를 다시 돌릴 것 (`--partial`이라 이어받는다).

---

## 5. Config 생성

```bash
$VENV/bin/python scripts/make_configs.py --root /scratch/$USER/EPA --num-workers 32 \
    --spokenwoz /home/sr5/SR_AISolution_ACU/database/EPA/SpokenWOZ \
    --dump /home/sr5/SR_AISolution_ACU/database/EPA/dump
```

**검증:** `configs/forecasting/mimi/` 에 fc320·fc640·fc960·fc1280·fc1600·fc1920·fc2240·fc2560·fcall 존재,
`configs/infer.yaml` 존재(평가용, §7.5),
`configs/data/swoz_v1.yaml` 의 `raw_path` 가 `/home/sr5/SR_AISolution_ACU/database/EPA/SpokenWOZ` 를 가리킬 것 — `preflight.sh`가 이 값을 읽어 데이터를 찾는다.

> `fc640`은 상류 리포에 없어서 이 스크립트가 fc960을 복제해 만든다. Table 1의 주력 horizon이므로 반드시 있어야 한다.
> 데이터 config 이름이 숫자로 끝나는 것은 의도적이다 — §8-E 참조.

---

## 6. 전처리 (CPU 잡, 반드시 먼저)

```bash
sbatch scripts/preprocess.sbatch          # train + val — 학습에 필요
sbatch scripts/preprocess.sbatch test     # test 분할 — 평가(§7.5)에 필요, 지금 같이 걸어두면 좋다
```

**이 단계를 건너뛰고 학습을 걸지 말 것.** 전처리는 CPU 바운드로 1~2시간 걸리는데 `run.py` 안에서 실행되므로,
바로 학습을 제출하면 **H100 4장이 그 시간 내내 리샘플링을 기다린다.**

**검증** — 파일 존재 여부를 눈으로 세지 말 것. 판정은 한 곳에 있다.

```bash
$VENV/bin/python scripts/check_preprocessed.py \
    --dump /home/sr5/SR_AISolution_ACU/database/EPA/dump --modes train val
```

```
mode   verdict  dialogues   resampled/expected  missing stages
train  OK            4103          8206/8206    -
val    OK             486           972/972     -
```

`preflight.sh`도 같은 스크립트를 부르므로 둘의 판정이 어긋나지 않는다.
로그 끝에는 `preprocessing complete.` 와 샘플 수가 찍힌다.

> **왜 `ls`로 세면 안 되는가.** `filtered_<mode>_....json` 은 **리샘플링이 시작되기 전에**
> 기록된다(`endpointing_dataset.__init__`: 필터링 → `get_audio_feature` → 리샘플링).
> 리샘플링 도중에 죽으면 파일 이름만으로는 "완료"로 보이고, 다음 GPU 잡이 **H100 4장을 쥔 채
> 리샘플링을 다시 시작한다** — 전처리를 CPU 잡으로 떼어낸 이유가 통째로 무의미해진다.
> 위 스크립트는 상류 `handle_resampling` 과 같은 규칙(대화수 × 채널수)으로 센다.

**중간에 끊겼다면** — `preflight.sh` 또는 위 스크립트가 어느 단계인지 짚어준다.

| verdict | 뜻 | 대응 |
|---|---|---|
| `PARTIAL` + `resampled/expected` 가 모자람 | 리샘플링 중단 | **그냥 재실행.** 개수가 모자라면 다시 돌리고 기존 파일은 덮어쓴다 |
| `PARTIAL`/`MISSING` + missing stages | 앞 단계 중단 | **그냥 재실행.** 산출물이 있는 단계는 건너뛴다 |
| `전처리가 train과 val 사이에서 끊겼다` | processed_train만 있음 | **재실행 전에 지울 것** (§8-I) — 스크립트가 `rm` 명령을 출력한다 |

*참고 — 스모크(40대화) 기준 단계별 세그먼트 수: preprocessed 1358 → vad 1358 → processed 2715 → filtered 2715 (train).
`processed`에서 2배가 되는 것은 user/system 두 채널의 턴을 하나로 병합하기 때문이며 정상이다.*

---

## 7. 학습

```bash
sbatch scripts/train.sbatch fig2      # fc640, fc1280, fcall, fc960, fc2560  (권장)
# 또는
sbatch scripts/train.sbatch table1    # fc640, fc1280, fcall — Table 1 최소 3런
```

`fcall`이 EPA-M(전 horizon 한 모델)이다. GPU 1장당 1런씩 채우고 wave 단위로 대기한다.

**전처리(§6)가 끝나지 않았으면 제출 자체를 거부한다** — 리샘플링 개수까지 세므로 §8-J 상태도
걸린다. GPU를 48시간 잡아놓고 리샘플링부터 시작하는 것을 막는 장치다.

> **`fig2`를 권장하는 이유:** GPU가 4장인데 `table1`은 3런이라 1장이 논다. 그리고
> **`fc960`은 공개 체크포인트와 같은 horizon이라, 우리 학습 결과를 외부 기준과 대조할 수 있는 유일한 지점**이다.

**예상 시간:** 런당 4~6시간 (50 epoch 상한, early stopping patience 6). 로컬 RTX 5080 실측 1.73 s/it 환산.

**검증(진행 중)**

```bash
tail -f logs/train/fc640.log
```

아래가 보이면 정상이다.

```
Number of trainable parameters: 25.19M      ← 논문 "25M streaming Transformer"와 일치해야 함
Mode: train, Loss: 0.xxxx, Acc: 0.xxxx
val epoch: {'val_total': ..., 'val_accuracy': ...}
Saving model at epoch N to .../best_val_acc.pt
```

> **val 곡선이 에폭마다 튀는 것은 정상이다.** validation 창을 매 에폭 무작위로 다시 뽑기
> 때문이다(상류 동작, 부록 A 참조). 다만 `save_best_from_val_acc`와 early stopping이 그 위에서
> 도니, **early stopping이 너무 이르게(예: 10에폭 이전) 걸리면 부록 A의 val 시드 고정을 검토할 것.**
>
> `preflight.sh`가 런별로 `epochs / best_val / @epoch / val range`를 찍는다. **`@epoch`이 0인 채로
> 여러 에폭이 지났으면** 조기 종료가 운 좋은 첫 에폭을 상대로 카운트다운 중이라는 뜻이다.

> ⚠️ **`val_accuracy`가 높다고 좋은 모델이 아니다 — 부록 A-3.** 이건 가중치 없는 프레임 정확도라
> **전부 0으로 예측하면 fc640에서 ~93.8%, fcall에서 ~86%가 나온다.** `best_val_acc.pt`가 그런
> 모델일 수 있고, 그 경우 평가에서 `!! never fired at ANY threshold` 로만 드러난다.
> **첫 런이 끝나는 즉시 그 런 하나만 평가해 발사 여부부터 확인할 것** — 4장을 몇 시간 더 태우기 전에:
>
> ```bash
> sbatch scripts/preprocess.sbatch test     # 아직 안 했다면
> bash scripts/evaluate.sh fc640            # 표에 0만 나오면 학습이 아니라 선택 지표 문제
> ```


**파라미터가 25.19M이 아니면 config가 잘못된 것이다.** 중단하고 §5를 다시 확인할 것.

**검증(끝난 뒤)** — `launch_train.sh`가 마지막에 런별 요약을 찍는다.

```
== summary ==
  fc640    done     epochs=23   checkpoint=ok
  fc1280   done     epochs=31   checkpoint=ok
  fcall    FAILED   epochs=3    checkpoint=ok      ← 체크포인트가 있어도 실패다
```

- **`FAILED`가 하나라도 있으면 스크립트가 non-zero로 끝난다. 평가로 넘어가지 말 것.**
  죽은 런도 직전 에폭의 `best_val_acc.pt`를 남기기 때문에, `evaluate.sh`는 그걸 정상 런과
  구별하지 못하고 **그럴듯한 Table 1 한 줄을 인쇄한다.** `logs/train/<run>.log` 를 먼저 볼 것.
- `epochs`는 `train.json`의 기록 수다. 조기 종료는 마지막 기록 전에 `exit()`하므로 1 적게
  나오는 것이 정상이고, **1~2로 끝났으면 완주가 아니라 사고다.**
- 이미 `best_val_acc.pt`가 있는 런은 **건너뛴다**(`[skip]`으로 표시). 그래서 웨이브 하나가
  깨졌을 때 그냥 다시 제출하면 나머지만 이어서 돈다. 처음부터 다시 학습하려면 `FORCE=1`.
  (상류의 `overwrite_prev_run: true`는 폴더를 비우지 않아서, 그냥 재실행하면 더 좋은 에폭이
  나올 때까지 **옛 체크포인트가 그대로 남는다.**)

---

## 7.5. 평가 — Table 1 뽑기

**학습은 val accuracy만 남긴다.** MRA/HEA/PAR/ERC는 `ForecastingTrainer.infer_loop`에서만 계산되고,
그건 config에 `infer_params`가 있을 때만 — 즉 `configs/infer.yaml`(§5에서 생성됨)로만 돈다.
이 단계를 하지 않으면 체크포인트만 남고 논문과 대조할 숫자는 하나도 나오지 않는다.

```bash
sbatch scripts/preprocess.sbatch test    # 아직 안 했다면 먼저 (CPU 잡)
sbatch scripts/evaluate.sbatch           # 학습된 런 전부 → 추론 → 표 출력
# 특정 런만: bash scripts/evaluate.sh fc640 fcall
```

**출력**

```
variant  h(ms)   thr  MRA(ms)  HEA(%)  PAR(%)  ERC(%)   @ERC   paper (MRA/HEA/PAR/ERC)
EPA-M      640  0.35      640    66.1    66.8    33.9   33.8   640 / 67.0 / 66.2 / 33.8
EPA-M     1280  0.30     1150    50.9    53.1    33.4   33.2   1120 / 49.7 / 52.8 / 33.2

GATE  EPA-M h=640:  MRA ... HEA ... ERC ...   => PASS
```

**읽는 법**

- 논문 Table 1은 곡선이 아니라 **특정 ERC 동작점에서 읽은 한 줄**이다. 스크립트가 threshold 40개
  스윕에서 그 행의 논문 ERC(33.8/33.9/33.7/33.2)에 가장 가까운 threshold를 고르고, 거기서 나머지
  세 지표를 읽는다. `@ERC` 열이 맞춘 목표값이다.
- 전체 곡선은 `logs/eval/threshold_sweep.csv`에 남는다. 다른 동작점에서 보려면
  `--target-erc 20` 또는 `--threshold 0.35`.
- **모든 값이 0이면 성능이 좋은 게 아니라 한 번도 발사하지 않은 것이다.** 스크립트가 경고를 띄운다.
  이때는 threshold 스윕 하한(0.05)보다 모델 출력이 낮은 것이므로, 학습이 덜 됐는지부터 확인할 것.
- `fcall` 런이 EPA-M이다. 관문(§1) 판정은 이 런에서만 나온다.

**test 분할 전처리를 건너뛰면 `evaluate.sh`가 실행을 거부한다** — 그대로 두면 H100을 잡은 채
1000개 대화를 리샘플링하기 때문이다. 판정은 §6과 같은 `check_preprocessed.py`가 하므로
**리샘플링만 덜 끝난 상태도 거부된다**(파일 이름만 보면 완료로 보이는 상태).

`evaluate.sh`는 시작 전에 런마다 두 가지를 더 확인한다.

- 체크포인트 폴더에 yaml이 **정확히 하나** 있을 것 — `load_config`가 그렇게 단언한다.
- 그 yaml의 `data_config:` 경로가 **아직 존재할 것.** 학습 시점의 절대 경로가 박혀 있어서,
  그 사이에 `configs/`를 옮기거나 다른 `--root`로 다시 생성했다면 추론이 assertion으로 죽는다.
  스크립트가 어떤 경로가 사라졌는지 먼저 알려준다.

추론이 실패한 런이 있으면 표를 출력한 뒤 non-zero로 끝난다 — **표가 전부가 아니라는 뜻이다.**

---

## 8. 실패 모드 표

| # | 증상 | 원인 | 대응 |
|---|---|---|---|
| **A** | `torch.__version__`에 `+cu128` 없음 / `cuda False` | **moshi가 `torch<2.10`을 요구해서, moshi를 먼저 설치하면 그 resolver가 cu128 빌드를 CPU 빌드로 교체한다** | `setup_offline.sh`는 torch를 먼저 깐다. 수동 재설치 시에도 **torch → 나머지** 순서를 지킬 것. 번들 wheels에는 cu128만 남겨두었다 |
| **B** | `ModuleNotFoundError: nvidia_*` / CUDA 초기화 실패 | CUDA 런타임 wheel 누락 **또는 win_amd64 wheel 혼입** | `ls bundle/wheels/ \| grep -i nvidia \| grep -c manylinux` 로 16개 확인. 이름만 세면 안 된다 — 빌드 머신이 Windows면 `pip download` 폴백이 win_amd64 wheel을 집어올 수 있고, 그건 여기서 설치되지 않는다. 부족하면 **인터넷 없이 해결 불가하므로 사용자에게 알릴 것** |
| **C** | `TritonMissing` | moshi의 RoPE가 `torch.compile`을 쓴다 | 리눅스에는 triton이 번들에 있어 정상 동작해야 한다. 그래도 나면 `NO_TORCH_COMPILE=1` (성능만 손해) |
| **D** | `AttributeError: Can't get local object 'mimi.<locals>.encode'` | DataLoader 워커가 spawn으로 뜨면 Mimi extractor(지역 클래스)를 pickle 못 한다. **리눅스는 fork라 정상적으로는 안 난다** | `EPA_NUM_WORKERS=0` |
| **E** | `run_name`이 잘려 체크포인트 폴더 이름이 이상함 | `src/utils/common.py`가 `basename(data_yaml).rstrip(".yaml")`을 쓰는데 `rstrip`은 접미사가 아니라 **문자 집합**을 지운다 (`spokenwoz_only.yaml` → `spokenwoz_on`) | 데이터 config 이름을 `. y a m l` 이외 문자로 끝낼 것. 현재 `swoz_v1` |
| **F** | `libtorchcodec_*.so` 로드 실패 / `ModuleNotFoundError: torchcodec` / 전처리 첫 wav에서 사망 | **torchaudio 2.9는 `load`/`save`를 전부 TorchCodec으로 보내고, 없으면 `ImportError`다.** TorchCodec은 런타임에 FFmpeg 공유 라이브러리도 요구한다 | `setup_offline.sh`가 torchcodec을 설치하고 **실제 wav 왕복으로 검증**한 뒤, 실패하면 soundfile 폴백(`scripts/sitecustomize.py`)을 깔고 새 프로세스에서 재검증한다. 수동으로 하려면: `$VENV/bin/pip install --no-index --find-links bundle/wheels torchcodec==0.16.0`, 안 되면 `cp scripts/sitecustomize.py $($VENV/bin/python -c 'import site;print(site.getsitepackages()[0])')/`. **TorchCodec이 정상이면 shim은 아무 것도 하지 않으므로 무해하다** |
| **J** | 전처리를 다시 돌렸는데 학습/추론이 또 리샘플링을 시작함 | `filtered_<mode>_....json` 은 **리샘플링 전에** 기록된다. 이름만 보는 검사는 이 상태를 "완료"로 읽는다 | `$VENV/bin/python scripts/check_preprocessed.py --dump <dump> --modes train val` — `resampled/expected` 를 상류 `handle_resampling` 과 같은 규칙으로 센다. 모자라면 그냥 재실행 |
| **K** | 표에 이상한 숫자가 나오는데 로그는 정상으로 보임 | 죽은 학습 런이 직전 에폭의 `best_val_acc.pt`를 남겼고, 예전 `launch_train.sh`는 자식 종료코드를 무시했다(`wait`는 항상 0을 반환) | 이제 런별로 `wait`하고 `== summary ==` 에 `FAILED`/`epochs`를 찍으며 non-zero로 끝난다. `FAILED`가 있으면 평가하지 말 것 (§7) |
| **M** | `bash preflight.sh` 가 `$'\r': command not found` / `syntax error` 를 쏟거나, `./setup_offline.sh` 가 `bad interpreter` | **CRLF 번들.** git의 `eol=lf` 는 *커밋되는* 내용만 다스리고, Windows 작업 트리는 `core.autocrlf=true` 로 CRLF다. 번들러는 작업 트리를 복사하므로 `#!/usr/bin/env bash\r` 이 그대로 실려 간다 | 번들을 **다시 빌드**할 것 — `build_offline_bundle.py` 가 이제 복사할 때 LF로 변환하고, 남아 있으면 `!! CRLF survived into the bundle` 로 실패시킨다. 급하면 HPC에서: `find bundle -name '*.sh' -o -name '*.sbatch' \| xargs sed -i 's/\r$//'` |
| **L** | 추론이 `AssertionError: Data config file not found` 로 죽음 | 체크포인트 폴더의 yaml에 **학습 시점의 절대 경로**가 박혀 있다. 그 사이 `configs/`를 옮기거나 다른 `--root`로 재생성했다 | `evaluate.sh`가 시작 전에 잡아 어떤 경로인지 알려준다. 같은 경로에 다시 만들거나 그 yaml의 `data_config:` 를 고칠 것 |
| **G** | HF 다운로드를 시도하다 멈춤 | `env.sh`를 source 하지 않음 | `source ./env.sh`. `HF_HUB_OFFLINE=1`이 없으면 DNS 차단 노드에서 timeout까지 매달린다 |
| **H** | wandb가 네트워크를 침 | `use_wandb: false`여도 모듈은 무조건 import된다 | `env.sh`의 `WANDB_MODE=offline`, `WANDB_DISABLED=true` |
| **I** | 전처리 재실행했는데 학습이 `FileNotFoundError: .../processed_val.json` 으로 죽음 | **전처리가 train과 val 사이에서 끊긴 상태.** 상류 `handle_and_add_turns`가 mode 루프 안에서 `continue`가 아니라 `return`을 쓴다 (`src/data/data_processing.py:29`) — `processed_train.json`이 있으면 val을 만들지 않고 함수를 빠져나간다 | **재실행 전에 지울 것.** `rm /home/sr5/SR_AISolution_ACU/database/EPA/dump/spokenwoz/processed_*.json` 후 `sbatch scripts/preprocess.sbatch`. `preflight.sh`가 이 상태를 따로 잡아 이 명령을 출력한다. 다른 단계에서 끊긴 것은 그냥 재실행하면 된다 |

---

## 9. 금지 사항

1. **어떤 것도 다운로드하지 말 것.** `--no-index` 없는 `pip install`, `hf_hub_download`, `git clone`, `wget`, `curl` 전부 금지. 필요한 것은 번들에 있다. 없으면 사용자에게 알릴 것.
2. **DDP를 추가하지 말 것.** 학습 코드는 단일 GPU fp32다. 그런데 EPA-S는 horizon마다 독립 모델이라 작업이 이미 embarrassingly parallel이다. `launch_train.sh`가 GPU를 채운다. DDP는 불필요하고 위험하다.
3. **torch 버전을 바꾸지 말 것.** 2.9.1+cu128은 moshi(`<2.10`)와 H100(sm_90)의 유일한 교집합이다.
4. **상류 코드(`EndpointAnticipation/`)를 수정하지 말 것.** 유일한 예외는 이미 적용된 `repo_local_changes.patch` (`num_workers`를 `EPA_NUM_WORKERS`로 여는 6줄)다. 추가 수정이 필요하면 사용자에게 먼저 물을 것.
5. **하이퍼파라미터를 바꾸지 말 것.** 재현이 목적이다. 상류 값을 그대로 쓴다 — 50 epoch, batch 8, lr 3e-4, patience 6, frame-level BCE (`non_forecast_frames` 0.1 / `forecast_frames` 0.5).
6. **평가 전용 데이터로 학습하지 말 것.** SpokenWOZ test split, `livekit/eot-bench` 등.

---

## 10. 결과 보고

학습이 끝나면 **§7.5를 실행해 표를 만든 뒤** 다음을 보고할 것.

```bash
sbatch scripts/evaluate.sbatch
```

1. 각 런의 **MRA / HEA / PAR / ERC** (SpokenWOZ, horizon별) — `report_table1.py` 출력 그대로
2. §1 관문(EPA-M h=640, HEA 67.0% ± 3%p) 통과 여부
3. **반드시 함께:** "SpokenWOZ 단독 학습이며 논문은 SpokenWOZ + Switchboard 조건" + 부록 A의 논문↔코드 불일치 4건
4. 전처리 후 필터링된 턴 수 (논문과 대조할 첫 숫자)
5. `fc960`을 돌렸다면 공개 체크포인트(h=960)와의 대조 결과

관문을 못 넘었다면 **원인을 규명하기 전까지 다음 단계(부록 C)로 넘어가지 말 것.**
가장 먼저 의심할 것은 학습 데이터 차이(Switchboard 부재), 그다음이 전처리 파라미터다.

---

## 부록 A — 모델 사양 (원문·config 직접 확인)

| 구성요소 | 사양 |
|---|---|
| 음향 front-end | **Mimi neural codec, 첫 8 codebook**, 12.5Hz, **lookahead 0**, 인코더 **frozen**, 24kHz 입력 |
| 예측기 | **25M 스트리밍 Transformer** — 6층, 헤드 4, FFN 1024, RoPE + causal, 좌측 컨텍스트 **240 프레임** (논문 본문은 250) |
| 변형 | **EPA-S**(horizon별 독립 모델) / **EPA-M**(`fcall`, 공유 백본) |
| 투기 실행 | 부분 전사 기반으로 **10 토큰**만 생성 → 오디오를 합성해두되 **재생하지 않음** |
| 확정 시 | 이진 채택/폐기가 아니라 **전체 전사 + 기생성 토큰으로 이어서 생성** |
| 라벨 | Silero VAD로 후행 무음 제거, 2초 미만 턴 마스킹 |

### 논문 본문과 저자 코드가 어긋나는 곳 4건

상류 config와 **저자가 배포한 h=960 체크포인트의 `config.yaml`이 완전히 동일**하다.
즉 아래는 우리가 논문에서 벗어난 것이 아니라, 저자의 실제 실행이 논문 본문과 다른 것이다.

| 항목 | 논문 본문 | 상류 config = 배포 ckpt = 우리 |
|---|---|---|
| batch size | "batch size of 16" | **8** |
| 좌측 컨텍스트 | "fixed 250-frame left context" | **240** (19.2초) |
| 클래스 가중 | "10:1 weighted loss" | **0.5 : 0.1 = 5:1** (파일명은 `loss1-01`인데 값이 다름) |
| 최소 단어수 | "min. of 3 words for Switchboard" | **4** (Switchboard 전용 — 우리는 미사용) |

**배포된 구현 쪽을 따른다.** 재현 대상은 논문 프로즈가 아니라 저자가 실제로 돌린 코드이고,
공개 체크포인트와 대조하려면 같은 설정이어야 한다. **단 결과 보고 시 이 표를 함께 낼 것**(§10).

### 학습 구성에서 알아둘 것 3건 (상류 동작, 고치지 않음)

논문과 어긋나는 것은 아니지만 **결과 해석에 영향을 준다.** 둘 다 저자 코드 그대로이고
배포된 h=960 체크포인트도 같은 조건에서 학습됐다.

**1. 1에폭 ≠ 전체 데이터 1회 순회.**
`__len__`이 대화 개수이고 `__getitem__`은 그 대화에서 **40초 창 하나**만 무작위로 잘라 온다
(`use_random_start: True`). 305초짜리 대화라면 한 에폭에 13%만 본다. train_dev 기준
에폭당 약 52시간 / 원본 205시간. 50에폭이면 여러 번 훑지만 **구간 커버리지는 보장되지 않는다.**

**2. validation 샘플이 에폭마다 바뀐다.**
`use_random_start`는 `src/utils/data_utils.py:119` 한 곳에서만 읽히고 **mode 분기가 없다.**
train과 val이 같은 경로를 타며 `random.choice`에 시드가 없다(`manual_seed`/`random.seed`/
`worker_init_fn`/`generator` 전부 부재). DataLoader의 `shuffle: False`는 순서만 고정할 뿐,
창 위치는 `__getitem__` 안에서 뽑히므로 무관하다. 같은 인덱스를 5번 읽은 실측:

```
1회차  MUL0011   43.6 -  83.6 초
2회차  MUL0011  241.9 - 281.9 초
3회차  MUL0011  171.8 - 211.8 초
```

**영향:** `save_best_from_val_acc: True`(best 체크포인트 선택)와 `early_stopping_patience: 6`이
**에폭마다 다른 데이터로 잰 값** 위에서 동작한다. val 대화가 수백 개라 평균이 어느 정도
상쇄하지만, 에폭 간 비교 가능성은 깨져 있다. 운 좋은 창이 best로 저장되거나 어려운 창이
연달아 나와 조기 종료될 수 있다.

**3. 체크포인트 선택 지표를 "전부 0 예측"이 이긴다.**
`save_best_from_val_acc: True`가 보는 `val_accuracy`는 `fc_base_lstm.py`의 **가중치 없는
프레임 정확도**다. 손실은 5:1로 가중(0.5/0.1)하지만 **선택은 가중하지 않는다.** 라벨이 크게
음성 쪽으로 치우쳐 있어서:

| 설정 | 40초 창의 양성 프레임 | 전부-0 예측기의 `val_accuracy` |
|---|---|---|
| fc640 | ~31 / 500 (6.2%) | **~93.8%** |
| fc1280 | ~62 / 500 (12.4%) | ~87.6% |
| fcall | ~70 / 500 (13.9%) | ~86.1% |

*(스모크 32대화 7019초에서 user 턴 679개 = 0.097 turn/s 로 계산. 2초 미만 턴 마스킹으로
양성이 더 줄므로 실제 상한은 이보다 높다.)*

즉 한 번도 발사하지 않는 모델이 `best_val_acc.pt`로 저장될 수 있고, 그 사실은 **평가 단계의
`!! never fired at ANY threshold` 경고로만 드러난다.** 그래서 §7에 "첫 런이 끝나면 그 런만
먼저 평가하라"를 넣어 두었다.

**판단: 지금은 고치지 않는다.** 재현 기준선(=배포 구현과 동일 조건)을 지키는 값이 더 크다.
**val 곡선이 심하게 요동치거나 early stopping이 납득 안 되는 시점에 걸리면** 그때 아래 최소
수정을 검토할 것 — 상류 수정이 1건에서 2건으로 늘고 배포 체크포인트와 조건이 달라진다.

```python
# src/utils/data_utils.py:119 — val만 대화 ID로 시드 고정 (train은 그대로 무작위)
rng = random.Random(hash(key)) if mode == "val" else random
start_time = rng.choice(turn_start_times)
```
(`mode`를 `convert_continous_labels_to_fixed_context_frames`까지 넘겨야 한다.)


**지표 정의** — MRA: `t_EOT − t_pred` 중앙값(실제 지연 절감) / HEA: 목표 horizon 경계에서의 발사 정밀도 /
PAR: 조기 발사가 1회 이상인 턴의 비율 / ERC: 턴당 최대 대비 실제 조기 발사 비율.

**Switchboard(자유대화)에서는 HEA가 22.1%로 떨어진다** — 과업지향(49.7~67.0%)의 절반 이하.
저자들도 *"currently most viable for structured applications"* 로 적용 범위를 한정했다.

---

## 부록 B — 로컬에서 이미 검증된 것

| 항목 | 결과 |
|---|---|
| 저자 참조 출력 대조 | **anticipation 판정 완전 일치** — 동일 4프레임 `[5.2, 5.28, 5.36, 5.44]`, 확률 최대오차 0.0095 |
| 전처리 4단계 | 정상 (스모크 40대화) |
| 학습 루프 | end-to-end 통과 — 스텝·검증·체크포인트·early stopping |
| 파라미터 수 | **25.19M** (논문 "25M"과 일치) |
| 평가 경로 | **end-to-end 통과** — 추론 → `infer_results.pt` → `report_table1.py` 표 출력 (스모크 3대화) |
| threshold 선택 로직 | 합성 스윕으로 검증 — 목표 ERC 매칭 / `--target-erc` / `--threshold` 3경로 |

`scripts/make_smoke_subset.py`로 40대화 부분집합을 만들면 전체 파이프라인을 수 분 안에 재확인할 수 있다.
GPU 4장을 큐에 걸기 전 점검용으로 쓸 것.

---

## 부록 C — 다음 단계 (HPC 학습 이후)

논문이 비워둔 자리를 채운다: **투기한 내용이 최종 전사와 여전히 맞는지 검증하는 단계.**
논문은 폐기를 "사용자가 계속 말하면"으로만 판단하고 **내용 유효성은 묻지 않는다.**
목표는 PAR 66.2% / ERC 33.8%를 **"진짜 낭비"와 "회수 가능"으로 분해**하는 것.

SpokenWOZ가 필요한 것을 전부 갖고 있다 — `words[i]`에 `{Word, BeginTime, EndTime, ChannelId}` 단어 단위
타임스탬프가 있어 임의 `t_pred`의 부분 전사를 **ASR 없이** 재구성할 수 있고,
턴별 `dialog_act` / `metadata` / `span_info` 로 **LLM 호출 없이** 내용 유효성을 판정할 수 있다.

---

## 부록 D — 디렉터리

```
EPA/                        작업 디렉터리 (데이터셋·dump는 여기 없음 — §4)
├── EndpointAnticipation/   상류 클론 (수정 금지, patch 1건만 적용됨)
├── configs/                make_configs.py 생성물
├── checkpoints/            학습 출력
├── logs/
├── scripts/
├── env.sh                  ★ 모든 실행 전 source
└── requirements-freeze.txt

/home/sr5/SR_AISolution_ACU/database/EPA/
├── SpokenWOZ/             원본 28 GB
└── dump/                  전처리 산출물 ~110 GB (재생성 가능)
```

| 스크립트 | 용도 |
|---|---|
| **`preflight.sh`** | **★ 전제·번들·데이터·설치·진행 상태를 한 번에 점검하고 실패마다 대응 명령 출력. 항상 먼저 실행** |
| `setup_offline.sh` | 오프라인 설치 (번들 안에 있음) |
| `make_configs.py` | 경로 교정 config 생성 + 누락된 fc640 + `infer.yaml` 생성 |
| `preprocess_only.py` / `preprocess.sbatch` | 전처리만 (CPU 잡). `test` 인자로 평가용 분할 |
| **`check_preprocessed.py`** | **전처리 완료 판정의 유일한 기준. 리샘플링 개수까지 센다 (§8-J). preflight·evaluate가 둘 다 이걸 부른다** |
| `evaluate.sh` / `evaluate.sbatch` | **학습 후 추론 → MRA/HEA/PAR/ERC** |
| `report_table1.py` | threshold 스윕을 논문 Table 1 한 줄로 환원 + 관문 판정 |
| `launch_train.sh` / `train.sbatch` | GPU에 런 분배 |
| `make_smoke_subset.py` | 40대화 축소본 (`--test N`으로 평가 경로까지 점검) |
| `sitecustomize.py` | torchaudio I/O 폴백 (§8-F) |
| `build_offline_bundle.py` | **인터넷 되는 곳에서만** — 리눅스 wheel + HF 스냅샷 수집 |
| `transfer.sh` / `transfer.ps1` | 번들+데이터 전송. **Windows는 `.ps1`** (rsync가 없으므로 scp 기반, 재개 가능) |
