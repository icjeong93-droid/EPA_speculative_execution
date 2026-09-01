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
 build_offline_bundle.py  ──┼── transfer.sh ──►  bundle/ + data/SpokenWOZ/
                            │    (rsync)          └─ setup_offline.sh 로 설치
```

`preflight.sh`가 `data/SpokenWOZ 없음` 또는 `bundle/wheels 없음`을 보고하면, **HPC에서 해결하려 하지 말 것.**
사용자에게 **회사 로컬 머신에서** 아래를 실행해달라고 요청할 것.

### 회사 로컬에서 (인터넷 필요, 1회)

```bash
pip install huggingface_hub                                    # 데이터 스크립트의 유일한 의존성

python scripts/download_spokenwoz.py --root .                  # 12.5GB 압축 → 29GB
python scripts/download_spokenwoz.py --root . --verify          # 4700/1000 wav 확인

python scripts/build_offline_bundle.py --root . --out offline_bundle   # wheel+HF, 4.8GB
bash scripts/transfer.sh user@hpc:/scratch/$USER/EPA_incoming   # 총 34GB, rsync --partial
```

**`download_spokenwoz.py`는 단독 실행 파일이다.** 이 repo의 나머지에 의존하지 않으므로
파일 하나만 복사해 가도 된다. 중단되면 그냥 다시 실행하면 이어받는다.

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
.\scripts\transfer.ps1 -Dest user@hpc:/scratch/$env:USERNAME/EPA_incoming
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
df -h /scratch                # ≥ 200 GB 여유
ls bundle/ data/SpokenWOZ/    # 전송된 번들과 데이터셋
```

**검증:** python 3.12.x / **glibc ≥ 2.28** / GPU 4장 / scratch ≥200GB /
`bundle/wheels`·`bundle/hf_cache`·`data/SpokenWOZ` 존재. 하나라도 없으면 **중단하고 사용자에게 알릴 것.**

> ✅ **확인됨 — 이 HPC의 glibc는 2.34다.** 번들에서 가장 높은 태그가 `manylinux_2_28`이고
> CUDA wheel 4개(cublas/cudnn/curand/cusolver)가 `manylinux_2_27`이므로 전부 호환된다.
>
> ⚠️ 다른 노드로 옮길 경우: **glibc가 2.28 미만이면 (예: CentOS 7 = 2.17) 설치가 실패한다.** 이 경우 **인터넷 없이는 해결할 수 없다** —
> 번들을 `manylinux2014`(glibc 2.17) 태그로 다시 빌드해야 하므로 사용자에게 알릴 것.
> 확인 명령: `ls bundle/wheels/torch-*.whl` 의 파일명에 찍힌 태그와 `ldd --version` 을 대조.

디스크 내역: 원본 28GB + 전처리 dump ~130GB + 번들 ~6GB + 체크포인트 런당 ~0.3GB.

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
```

하나라도 빠지면 진행하지 말고 §8을 볼 것.

---

## 4. 데이터 배치

```bash
mkdir -p /scratch/$USER/EPA/data
mv data/SpokenWOZ /scratch/$USER/EPA/data/
cd /scratch/$USER/EPA
source ./env.sh                 # 모든 실행 전에 반드시 source
```

**검증**

```bash
ls data/SpokenWOZ/audio_5700_train_dev/*.wav | wc -l   # 4700
ls data/SpokenWOZ/audio_5700_test/*.wav | wc -l        # 1000
ls data/SpokenWOZ/text_5700_train_dev/                 # data.json, valListFile.json
```

숫자가 다르면 전송이 불완전한 것이다. rsync를 다시 돌릴 것 (`--partial`이라 이어받는다).

---

## 5. Config 생성

```bash
$VENV/bin/python scripts/make_configs.py --root /scratch/$USER/EPA --num-workers 32
```

**검증:** `configs/forecasting/mimi/` 에 fc320·fc640·fc960·fc1280·fc1600·fc1920·fc2240·fc2560·fcall 존재,
`configs/infer.yaml` 존재(평가용, §7.5),
`configs/data/swoz_v1.yaml` 의 `raw_path` 가 실제 데이터 경로를 가리킬 것.

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

**검증**

```bash
ls dump/spokenwoz/*.json     # preprocessed / vad_processed / processed / filtered × {train,val}
du -sh dump                  # ~130 GB
```

로그 끝에 `preprocessing complete.` 와 샘플 수가 찍힌다.

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

**파라미터가 25.19M이 아니면 config가 잘못된 것이다.** 중단하고 §5를 다시 확인할 것.

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
1000개 대화를 리샘플링하기 때문이다.

---

## 8. 실패 모드 표

| # | 증상 | 원인 | 대응 |
|---|---|---|---|
| **A** | `torch.__version__`에 `+cu128` 없음 / `cuda False` | **moshi가 `torch<2.10`을 요구해서, moshi를 먼저 설치하면 그 resolver가 cu128 빌드를 CPU 빌드로 교체한다** | `setup_offline.sh`는 torch를 먼저 깐다. 수동 재설치 시에도 **torch → 나머지** 순서를 지킬 것. 번들 wheels에는 cu128만 남겨두었다 |
| **B** | `ModuleNotFoundError: nvidia_*` / CUDA 초기화 실패 | CUDA 런타임 wheel 누락 **또는 win_amd64 wheel 혼입** | `ls bundle/wheels/ \| grep -i nvidia \| grep -c manylinux` 로 16개 확인. 이름만 세면 안 된다 — 빌드 머신이 Windows면 `pip download` 폴백이 win_amd64 wheel을 집어올 수 있고, 그건 여기서 설치되지 않는다. 부족하면 **인터넷 없이 해결 불가하므로 사용자에게 알릴 것** |
| **C** | `TritonMissing` | moshi의 RoPE가 `torch.compile`을 쓴다 | 리눅스에는 triton이 번들에 있어 정상 동작해야 한다. 그래도 나면 `NO_TORCH_COMPILE=1` (성능만 손해) |
| **D** | `AttributeError: Can't get local object 'mimi.<locals>.encode'` | DataLoader 워커가 spawn으로 뜨면 Mimi extractor(지역 클래스)를 pickle 못 한다. **리눅스는 fork라 정상적으로는 안 난다** | `EPA_NUM_WORKERS=0` |
| **E** | `run_name`이 잘려 체크포인트 폴더 이름이 이상함 | `src/utils/common.py`가 `basename(data_yaml).rstrip(".yaml")`을 쓰는데 `rstrip`은 접미사가 아니라 **문자 집합**을 지운다 (`spokenwoz_only.yaml` → `spokenwoz_on`) | 데이터 config 이름을 `. y a m l` 이외 문자로 끝낼 것. 현재 `swoz_v1` |
| **F** | `libtorchcodec_*.so` 로드 실패 | torchaudio 2.9는 모든 I/O를 TorchCodec으로 보낸다 | `scripts/sitecustomize.py`를 venv의 site-packages에 복사 (soundfile로 대체). **TorchCodec이 정상이면 아무 것도 하지 않으므로 무해하다** |
| **G** | HF 다운로드를 시도하다 멈춤 | `env.sh`를 source 하지 않음 | `source ./env.sh`. `HF_HUB_OFFLINE=1`이 없으면 DNS 차단 노드에서 timeout까지 매달린다 |
| **H** | wandb가 네트워크를 침 | `use_wandb: false`여도 모듈은 무조건 import된다 | `env.sh`의 `WANDB_MODE=offline`, `WANDB_DISABLED=true` |

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
EPA/
├── EndpointAnticipation/   상류 클론 (수정 금지, patch 1건만 적용됨)
├── configs/                make_configs.py 생성물
├── data/SpokenWOZ/         원본 (28 GB)
├── dump/                   전처리 산출물 (~130 GB)
├── checkpoints/            학습 출력
├── logs/
├── scripts/
├── env.sh                  ★ 모든 실행 전 source
└── requirements-freeze.txt
```

| 스크립트 | 용도 |
|---|---|
| **`preflight.sh`** | **★ 전제·번들·데이터·설치·진행 상태를 한 번에 점검하고 실패마다 대응 명령 출력. 항상 먼저 실행** |
| `setup_offline.sh` | 오프라인 설치 (번들 안에 있음) |
| `make_configs.py` | 경로 교정 config 생성 + 누락된 fc640 + `infer.yaml` 생성 |
| `preprocess_only.py` / `preprocess.sbatch` | 전처리만 (CPU 잡). `test` 인자로 평가용 분할 |
| `evaluate.sh` / `evaluate.sbatch` | **학습 후 추론 → MRA/HEA/PAR/ERC** |
| `report_table1.py` | threshold 스윕을 논문 Table 1 한 줄로 환원 + 관문 판정 |
| `launch_train.sh` / `train.sbatch` | GPU에 런 분배 |
| `make_smoke_subset.py` | 40대화 축소본 (`--test N`으로 평가 경로까지 점검) |
| `sitecustomize.py` | torchaudio I/O 폴백 (§8-F) |
| `build_offline_bundle.py` | **인터넷 되는 곳에서만** — 리눅스 wheel + HF 스냅샷 수집 |
| `transfer.sh` / `transfer.ps1` | 번들+데이터 전송. **Windows는 `.ps1`** (rsync가 없으므로 scp 기반, 재개 가능) |
