# EPA 재현 환경

[Endpoint Anticipation for Low-Latency Spoken Dialogue](https://arxiv.org/abs/2606.13450) (Interspeech 2026) 재현 및
논문이 비워둔 "투기 내용 유효성 검증" 단계 추가를 위한 작업 환경.

상류 코드: [bloodraven66/EndpointAnticipation](https://github.com/bloodraven66/EndpointAnticipation) (`EndpointAnticipation/`, 클론 그대로 두고 수정하지 않음)

```
EPA/
├── EndpointAnticipation/   상류 클론 (건드리지 않음)
├── configs/                경로 교정된 config (scripts/make_configs.py 생성물)
├── data/SpokenWOZ/         원본 데이터
├── dump/                   전처리 중간 산출물 (VAD/리샘플)
├── checkpoints/            학습 출력
├── logs/                   실행 로그
├── scripts/                셋업·데이터·학습 스크립트 (make_smoke_subset.py 포함)
└── requirements-freeze.txt 검증된 패키지 고정본 (81개)
```

---

## 확인된 사실 — 계획에 영향을 준 것들

**1. 공개 체크포인트는 EPA-M이 아니라 h=960 단일 horizon EPA-S다.**
`viks66/endpoint-anticipation`의 `config.yaml`이 `forecast_intervals_ms: [960]`.
논문 Table 1의 주력 행은 h=640·1280이므로 **평가만으로는 Table 1의 어떤 행도 재현할 수 없다.**
h=960은 Figure 2(latency-tradeoff 곡선)에만 등장한다. → **재현하려면 학습이 필요하다.**
또한 그 체크포인트는 `data_mix_4`(SpokenWOZ + Switchboard)로 학습됐다.

**2. 상류에 `fc640` config가 없다.** Table 1 주력 horizon인데 빠져 있다.
`scripts/make_configs.py`가 fc960을 복제해 `forecast_intervals_ms: [640]`로 생성한다.

**3. 학습 코드는 DDP도 AMP도 없다.** 순수 단일 GPU / fp32.
그런데 **DDP를 붙일 필요가 없다** — EPA-S는 horizon마다 별도 모델이라 작업이 이미
embarrassingly parallel이다. GPU를 채우기만 하면 된다 (`scripts/launch_train.sh`).

**4. `requirements.txt`가 불완전하다.** 아래 셋이 빠져 있고 없으면 실행이 안 된다.
- `silero-vad` — 데이터 전처리의 턴 경계 VAD
- `transformers` — Mimi 로딩 (`MimiModel`)
- `wandb` — `src/utils/wandb_logger.py`가 **`use_wandb: false`여도 최상단에서 무조건 import**

**5. config 이름 함정.** `src/utils/common.py`가 `basename(data_yaml).rstrip(".yaml")`로 `run_name`을 만든다.
`rstrip`은 접미사가 아니라 **문자 집합**을 지우므로 `. y a m l` 로 끝나는 이름은 잘려나간다
(`spokenwoz_only.yaml` → `spokenwoz_on`). 데이터 config 이름은 숫자로 끝내 두었다 (`swoz_v1`).

**6. Mimi 출처가 학습과 추론에서 다르다.** 학습 config는 `kyutai/mimi`,
`infer.py`는 `kyutai/stt-1b-en_fr`에서 로드한다. 재현 시 확인 필요.

---

## 환경

torch 버전이 두 제약의 교집합으로 **고정**된다.
- `moshi`(모델 본체의 `StreamingTransformer` 제공)가 `torch<2.10` 요구
- RTX 50xx(sm_120)는 cu128 빌드 필요
→ **torch 2.9.1+cu128** 만이 둘을 동시에 만족하고, H100(sm_90)도 커버한다.

⚠️ **torch를 먼저 설치할 것.** moshi를 먼저 깔면 moshi의 핀이 cu128 빌드를 조용히 CPU 빌드로 덮어쓴다.

```bash
bash scripts/setup_env.sh          # 위 순서를 지켜 설치 + 검증까지
```

### Windows 전용 이슈 두 개 (HPC 리눅스에서는 불필요)

| 증상 | 원인 | 대응 |
|---|---|---|
| `torchaudio.load` → `libtorchcodec_image.dll` 로드 실패 | torchaudio 2.9는 모든 I/O를 TorchCodec으로 보내고, Windows 휠은 FFmpeg 공유 라이브러리 없이는 import 단계에서 죽는다 | `scripts/sitecustomize.py`를 venv의 site-packages에 복사. **TorchCodec이 정상이면 아무 일도 하지 않으므로 리눅스에서는 자동 비활성.** soundfile로 대체하며 PCM WAV 왕복 오차 0.0 확인 |
| `AttributeError: Can't get local object 'mimi.<locals>.encode'` (DataLoader) | Windows DataLoader workers use **spawn**, so the dataset must pickle. Mimi's extractor is a class defined *inside* `models/mimi.py::mimi`, so it cannot. Linux forks and never pickles it. | `EPA_NUM_WORKERS=0`. (`src/data/__init__.py` hardcoded `num_workers=8`; patched to read `EPA_NUM_WORKERS`, default still 8 — see `git diff`) |
| `TritonMissing` | moshi의 RoPE가 `torch.compile`을 쓰는데 Windows에 Triton이 없다 | `NO_TORCH_COMPILE=1` 환경변수. 리눅스에서는 설정하지 말 것 (compile이 더 빠름) |

---

## 데이터

```bash
python scripts/download_spokenwoz.py --root .           # 12.5GB 압축 / ~30GB 해제
#   전처리 dump는 별도로 ~130GB (스모크 40대화 1.1GB 실측 기준 환산)
#   -> scratch에 최소 160GB 확보 필요
```

SpokenWOZ (CC BY-NC 4.0, 249시간 / 5.7k 대화 / 203k 턴). HF의 4개 repo에서 받아
로더가 기대하는 구조로 배치한다.

```
data/SpokenWOZ/
  audio_5700_train_dev/   *.wav   2채널 8kHz (ch0=user, ch1=system)
  audio_5700_test/        *.wav
  text_5700_train_dev/    data.json, valListFile.json
  text_5700_test/         data.json, testListFile.json
```

**Switchboard**는 LDC 라이선스가 필요하고 상류는 Kaldi `fisher_swbd/s5/data/` 디렉터리를 기대한다.
없으면 `swoz_v1`(SpokenWOZ 단독)로 진행 — 다만 논문에서 EPA가 가장 약한 수치(자유대화 HEA 22.1%)가
Switchboard 쪽이라 자유대화 일반화 주장은 못 하게 된다. 확보하면 `swoz_swbd_v1` 사용.

> **Phase 2에 중요:** SpokenWOZ는 `words[i].BeginTime/EndTime` 로 **단어 단위 타임스탬프**를 준다.
> 임의의 `t_pred`에서 부분 전사를 ASR 없이 정확히 잘라낼 수 있어, "부분 전제가 충분한가"와
> "ASR이 정확한가"를 분리해 측정할 수 있다.

### config 생성

```bash
python scripts/make_configs.py --root .                        # 로컬
python scripts/make_configs.py --root /scratch/$USER/EPA \
       --data-config swoz_v1 --num-workers 32                  # HPC
```

상류 config를 템플릿으로 삼아 **경로 필드만** 재작성한다 (하이퍼파라미터·주석은 그대로).

---

## 학습 (4× H100)

```bash
bash scripts/launch_train.sh table1   # fc640, fc1280, fcall  — Table 1에 필요한 최소
bash scripts/launch_train.sh fig2     # + fc960, fc2560       — Figure 2 horizon
bash scripts/launch_train.sh all      # 전 horizon + EPA-M    (9런)
sbatch scripts/train.sbatch table1    # SLURM
```

GPU당 1런씩 채우고 wave 단위로 대기한다. `fcall`이 EPA-M(전 horizon 한 모델)이다.

학습 설정(상류 그대로): 50 epoch, batch 8, lr 3e-4, early stopping patience 6,
`save_best_from_val_acc`, frame-level BCE (`non_forecast_frames` 0.1 / `forecast_frames` 0.5).

모델: 6층 Transformer, hidden 512, heads 4, FFN 1024, RoPE, **context 240 프레임**
(논문 본문은 250이라고 적었으나 config는 240 — 12.5Hz 기준 19.2초).
Mimi frozen, 첫 8 codebook, 12.5Hz, lookahead 0, 24kHz 입력.

---

## 검증 상태

| 항목 | 상태 |
|---|---|
| 환경 (torch/moshi/Mimi/silero/wandb) | ✅ import 전부 통과 |
| `infer.py` 스모크 | ✅ 136 프레임, anticipation 4회 |
| **저자 참조 출력 대조** | ✅ **anticipation 판정 완전 일치** — 동일 4프레임 `[5.2, 5.28, 5.36, 5.44]`, 확률 최대오차 0.0095 / 평균 0.00085 (GPU·빌드 차이 범위) |
| config 로드 + `run_name` | ✅ fc640/fc960/fcall 정상, rstrip 버그 회피 |
| 데이터 | ✅ 4,700 train-dev + 1,000 test 대화, 무결성 확인 |
| **전처리 파이프라인** | ✅ 4단계 전부 정상 (스모크 40대화) — 채널분리 → 24kHz 리샘플 → Silero VAD → 필터.<br>세그먼트 train 1,358 → 2,715 (user/system 병합), val 392 → 784 |
| **학습 루프** | ✅ **end-to-end 통과** — 스텝 진행, 검증, 체크포인트 저장(`best_val_acc.pt`), config 스냅샷, `train.json`, early-stopping 카운터.<br>**파라미터 25.19M — 논문의 "25M streaming Transformer"와 일치** |
| 소요 시간 참고 | 스모크(32대화) 에폭당 64s, step 1.73s/it @ RTX 5080.<br>환산: 전체 4,200대화 → 5080에서 ~15분/에폭, H100 ~5-7분/에폭 → 런당 4~6시간 (50에폭 상한) |
| Table 1 재현 | ⛔ 학습 필요 (위 발견 1) |
