# 상류 저장소 패치

상류: https://github.com/bloodraven66/EndpointAnticipation
고정 커밋: `531e1d7e70751e980b82088a3a358a2e75fa8a12` (main)

`EndpointAnticipation/` 은 이 저장소에서 추적하지 않는다(자체 `.git` 보유).
재현 시:

```bash
git clone https://github.com/bloodraven66/EndpointAnticipation.git
git -C EndpointAnticipation checkout 531e1d7e70751e980b82088a3a358a2e75fa8a12
git -C EndpointAnticipation apply ../patches/epa-num-workers.patch
```

## 적용됨

### `epa-num-workers.patch`

`src/data/__init__.py` 의 하드코딩된 `num_workers=8` 을 환경변수
`EPA_NUM_WORKERS` (기본 8)로 뺀다. Windows 는 워커 프로세스 spawn 비용이 커서
로컬 스모크 실행 시 `EPA_NUM_WORKERS=0` 이 필요하다.

## 적용 안 함 — 필요할 때 판단해서 적용할 것

### `epa-val-nograd.patch`

**수치에 영향 없음.** 학습 결과·지표·체크포인트가 전부 동일하고, 검증 루프의
메모리와 시간만 줄인다. 그럼에도 기본 적용하지 않는 이유는 README §9-4다 —
상류 수정이 1건에서 2건으로 늘고, 배포된 h=960 체크포인트와의 "동일 코드" 주장이
약해진다. 학습이 메모리로 막히거나 val이 눈에 띄게 느릴 때 적용할 것.

두 가지를 고친다.

1. **검증이 `torch.no_grad()` 없이 돈다.**
   `ForecastingTrainer.train` 은 train/val이 같은 함수다. `model.eval()` 은
   dropout·norm만 바꾸고 autograd는 그대로라, val 배치마다 6층 트랜스포머의
   그래프를 만들고 아무도 backward 하지 않는다. `handle_start_of_epoch` 에서
   `torch.set_grad_enabled(mode == "train")` 로 모드에 맞춰 껐다 켠다
   (train 에폭 시작 때 다시 켜지므로 상태가 새지 않는다).

2. **쓰지 않는 confusion-matrix 버퍼를 에폭 내내 쌓는다.**
   `DefaultTrainer.handle_batch_loss` 는 `self.cm` 에 전 프레임 라벨을
   `.detach().cpu().tolist()` 로 누적하고 `DefaultTrainer.handle_end_of_epoch` 가
   그걸로 confusion matrix를 만든다. 그런데 **`ForecastingTrainer.handle_end_of_epoch`
   는 `self.cm` 을 읽지 않는다.** 상속된 채로 배치마다 GPU→CPU 동기화를 하고,
   `fcall` 기준 에폭당 약 3400만 개의 파이썬 float을 쌓았다가 다음 에폭 시작에서
   통째로 버린다. 해당 메서드를 `ForecastingTrainer` 에서 오버라이드해 손실 기록만
   남긴다.

```bash
git -C EndpointAnticipation apply ../patches/epa-val-nograd.patch
```

적용했다면 **결과 보고 시 상류 수정이 2건임을 함께 적을 것** (README §10-3).

## 검토했지만 패치하지 않은 상류 동작

고치면 배포 체크포인트와 조건이 달라지거나, 지금 경로에서는 발동하지 않는 것들.

| 위치 | 내용 | 왜 두는가 |
|---|---|---|
| `src/models/mimi.py:50` | `encode.__call__` 이 `chunk_length` 인자를 무시한다. 추론이 대화 전체를 Mimi에 한 번에 넣는다 | Mimi가 causal이라 수치는 동일(부록 B 대조와 일치). 평가에서 메모리만 튄다 — OOM이 나면 그때 청크 분할 |
| `src/models/fc_base_lstm.py:171-174` | `delay_frames` 분기가 시간축이 아니라 horizon축을 자른다 | mimi config에 `audio_params.delay_frames` 가 없어 휴면. **추가하지 말 것** |
| `src/training/forecasting_trainer.py:200` | HEA를 분수 상태에서 `.round(2)` → 1%p로 양자화 | 관문 허용오차(±3%p) 안. `report_table1.py` 가 출력에 각주를 단다 |
| `src/utils/data_utils.py:119` | val 창을 에폭마다 무작위로 다시 뽑는다 (시드 없음) | 재현 기준선. 조기 종료가 납득 안 되는 시점에 걸리면 README 부록 A의 최소 수정을 검토 |
| `src/training/default_trainer.py:197` | 조기 종료가 `exit()` 로 프로세스를 끝낸다 — 그 에폭의 `train.json` 기록 전에 | 종료코드 0이라 `launch_train.sh` 가 정상으로 본다. 그래서 요약에 epochs 수를 함께 찍는다 |
