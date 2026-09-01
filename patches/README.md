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

## 적용된 패치 — 1건뿐

### `epa-num-workers.patch`

`src/data/__init__.py` 의 하드코딩된 `num_workers=8` 을 환경변수
`EPA_NUM_WORKERS` (기본 8)로 뺀다. Windows 는 워커 프로세스 spawn 비용이 커서
로컬 스모크 실행 시 `EPA_NUM_WORKERS=0` 이 필요하다.

**상류 수정은 이 1건으로 유지한다** (README §9-4). 재현 대상은 저자가 실제로 돌린
코드이고, 배포된 h=960 체크포인트와 대조하려면 같은 코드여야 한다. 아래 항목들은
전부 상류 그대로 두고, 대신 하네스 쪽에서 **감지·보고**한다.

## 검토했지만 패치하지 않은 상류 동작

문제가 아니라고 판단한 것이 아니라, **고치는 값보다 기준선을 지키는 값이 크다**고
판단한 것들이다. 결과 해석에 영향을 주는 항목은 README 부록 A에도 적어 두었다.

| 위치 | 내용 | 왜 두는가 / 어떻게 대응하는가 |
|---|---|---|
| `src/training/forecasting_trainer.py:54` | **검증이 `torch.no_grad()` 없이 돈다.** train/val이 같은 함수이고 `model.eval()` 은 dropout·norm만 바꾼다. val 배치마다 6층 트랜스포머 그래프를 만들고 아무도 backward 하지 않는다 | 수치에는 영향이 없고 메모리·시간만 손해다. 학습이 메모리로 막히면 `handle_start_of_epoch` 에 `torch.set_grad_enabled(mode == "train")` 한 줄이면 된다 |
| `src/training/default_trainer.py:135` | **쓰지 않는 confusion-matrix 버퍼를 에폭 내내 쌓는다.** `handle_batch_loss` 가 `self.cm` 에 전 프레임 라벨을 `.tolist()` 로 누적하는데, `ForecastingTrainer.handle_end_of_epoch` 는 그걸 읽지 않는다(`DefaultTrainer` 만 쓴다). 배치마다 GPU→CPU 동기화 + `fcall` 기준 에폭당 약 3400만 개의 파이썬 float | 위와 같은 이유. 필요하면 `ForecastingTrainer` 에서 `handle_batch_loss` 를 오버라이드해 손실 기록만 남기면 된다 |
| `src/models/mimi.py:50` | `encode.__call__` 이 `chunk_length` 인자를 받고도 **무시한다.** 추론이 대화 전체를 Mimi에 한 번에 넣는다 | Mimi가 causal이라 수치는 동일(부록 B의 저자 참조 출력 대조와 일치). 평가에서 메모리만 튄다 — 긴 대화에서 OOM이 나면 그때 청크 분할 |
| `src/models/fc_base_lstm.py:171-174` | `delay_frames` 분기가 시간축이 아니라 **horizon축**을 자른다. docstring의 `(B, num_horizons, T)` 도 틀렸다(실제로는 `(B, T, num_horizons)` 이라 두 `permute` 가 왕복 무연산) | mimi config에 `audio_params.delay_frames` 가 없어 휴면 상태다. **추가하지 말 것** |
| `src/training/forecasting_trainer.py:200` | HEA를 분수 상태에서 `.round(2)` → **1%p로 양자화**. 논문의 66.3 같은 값은 원리상 나올 수 없다 | 관문 허용오차(±3%p) 안. `report_table1.py` 가 표 아래에 각주를 단다 |
| `src/utils/data_utils.py:119` | val 창을 **에폭마다 무작위로 다시 뽑는다**(시드 없음). `save_best_from_val_acc` 와 early stopping이 그 위에서 돈다 | 재현 기준선. `preflight.sh` 가 런별 `@epoch` / `val range` 를 찍어 조기 종료가 노이즈에 걸렸는지 보여준다. 납득이 안 되면 README 부록 A의 최소 수정을 검토 |
| `src/models/fc_base_lstm.py:194-202` | 체크포인트 선택 지표 `val_accuracy` 가 **가중치 없는** 프레임 정확도다. 손실은 5:1로 가중하지만 선택은 아니다 — 전부 0으로 예측하면 fc640에서 ~93.8% | 부록 A-3. `report_table1.py` 의 `never fired at ANY threshold` 경고와 §7의 "첫 런만 먼저 평가" 절차로 잡는다 |
| `src/data/data_processing.py:29` | `handle_and_add_turns` 가 mode 루프 안에서 `continue` 가 아니라 **`return`** 이다. `processed_train.json` 이 있으면 val을 만들지 않고 함수를 빠져나간다 | `check_preprocessed.py` 가 이 상태를 따로 잡아 `rm processed_*.json` 명령을 출력한다 (README §8-I) |
| `src/training/default_trainer.py:197` | 조기 종료가 `exit()` 로 프로세스를 끝낸다 — 그 에폭의 `train.json` 기록 **전에**. 종료코드는 0이다 | `launch_train.sh` 요약이 런별 epochs 수를 함께 찍으므로 완주와 사고를 구분할 수 있다 |
