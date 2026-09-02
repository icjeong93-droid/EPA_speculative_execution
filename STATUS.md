# EPA 재현 — 현재 상태와 남은 절차

작성: 2026-09-02 · 대상 클러스터: `SR_AISolution_ACU` · 계정: `ic.jeong`

> 이 문서는 **이번 실행의 상태 스냅샷**이다. 절차의 근거와 배경은 `README.md`(런북)에,
> 실패 모드 표는 README §8에 있다. 여기에는 **지금 어디까지 왔고 다음에 무엇을 치는지**만 적는다.

---

## 0. 한 줄 요약

**설치까지 끝났다.** 다음은 config 생성 → 전처리 → 학습 → 평가.

---

## 1. 확정된 경로

| 무엇 | 경로 |
|---|---|
| 작업 디렉터리 | `/home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA` |
| 데이터셋 | `/home/sr5/SR_AISolution_ACU/database/EPA/SpokenWOZ` |
| dump (전처리 산출물, ~110GB) | `/home/sr5/SR_AISolution_ACU/database/EPA/dump` |
| 번들 (설치 후 삭제 가능) | `~/bundle` |

작업과 데이터가 다른 파일시스템에 나뉘어 있다. **이 분리는 의도된 것이다** — 데이터·dump는
공용 database 영역, venv·체크포인트·로그는 개인 작업 공간.

모든 명령 전에:

```bash
cd /home/sr5/SR_AISolution_ACU/workspace/ic.jeong/EPA
source ./env.sh
```

---

## 2. 완료된 것

- [x] 오프라인 번들 전송 (4.6GB) 및 설치
- [x] `hf-xet`, `bitsandbytes` wheel 수동 보충 — 아래 §5 참조
- [x] `hf_cache` 중첩(`hf_cache/hf_cache`) 해소
- [x] `setup_offline.sh` 통과 — `torch 2.9.1+cu128` / `cuda True` / `sm_90` / `silero OK` / `mimi OK`
- [x] 데이터 배치 확인 (train_dev 4700 wav, test 1000 wav)

---

## 3. 남은 절차

### 3-1. Config 생성  ← **지금 여기**

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

### 3-2. 전처리 — CPU 잡, 1~2시간

```bash
sbatch scripts/preprocess.sbatch          # train + val — 학습에 필요
sbatch scripts/preprocess.sbatch test     # test 분할 — 평가에 필요, 지금 같이 걸어둘 것
```

**이 단계를 건너뛰고 학습을 제출하지 말 것.** 전처리는 `run.py` 안에서도 돌기 때문에,
바로 학습을 걸면 H100 4장이 리샘플링이 끝날 때까지 논다.

**검증**

```bash
bash scripts/preflight.sh $PWD          # 전처리 완료 여부를 리샘플 파일 개수까지 세어 판정
du -sh /home/sr5/SR_AISolution_ACU/database/EPA/dump    # ~110GB
```

`filtered_*.json` 이 있다고 완료가 아니다 — 그 파일은 리샘플링 **시작 전에** 쓰인다.
`check_preprocessed.py` 가 그 구분을 하고, preflight·evaluate·학습 제출이 모두 그것을 기준으로 삼는다.

**중간에 끊겼다면** preflight가 두 경우를 구분해 알려준다.

- `[WARN] 전처리 미완` → **그냥 재실행.** 완료된 단계는 건너뛴다.
- `[FAIL] train과 val 사이에서 끊김` → **지우고 재실행.**
  `rm /home/sr5/SR_AISolution_ACU/database/EPA/dump/spokenwoz/processed_*.json`

---

### 3-3. 학습 — GPU 잡, 8~12시간

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
Number of trainable parameters: 25.19M      ← 다르면 config가 잘못된 것. 중단하고 3-1 재확인
Mode: train, Loss: 0.xxxx, Acc: 0.xxxx
val epoch: {'val_total': ..., 'val_accuracy': ...}
Saving model at epoch N to .../best_val_acc.pt
```

**val 곡선이 에폭마다 튀는 것은 정상이다** — 상류가 validation 창을 매 에폭 무작위로 다시 뽑는다.
다만 `save_best_from_val_acc` 와 early stopping이 그 위에서 도니, **조기 종료가 10에폭 이전에
걸리면** README 부록 A의 val 시드 고정을 검토할 것.

재제출은 안전하다 — `best_val_acc.pt` 가 이미 있는 런은 건너뛰고 요약에 이름이 찍힌다
(강제 재학습은 `FORCE=1`).

---

### 3-4. 평가 — Table 1 뽑기, 1~2시간

```bash
sbatch scripts/evaluate.sbatch           # 학습된 런 전부 → 추론 → 표 출력
# 특정 런만: bash scripts/evaluate.sh fc640 fcall
```

**이 단계를 해야 숫자가 나온다.** 학습은 val accuracy만 남기고, MRA/HEA/PAR/ERC는
`infer_params` 가 있는 config(=`configs/infer.yaml`)로 도는 추론 경로에서만 계산된다.
여기까지 안 하면 체크포인트만 남고 논문과 대조할 값은 하나도 없다.

---

## 4. 결과를 보고할 때

관문: **SpokenWOZ EPA-M h=640에서 HEA 67.0% ± 3%p, MRA 640ms ± 100ms, ERC 33.8% ± 3%p.**

반드시 함께 적을 것:

1. **학습 데이터가 논문과 다르다** — 논문은 SpokenWOZ + Switchboard, 우리는 SpokenWOZ 단독
   (Switchboard는 LDC 라이선스 미확보). **관문 미달은 재현 실패가 아니라 조건 차이다.**
2. 논문 본문과 저자 코드가 어긋나는 4건(배포 구현을 따랐음) — README 부록 A.

---

## 5. 이번 설치에서 걸렸던 것 (재발 시 대응)

| 증상 | 원인 | 대응 |
|---|---|---|
| `No matching distribution found for hf-xet` / `bitsandbytes` | pip은 환경 마커를 **빌드 호스트** 기준으로 평가한다. Windows에서 번들을 만들면 `sys_platform == "linux"` / `platform_machine == "x86_64"` 로 걸린 의존성이 통째로 빠진다 | 해당 wheel만 받아 `bundle/wheels/` 에 넣고 `setup_offline.sh` 재실행. **빌드 스크립트에 검사가 추가되어 이제는 빌드 시점에 잡힌다** (`linux_only_gaps`) |
| `OSError: We couldn't connect to https://huggingface.co ... couldn't find them in the cached files` | `hf_cache/hf_cache` 중첩. `scp -r hf_cache remote:bundle/` 인데 원격에 이미 `bundle/hf_cache` 가 있으면 그 **안으로** 들어간다 | `mv $H/hf_cache/* $H/ && rmdir $H/hf_cache` 를 번들과 대상 양쪽에 |
| `mkdir: Permission denied` | 존재하지 않는 상위 디렉터리를 만들려 한 것 (경로 오타) | `pwd` 로 실제 경로를 확인. scp는 `host:` 뒤를 비우면 원격 홈으로 간다 |
| `transfer.ps1` 이 매번 비밀번호를 물음 | 스크립트가 ssh를 30회 이상 호출한다. Windows OpenSSH는 접속 재사용(ControlMaster) 미지원 | 공개키를 `~/.ssh/authorized_keys` 에 등록. 홈이 그룹 쓰기 가능이면 sshd가 키를 무시하므로 `chmod go-w ~` 도 함께 |

**교훈 하나** — wheel을 수동으로 받을 때는 **플랫폼·ABI 태그를 PyPI에서 먼저 확인**할 것.
번들의 다른 wheel과 같은 태그일 것이라 짐작하면 틀린다. 실제로 `hf-xet` 은 `cp38-abi3` +
`manylinux_2_17`, `bitsandbytes` 는 `py3-none` + `manylinux_2_24` 로 서로 달랐다.

---

## 6. 참고

- 절차의 근거·배경: `README.md`
- 실패 모드 표: `README.md` §8
- 금지 사항(어떤 경우에도 위반 금지): `README.md` §9
- 상류 동작 중 고치지 않은 것들과 그 이유: `patches/README.md`
