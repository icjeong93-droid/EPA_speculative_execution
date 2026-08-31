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

## epa-num-workers.patch

`src/data/__init__.py` 의 하드코딩된 `num_workers=8` 을 환경변수
`EPA_NUM_WORKERS` (기본 8)로 뺀다. Windows 는 워커 프로세스 spawn 비용이 커서
로컬 스모크 실행 시 `EPA_NUM_WORKERS=0` 이 필요하다.
