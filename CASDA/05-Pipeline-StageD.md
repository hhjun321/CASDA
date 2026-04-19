# 05 — Stage D: 평가

> **Claude 요약:** FID로 합성 품질을 측정하고, 최적 분할 비율을 탐색한 뒤, 3개 모델 × 7개 데이터셋 그룹 벤치마크와 통계 가설 검정으로 증강 효과를 검증한다. GPU 필수.

## 목적

- FID 평가: 합성 이미지의 분포 거리 측정 (ROI 수준 + 전체 이미지 수준)
- 최적 분할 비율 탐색: train/val/test 비율 실험
- 벤치마크: 3개 모델 × 5개 주요 그룹 + 2개 ablation 그룹
- 통계 검정: H3~H6 가설 검증

## 입력 / 출력

| 항목 | 경로 변수 | 설명 |
|------|-----------|------|
| **입력** | `AUG_DATASET` | Stage C 출력 전체 |
| **입력** | `AUG_IMAGES` | Stage B 생성 이미지 |
| **입력** | `TRAIN_IMAGES` | 원본 이미지 |
| **입력** | `CONFIG` | benchmark_experiment.yaml |
| **출력** | `FID_RESULTS` | FID 평가 결과 |
| **출력** | `SPLIT_RESULTS` | 분할 비율 실험 결과 |
| **출력** | `BENCHMARK_RESULTS` | 벤치마크 결과 |
| **출력** | `YOLO_DATASETS` | YOLO 포맷 데이터셋 |

## Step 11: run_fid.py — FID 평가

```bash
!python ${SCRIPTS}/run_fid.py \
  --config             ${CONFIG} \
  --data-dir           ${TRAIN_IMAGES} \
  --csv                ${TRAIN_CSV} \
  --casda-dir          ${AUG_DATASET} \
  --casda-roi-dir      ${AUG_IMAGES}/generated \
  --roi-metadata-csv   ${ROI_DIR}/roi_metadata.csv \
  --output-dir         ${FID_RESULTS} \
  --workers 12
```

### FID 평가 수준

| 수준 | 설명 |
|------|------|
| ROI 수준 | 512×512 패치 FID |
| 전체 이미지 수준 | 1600×256 합성 이미지 FID |
| 클래스별 | 4개 결함 클래스 각각 |
| 서브타입별 | 세분화된 결함 유형별 |

## Step 12: run_split_experiment.py — 최적 분할 비율 탐색

> DeepLabV3+를 기준 모델로 사용하여 최적 train/val/test 분할 비율 탐색

```bash
!python ${SCRIPTS}/run_split_experiment.py \
  --csv          ${TRAIN_CSV} \
  --data-dir     ${TRAIN_IMAGES} \
  --config       ${CONFIG} \
  --output-dir   ${SPLIT_RESULTS} \
  --ratios "60/20/20,70/15/15,80/10/10" \
  --epochs 100 \
  --seed 42
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--ratios` | `"60/20/20,70/15/15,80/10/10"` | 탐색할 분할 비율 목록 |
| `--epochs` | 100 | 각 비율별 학습 에폭 |
| `--seed` | 42 | 재현성 시드 |

## Step 13: run_benchmark.py — 주요 벤치마크

> 3개 모델 × 5개 주요 데이터셋 그룹

```bash
!python ${SCRIPTS}/run_benchmark.py \
  --config              ${CONFIG} \
  --data-dir            ${LOCAL_IMAGES} \
  --groups              baseline_raw baseline_trad casda_composed casda_composed_pruning copypaste \
  --casda-dir           ${AUG_DATASET} \
  --yolo-dir            ${YOLO_DATASETS} \
  --output-dir          ${BENCHMARK_RESULTS} \
  --no-fid \
  --reference-results   ${REFERENCE_RESULTS}
```

## Step 14: run_benchmark.py — Ablation 연구

> yolo_mfd 모델만, 2개 ablation 그룹

```bash
!python ${SCRIPTS}/run_benchmark.py \
  --config              ${CONFIG} \
  --data-dir            ${LOCAL_IMAGES} \
  --models              yolo_mfd \
  --groups              ablation_no_pruning ablation_no_blending \
  --casda-dir           ${AUG_DATASET} \
  --yolo-dir            ${YOLO_DATASETS} \
  --output-dir          ${BENCHMARK_RESULTS} \
  --no-fid \
  --reference-results   ${REFERENCE_RESULTS}
```

## 통계 가설

| 가설 | 내용 |
|------|------|
| H3 | 아키텍처 독립성 — 증강 효과가 모델에 무관하게 일관됨 |
| H4 | 소수 클래스 개선 — 희귀 클래스 탐지 성능 향상 |
| H5 | 물리적 타당성 (FID) — CASDA가 CopyPaste보다 낮은 FID |
| H6 | 최적 증강 비율 — 최적 합성:실제 비율 존재 |

## 주의사항

- Step 13은 `LOCAL_IMAGES`(Colab 로컬 디스크) 사용 → I/O 속도 최적화
- `--no-fid` 플래그로 벤치마크 중 FID 재계산 생략 (Step 11에서 별도 수행)
- Ablation은 `yolo_mfd`만 실행 (Step 14)
- `REFERENCE_RESULTS`는 이미 완료된 실험 결과 파일 경로

## 관련 노트

[[00-INDEX]] | [[04-Pipeline-StageC]] | [[06-Scripts-Reference]] | [[07-Models]] | [[08-Dataset-Groups]]
