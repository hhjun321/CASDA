# 04 — Stage C: 후처리 + 품질관리

> **Claude 요약:** 생성된 512×512 ROI를 Poisson Blending으로 실제 1600×256 배경에 합성하고, 적합도 점수로 품질 필터링한다. CopyPaste 베이스라인과 Blending 없는 ablation 변형도 생성.

## 목적

- 합성된 ROI를 실제 철강 이미지 배경에 자연스럽게 합성 (Poisson Blending)
- 색상 일관성·아티팩트·선명도 기반 품질 점수 산출 및 프루닝
- 비교 베이스라인 생성: CopyPaste (블렌딩 없음), Blending 없는 ablation

## 입력 / 출력

| 항목 | 경로 변수 | 설명 |
|------|-----------|------|
| **입력** | `AUG_IMAGES` | Stage B 생성 이미지 |
| **입력** | `CN_DATASET` | 힌트, packaged_roi_metadata.csv |
| **입력** | `TRAIN_IMAGES` | 깨끗한 배경 이미지 소스 |
| **입력** | `TRAIN_CSV` | 배경 선택용 레이블 |
| **출력** | `CASDA_COMPOSED` | Poisson Blending 합성 이미지 |
| **출력** | `COPYPASTE_DIR` | CopyPaste 베이스라인 |
| **출력** | `CASDA_NO_BLEND` | Blending 없는 ablation |
| **캐시** | `BG_CACHE` | 배경 타입 캐시 JSON |

## Step 6: compose_casda_images.py — Poisson Blending 합성

```bash
!python ${SCRIPTS}/compose_casda_images.py \
  --generated-dir       ${AUG_IMAGES}/generated \
  --hint-dir            ${CN_DATASET}/hints \
  --metadata-csv        ${CN_DATASET}/packaged_roi_metadata.csv \
  --summary-json        ${AUG_IMAGES}/generation_summary.json \
  --clean-images-dir    ${TRAIN_IMAGES} \
  --train-csv           ${TRAIN_CSV} \
  --output-dir          ${CASDA_COMPOSED} \
  --workers 8 \
  --bg-cache            ${BG_CACHE} \
  --compositions-per-roi 5
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--compositions-per-roi` | 5 | ROI 1개당 배경 합성 횟수 |
| `--workers` | 8 | 병렬 워커 수 |
| `--bg-cache` | `BG_CACHE` | 배경 분류 캐시 (재사용) |

## Step 7: create_copypaste_baseline.py — CopyPaste 베이스라인

> ControlNet, Blending 없이 원본 ROI를 배경에 직접 붙이는 비교 베이스라인

```bash
!python ${SCRIPTS}/create_copypaste_baseline.py \
  --roi-dir            ${ROI_DIR} \
  --metadata-csv       ${ROI_DIR}/roi_metadata.csv \
  --clean-images-dir   ${TRAIN_IMAGES} \
  --train-csv          ${TRAIN_CSV} \
  --output-dir         ${COPYPASTE_DIR} \
  --workers 8 \
  --bg-cache           ${BG_CACHE}
```

## Step 8: compose_casda_images.py — Blending 없는 ablation

> Poisson Blending 효과를 격리하는 ablation 변형 (`--no-blend` 플래그)

```bash
!python ${SCRIPTS}/compose_casda_images.py \
  --generated-dir       ${AUG_IMAGES}/generated \
  --hint-dir            ${CN_DATASET}/hints \
  --metadata-csv        ${CN_DATASET}/packaged_roi_metadata.csv \
  --summary-json        ${AUG_IMAGES}/generation_summary.json \
  --clean-images-dir    ${TRAIN_IMAGES} \
  --train-csv           ${TRAIN_CSV} \
  --output-dir          ${CASDA_NO_BLEND} \
  --no-blend \
  --workers 8 \
  --bg-cache            ${BG_CACHE}
```

## Step 9: score_casda_quality.py — 품질 점수 산출

```bash
!python ${SCRIPTS}/score_casda_quality.py \
  --casda-dir  ${CASDA_COMPOSED} \
  --workers 10
```

### 품질 점수 구성요소

| 구성요소 | 설명 |
|----------|------|
| 색상 일관성 | 합성 ROI와 배경 색상 분포 일치도 |
| 아티팩트 탐지 | 블렌딩 경계 아티팩트 탐지 |
| 선명도 | 이미지 선명도 (Laplacian variance) |

## Step 10: validate_augmented_quality.py — 품질 검증

```bash
!python ${SCRIPTS}/validate_augmented_quality.py \
  --augmented_dir       ${CASDA_COMPOSED} \
  --min_quality_score   0.7 \
  --workers 12
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--min_quality_score` | 0.7 | 최소 품질 점수 (이하 제외) |
| `--workers` | 12 | 병렬 워커 수 |

## 주의사항

- Step 6, 8은 동일 스크립트(`compose_casda_images.py`)를 `--no-blend` 유무로 구분
- `BG_CACHE`는 Step 6에서 생성되면 Step 7, 8에서 재사용 (속도 향상)
- 품질 점수 0.7 미만 이미지는 이후 `casda_composed_pruning` 그룹에서 제외됨

## 관련 노트

[[00-INDEX]] | [[03-Pipeline-StageB]] | [[05-Pipeline-StageD]] | [[06-Scripts-Reference]] | [[08-Dataset-Groups]]
