# 02 — Stage A: 데이터 전처리

> **Claude 요약:** Severstal 원본 이미지에서 256×256 ROI 패치를 추출하고, ControlNet 학습을 위한 멀티채널 힌트 이미지와 train.jsonl을 생성한다. CPU 전용, 1회 실행.

## 목적

- 결함 영역이 포함된 ROI 패치를 추출하여 ControlNet 학습 데이터로 사용
- 멀티채널 힌트 (R=결함 마스크 형태, G=Canny 엣지 구조, B=텍스처) 생성
- 하이브리드 텍스트 프롬프트 포함 train.jsonl 생성

## 입력 / 출력

| 항목 | 경로 변수 | 설명 |
|------|-----------|------|
| **입력** | `TRAIN_IMAGES` | Severstal 원본 학습 이미지 (1600×256) |
| **입력** | `TRAIN_CSV` | 결함 RLE 마스크 레이블 |
| **출력** | `ROI_DIR` | ROI 패치 + roi_metadata.csv |
| **출력** | `CN_DATASET` | 멀티채널 힌트, train.jsonl, packaged_roi_metadata.csv |

## Step 1: extract_rois.py — ROI 패치 추출

```bash
!python ${SCRIPTS}/extract_rois.py \
  --image_dir      ${TRAIN_IMAGES} \
  --train_csv      ${TRAIN_CSV} \
  --output_dir     ${ROI_DIR} \
  --roi_size 256 \
  --grid_size 64 \
  --min_suitability 0.5 \
  --num_workers 8
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--roi_size` | 256 | ROI 패치 크기 (px) |
| `--grid_size` | 64 | 그리드 슬라이딩 간격 (px) |
| `--min_suitability` | 0.5 | 최소 적합도 점수 (이하 제외) |
| `--num_workers` | 8 | 병렬 워커 수 |

### 출력

- `${ROI_DIR}/` — 추출된 ROI 패치 이미지
- `${ROI_DIR}/roi_metadata.csv` — 패치별 메타데이터 (위치, 클래스, 적합도 점수)

## Step 2: prepare_controlnet_data.py — ControlNet 학습 데이터 준비

```bash
!python ${SCRIPTS}/prepare_controlnet_data.py \
  --roi_metadata         ${ROI_DIR}/roi_metadata.csv \
  --train_images         ${TRAIN_IMAGES} \
  --train_csv            ${TRAIN_CSV} \
  --output_dir           ${CN_DATASET} \
  --per_class_cap 1200 \
  --rare_class_threshold 200 \
  --class_edge_override "4:0.05,0.0" \
  --skip_validation
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--per_class_cap` | 1200 | 클래스별 최대 학습 샘플 수 |
| `--rare_class_threshold` | 200 | 희귀 클래스 판단 임계값 |
| `--class_edge_override` | `"4:0.05,0.0"` | 클래스 4의 Canny 엣지 파라미터 오버라이드 |
| `--skip_validation` | flag | 데이터 검증 단계 건너뜀 |

### 출력

- `${CN_DATASET}/hints/` — 멀티채널 힌트 이미지
- `${CN_DATASET}/train.jsonl` — ControlNet 학습용 JSONL
- `${CN_DATASET}/packaged_roi_metadata.csv` — 패키징된 메타데이터

## 주의사항

- CPU 전용, 1회만 실행하면 됨 (반복 불필요)
- `--min_suitability 0.5` 미만 패치는 이후 단계에서 사용 불가
- 클래스 4는 엣지 특성이 달라 `class_edge_override` 필요

## 관련 노트

[[00-INDEX]] | [[03-Pipeline-StageB]] | [[06-Scripts-Reference]] | [[08-Dataset-Groups]]
