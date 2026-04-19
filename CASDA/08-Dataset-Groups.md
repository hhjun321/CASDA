# 08 — 데이터셋 및 실험 그룹

> **Claude 요약:** Severstal 데이터셋 기반 4개 결함 클래스, 7개 실험 그룹(5개 주요 + 2개 ablation). 각 그룹은 파이프라인 구성요소 기여도를 격리하여 검증.

## Severstal Steel Defect Detection 데이터셋

| 항목 | 값 |
|------|----|
| 출처 | [Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection/data) |
| 이미지 크기 | 1600×256 px |
| 결함 클래스 | 4개 (Class 1, 2, 3, 4) |
| 레이블 형식 | RLE 마스크 (train.csv) |

### 클래스별 특성

| 클래스 | 특성 | 빈도 |
|--------|------|------|
| Class 1 | - | 보통 |
| Class 2 | 희귀 클래스 | 매우 낮음 |
| Class 3 | - | 높음 |
| Class 4 | 엣지 특성 다름 (`class_edge_override` 필요) | 보통 |

## 7개 실험 그룹

| 그룹 | 설명 | 포함 데이터 |
|------|------|-------------|
| `baseline_raw` | 증강 없음 | 원본 Severstal만 |
| `baseline_trad` | 전통적 증강 | 원본 + flip/rotation/scale/brightness/contrast |
| `casda_composed` | CASDA 전체 | 원본 + Poisson Blending 합성 CASDA 이미지 전체 |
| `casda_composed_pruning` | CASDA 프루닝 | 원본 + 품질 필터링된 CASDA 이미지 (top-k) |
| `copypaste` | CopyPaste 베이스라인 | 원본 ROI를 직접 배경에 붙임 (ControlNet·Blending 없음) |
| `ablation_no_blending` | Blending 없는 ablation | ControlNet ROI 직접 붙임 (Poisson Blending 없음) |
| `ablation_no_pruning` | 프루닝 없는 ablation | 품질 필터링 없음, 수량 제한만 적용 |

### 그룹별 벤치마크 실행

| 그룹 | Step 13 | Step 14 |
|------|---------|---------|
| `baseline_raw` | 3개 모델 | - |
| `baseline_trad` | 3개 모델 | - |
| `casda_composed` | 3개 모델 | - |
| `casda_composed_pruning` | 3개 모델 | - |
| `copypaste` | 3개 모델 | - |
| `ablation_no_blending` | - | YOLO-MFD만 |
| `ablation_no_pruning` | - | YOLO-MFD만 |

## 평가 지표

### Detection (YOLO-MFD, EB-YOLOv8)

| 지표 | 설명 |
|------|------|
| mAP@0.5 | mean Average Precision at IoU 0.5 |
| per-class AP | 클래스별 Average Precision |
| Precision-Recall curve | PR 곡선 |

### Segmentation (DeepLabV3+)

| 지표 | 설명 |
|------|------|
| Dice coefficient | 전체 + 클래스별 |
| IoU | Intersection over Union |

### Synthesis Quality

| 지표 | 설명 |
|------|------|
| FID (ROI) | 512×512 패치 수준 Fréchet Inception Distance |
| FID (full-image) | 1600×256 전체 이미지 수준 FID |
| Suitability score | 색상 일관성 + 아티팩트 + 선명도 합산 |

### 통계 가설 검정

| 가설 | 검정 내용 |
|------|-----------|
| H3 | 증강 효과의 아키텍처 독립성 |
| H4 | 소수 클래스(Class 2) 탐지 성능 향상 |
| H5 | CASDA FID < CopyPaste FID (물리적 타당성) |
| H6 | 최적 합성:실제 비율 존재 여부 |

## 관련 노트

[[00-INDEX]] | [[05-Pipeline-StageD]] | [[07-Models]]
