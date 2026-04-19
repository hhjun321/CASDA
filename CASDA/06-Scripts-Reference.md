# 06 — 스크립트 전체 입출력 매핑

> **Claude 요약:** 파이프라인 핵심 14개 스크립트의 입출력 변수, Stage, CPU/GPU 여부를 한 곳에서 확인. 코드 수정·디버깅·파라미터 확인 시 이 노트를 먼저 참조.

## 전체 스크립트 매핑

| Step | 스크립트 | Stage | 주요 입력 변수 | 주요 출력 변수 | CPU/GPU |
|------|----------|-------|----------------|----------------|---------|
| 1 | `extract_rois.py` | A | `TRAIN_IMAGES`, `TRAIN_CSV` | `ROI_DIR` | CPU |
| 2 | `prepare_controlnet_data.py` | A | `ROI_DIR`, `TRAIN_IMAGES`, `TRAIN_CSV` | `CN_DATASET` | CPU |
| 3 | `train_controlnet.py` | B | `CN_DATASET` | `CN_TRAINING`, `BEST_MODEL` | GPU |
| 4 | `run_validation_phases.py` | B | `BEST_MODEL`, `CN_DATASET`, `ROI_DIR` | `CN_VALIDATION` | GPU |
| 5 | `test_controlnet.py` | B | `BEST_MODEL`, `CN_DATASET` | `AUG_IMAGES` | GPU |
| 6 | `compose_casda_images.py` | C | `AUG_IMAGES`, `CN_DATASET`, `TRAIN_IMAGES` | `CASDA_COMPOSED`, `BG_CACHE` | CPU |
| 7 | `create_copypaste_baseline.py` | C | `ROI_DIR`, `TRAIN_IMAGES`, `TRAIN_CSV` | `COPYPASTE_DIR` | CPU |
| 8 | `compose_casda_images.py` (--no-blend) | C | `AUG_IMAGES`, `CN_DATASET`, `TRAIN_IMAGES` | `CASDA_NO_BLEND` | CPU |
| 9 | `score_casda_quality.py` | C | `CASDA_COMPOSED` | 점수 파일 (in-place) | CPU |
| 10 | `validate_augmented_quality.py` | C | `CASDA_COMPOSED` | 검증 리포트 | CPU |
| 11 | `run_fid.py` | D | `TRAIN_IMAGES`, `AUG_DATASET`, `AUG_IMAGES` | `FID_RESULTS` | GPU |
| 12 | `run_split_experiment.py` | D | `TRAIN_CSV`, `TRAIN_IMAGES`, `CONFIG` | `SPLIT_RESULTS` | GPU |
| 13 | `run_benchmark.py` | D | `LOCAL_IMAGES`, `AUG_DATASET`, `CONFIG` | `BENCHMARK_RESULTS`, `YOLO_DATASETS` | GPU |
| 14 | `run_benchmark.py` (ablation) | D | `LOCAL_IMAGES`, `AUG_DATASET`, `CONFIG` | `BENCHMARK_RESULTS` | GPU |

## 경로 변수 → 실제 경로 매핑

| 변수 | 실제 경로 |
|------|-----------|
| `PROJ` | `/content/CASDA` |
| `SCRIPTS` | `/content/CASDA/scripts` |
| `CONFIG` | `/content/CASDA/configs/benchmark_experiment.yaml` |
| `DRIVE_DATA` | `/content/drive/MyDrive/data/Severstal` |
| `TRAIN_IMAGES` | `${DRIVE_DATA}/train_images` |
| `TRAIN_CSV` | `${DRIVE_DATA}/train.csv` |
| `ROI_DIR` | `${DRIVE_DATA}/roi_patches` |
| `CN_DATASET` | `${DRIVE_DATA}/controlnet_dataset` |
| `CN_TRAINING` | `${DRIVE_DATA}/controlnet_training` |
| `BEST_MODEL` | `${CN_TRAINING}/best_model` |
| `AUG_IMAGES` | `${DRIVE_DATA}/augmented_images` |
| `CASDA_COMPOSED` | `${DRIVE_DATA}/augmented_dataset/casda_composed` |
| `COPYPASTE_DIR` | `${DRIVE_DATA}/augmented_dataset/copypaste_baseline` |
| `CASDA_NO_BLEND` | `${DRIVE_DATA}/augmented_dataset/casda_no_blend` |
| `BG_CACHE` | `${DRIVE_DATA}/cache/bg_types.json` |
| `FID_RESULTS` | `${DRIVE_DATA}/fid_results` |
| `SPLIT_RESULTS` | `${DRIVE_DATA}/split_experiment` |
| `BENCHMARK_RESULTS` | `${DRIVE_DATA}/benchmark_results` |
| `YOLO_DATASETS` | `${DRIVE_DATA}/yolo_datasets` |
| `LOCAL_IMAGES` | `/content/dataset_local/train_images` |

## src/ 모듈 구조

| 모듈 경로 | 역할 |
|-----------|------|
| `src/models/yolo_mfd.py` | YOLO-MFD 모델 정의 |
| `src/models/eb_yolov8.py` | EB-YOLOv8 모델 정의 |
| `src/models/deeplabv3plus.py` | DeepLabV3+ 모델 정의 |
| `src/training/` | 데이터셋, 트레이너, 평가기 |
| `src/preprocessing/` | ROI 추출, ControlNet 데이터 준비 |
| `src/analysis/` | FID, 벤치마크 분석, 통계 검정 |
| `src/utils/` | I/O, 시각화, 설정 헬퍼 |

## 관련 노트

[[00-INDEX]] | [[02-Pipeline-StageA]] | [[03-Pipeline-StageB]] | [[04-Pipeline-StageC]] | [[05-Pipeline-StageD]]
