# CASDA — Claude Context Map

> **Claude에게:** 새 대화에서 이 파일을 먼저 읽으세요. 프로젝트 전체 컨텍스트가 여기 있습니다.

## 프로젝트 한 줄 요약

ControlNet으로 철강 결함 이미지를 합성하고, Poisson Blending으로 실제 배경에 합성한 뒤, 품질 필터링 후 3개 모델 벤치마크로 데이터 증강 효과를 검증하는 연구 파이프라인.

## 실행 환경

| 항목 | 값 |
|------|----|
| 환경 | Google Colab (GPU T4+) |
| 코드 경로 | `/content/CASDA` |
| 데이터 루트 | `/content/drive/MyDrive/data/Severstal` |
| Python | 3.10+ |
| GPU | CUDA (Stage B, D 필수) |

## 전체 경로 변수 (QUICKSTART 기준)

```bash
PROJ=/content/CASDA
SCRIPTS=${PROJ}/scripts
CONFIG=${PROJ}/configs/benchmark_experiment.yaml

DRIVE_DATA=/content/drive/MyDrive/data/Severstal
TRAIN_IMAGES=${DRIVE_DATA}/train_images
TRAIN_CSV=${DRIVE_DATA}/train.csv

ROI_DIR=${DRIVE_DATA}/roi_patches
CN_DATASET=${DRIVE_DATA}/controlnet_dataset

CN_TRAINING=${DRIVE_DATA}/controlnet_training
CN_VALIDATION=${DRIVE_DATA}/controlnet_validation
BEST_MODEL=${CN_TRAINING}/best_model
AUG_IMAGES=${DRIVE_DATA}/augmented_images

AUG_DATASET=${DRIVE_DATA}/augmented_dataset
CASDA_COMPOSED=${AUG_DATASET}/casda_composed
COPYPASTE_DIR=${AUG_DATASET}/copypaste_baseline
CASDA_NO_BLEND=${AUG_DATASET}/casda_no_blend
BG_CACHE=${DRIVE_DATA}/cache/bg_types.json

FID_RESULTS=${DRIVE_DATA}/fid_results
SPLIT_RESULTS=${DRIVE_DATA}/split_experiment
BENCHMARK_RESULTS=${DRIVE_DATA}/benchmark_results
YOLO_DATASETS=${DRIVE_DATA}/yolo_datasets
REFERENCE_RESULTS=${DRIVE_DATA}/casda/benchmark_results.json

LOCAL_IMAGES=/content/dataset_local/train_images
```

## 파이프라인 흐름

```
Stage A (CPU) → Stage B (GPU) → Stage C (CPU) → Stage D (GPU)
전처리          ControlNet      후처리/품질      평가/벤치마크
```

## 노트 맵

| 노트 | 내용 | Claude 활용 시점 |
|------|------|-----------------|
| [[01-Overview]] | 전체 아키텍처, Abstract, Highlights | 프로젝트 첫 파악 시 |
| [[02-Pipeline-StageA]] | ROI 추출, ControlNet 데이터 준비 | Stage A 작업·디버깅 시 |
| [[03-Pipeline-StageB]] | ControlNet 학습, 검증, 이미지 생성 | Stage B 작업·디버깅 시 |
| [[04-Pipeline-StageC]] | Poisson Blending, 품질 점수, 검증 | Stage C 작업·디버깅 시 |
| [[05-Pipeline-StageD]] | FID, 벤치마크, 통계 검정 | Stage D 작업·디버깅 시 |
| [[06-Scripts-Reference]] | 전체 스크립트 입출력 표 | 코드 수정·파라미터 확인 시 |
| [[07-Models]] | YOLO-MFD, EB-YOLOv8, DeepLabV3+ | 모델 구조 파악·수정 시 |
| [[08-Dataset-Groups]] | Severstal, 7개 실험 그룹, 평가 지표 | 데이터셋·그룹 구성 파악 시 |
| [[09-Experiments]] | 완료된 실험 결과, 새 실험 템플릿 | 실험 기록·계획 시 |

## 완료된 실험

→ [[09-Experiments]] 참조

## 주요 참조 파일

| 파일 | 위치 | 용도 |
|------|------|------|
| README.md | `/content/CASDA/README.md` | 전체 프로젝트 문서 |
| QUICKSTART.md | `/content/CASDA/QUICKSTART.md` | 단계별 실행 가이드 |
| benchmark_experiment.yaml | `configs/benchmark_experiment.yaml` | 실험 설정 |
