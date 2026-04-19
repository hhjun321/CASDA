# CASDA Obsidian Vault Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CASDA 프로젝트를 위한 Obsidian vault 10개 노트 생성 — Claude 참조용 및 사용자 위키 겸용

**Architecture:** `D:/project/CASDA/CASDA/` vault root에 숫자 접두사 파일 10개 생성. `00-INDEX.md`가 Claude 진입점. 파이프라인 노트(02~05)는 공통 템플릿 적용. `06-Scripts-Reference.md`가 전체 스크립트 입출력 매핑.

**Tech Stack:** Markdown, Obsidian wikilinks (`[[파일명]]`), README.md + QUICKSTART.md 원본 내용 기반

---

## 파일 맵

| 파일 | 역할 |
|------|------|
| `CASDA/00-INDEX.md` | Claude 진입점, 전체 컨텍스트 맵 |
| `CASDA/01-Overview.md` | 프로젝트 개요, 아키텍처 |
| `CASDA/02-Pipeline-StageA.md` | 데이터 전처리 |
| `CASDA/03-Pipeline-StageB.md` | ControlNet 학습 + 생성 |
| `CASDA/04-Pipeline-StageC.md` | 후처리 + 품질관리 |
| `CASDA/05-Pipeline-StageD.md` | 평가 |
| `CASDA/06-Scripts-Reference.md` | 전체 스크립트 입출력 매핑 표 |
| `CASDA/07-Models.md` | 3개 모델 상세 |
| `CASDA/08-Dataset-Groups.md` | Severstal 데이터셋, 7개 실험 그룹 |
| `CASDA/09-Experiments.md` | 실험 기록 (완료 + 템플릿) |

---

## Task 1: 00-INDEX.md — Claude 진입점

**Files:**
- Create: `CASDA/00-INDEX.md`

- [ ] **Step 1: 파일 생성**

```markdown
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
```

- [ ] **Step 2: 파일 생성 확인**

```bash
ls CASDA/00-INDEX.md
```

- [ ] **Step 3: 커밋**

```bash
git add CASDA/00-INDEX.md
git commit -m "docs: add Obsidian vault index (Claude context map)"
```

---

## Task 2: 01-Overview.md — 프로젝트 개요

**Files:**
- Create: `CASDA/01-Overview.md`

- [ ] **Step 1: 파일 생성**

```markdown
# 01 — 프로젝트 개요

> **Claude 요약:** CASDA는 ControlNet 기반 철강 결함 이미지 합성 + Poisson Blending 후처리 + 품질 필터링으로 데이터를 증강하고, 3개 모델(YOLO-MFD, EB-YOLOv8, DeepLabV3+)로 효과를 검증하는 연구 파이프라인이다.

## Abstract

철강 제조 표면 결함 탐지는 심각한 클래스 불균형, 제한된 결함 샘플, 벤치마크 데이터셋의 배경 다양성 부족으로 인해 어렵다. CASDA(Context-Aware Steel Defect Augmentation)는 기하학적 결함 특성 분석과 ControlNet 기반 조건부 이미지 합성을 결합하여 물리적으로 타당한 결함 이미지를 생성하는 생성적 데이터 증강 프레임워크다.

- 데이터셋: [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/data)
- Backbone: Stable Diffusion v1.5 + ControlNet (sd-controlnet-canny)
- 평가: 3개 모델 아키텍처 × 7개 데이터셋 그룹

## 핵심 특징 (Highlights)

1. **ControlNet 기반 결함 합성** — 멀티채널 힌트 (R=결함 형태, G=구조, B=텍스처) + 하이브리드 텍스트 프롬프트
2. **Poisson Blending 합성** — 생성된 512×512 ROI를 실제 1600×256 철강 이미지에 자연스럽게 합성
3. **품질 인식 프루닝** — 적합도 점수 기반 필터링 (색상 일관성, 아티팩트 탐지, 선명도)
4. **3개 벤치마크 모델** — YOLO-MFD, EB-YOLOv8, DeepLabV3+
5. **7개 데이터셋 그룹** — 각 파이프라인 구성요소 기여도 격리 ablation 포함
6. **통계적 가설 검정** — 증강 효과의 엄격한 검증
7. **FID 평가** — ROI 수준 및 전체 이미지 수준, 클래스별·서브타입별 세분화

## 전체 파이프라인

```
Stage A: 데이터 전처리 (CPU)
  ├── ROI 패치 추출 + 적합도 점수
  └── 멀티채널 ControlNet 힌트 + 하이브리드 텍스트 프롬프트 준비
          ↓
Stage B: ControlNet 학습 + 생성 (GPU)
  ├── ControlNet 학습 (SD v1.5 + sd-controlnet-canny)
  ├── 멀티페이즈 검증 (선택)
  └── 합성 결함 이미지 생성
          ↓
Stage C: 후처리 + 품질관리 (CPU)
  ├── Poisson Blending 합성 (깨끗한 배경에 합성)
  ├── 적합도 점수 (색상 / 아티팩트 / 선명도)
  ├── 품질 인식 프루닝 (계층별 top-k)
  └── CopyPaste 베이스라인 생성
          ↓
Stage D: 평가 (GPU)
  ├── FID 평가 (ROI 수준 + 전체 이미지)
  ├── 최적 분할 비율 탐색
  ├── 벤치마크: 3개 모델 × 7개 데이터셋 그룹
  └── 통계 가설 검정 + 결과 분석
```

## 리포지토리 구조

```
CASDA/
├── configs/
│   └── benchmark_experiment.yaml
├── scripts/          # 실행 가능한 파이프라인 스크립트 (14개)
├── src/
│   ├── models/       # YOLO-MFD, EB-YOLOv8, DeepLabV3+
│   ├── training/     # 데이터셋, 트레이너, 평가기
│   ├── preprocessing/
│   ├── analysis/
│   └── utils/
├── QUICKSTART.md
├── README.md
└── requirements.txt
```

## 관련 노트

[[00-INDEX]] | [[02-Pipeline-StageA]] | [[07-Models]] | [[08-Dataset-Groups]]
```

- [ ] **Step 2: 커밋**

```bash
git add CASDA/01-Overview.md
git commit -m "docs: add Obsidian vault overview note"
```

---

## Task 3: 02-Pipeline-StageA.md — 데이터 전처리

**Files:**
- Create: `CASDA/02-Pipeline-StageA.md`

- [ ] **Step 1: 파일 생성**

```markdown
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
```

- [ ] **Step 2: 커밋**

```bash
git add CASDA/02-Pipeline-StageA.md
git commit -m "docs: add Obsidian vault Stage A note"
```

---

## Task 4: 03-Pipeline-StageB.md — ControlNet 학습 + 생성

**Files:**
- Create: `CASDA/03-Pipeline-StageB.md`

- [ ] **Step 1: 파일 생성**

```markdown
# 03 — Stage B: ControlNet 학습 + 생성

> **Claude 요약:** Stage A에서 준비한 멀티채널 힌트로 ControlNet을 학습시키고(SD v1.5 기반), 4단계 검증 후 클래스별 합성 결함 이미지를 생성한다. GPU 필수.

## 목적

- SD v1.5 + sd-controlnet-canny 기반 ControlNet을 결함 이미지 합성에 파인튜닝
- 학습된 모델로 클래스별 합성 결함 이미지 생성
- 선택적 4단계 검증으로 모델 품질 확인

## 입력 / 출력

| 항목 | 경로 변수 | 설명 |
|------|-----------|------|
| **입력** | `CN_DATASET` | Stage A 출력 (힌트, train.jsonl) |
| **출력** | `CN_TRAINING` | 학습 체크포인트, 로그 |
| **출력** | `BEST_MODEL` | 최적 모델 (`${CN_TRAINING}/best_model`) |
| **출력** | `CN_VALIDATION` | 검증 결과 (선택) |
| **출력** | `AUG_IMAGES` | 생성된 합성 결함 이미지 |

## Step 3: train_controlnet.py — ControlNet 학습

```bash
!python ${SCRIPTS}/train_controlnet.py \
  --data_dir                        ${CN_DATASET} \
  --output_dir                      ${CN_TRAINING} \
  --pretrained_model_name_or_path   runwayml/stable-diffusion-v1-5 \
  --controlnet_model_name_or_path   lllyasviel/sd-controlnet-canny \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --mixed_precision fp16 \
  --num_train_epochs 20 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine \
  --lr_warmup_steps 50 \
  --controlnet_conditioning_scale 1.0 \
  --snr_gamma 5.0 \
  --early_stopping_patience 20 \
  --validation_steps 200 \
  --logging_steps 10 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 3 \
  --save_fp16 \
  --skip_save_pipeline \
  --seed 42
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--resolution` | 512 | 학습 이미지 해상도 |
| `--train_batch_size` | 1 | 배치 크기 (T4 기준 1) |
| `--gradient_accumulation_steps` | 4 | 실질적 배치 크기 = 4 |
| `--mixed_precision` | fp16 | 메모리 절약 |
| `--num_train_epochs` | 20 | 최대 에폭 |
| `--learning_rate` | 1e-5 | 학습률 |
| `--lr_scheduler` | cosine | 스케줄러 |
| `--early_stopping_patience` | 20 | 조기 종료 patience |
| `--controlnet_conditioning_scale` | 1.0 | ControlNet 조건화 강도 |
| `--snr_gamma` | 5.0 | SNR 가중치 gamma |
| `--checkpoints_total_limit` | 3 | 저장할 최대 체크포인트 수 |

## Step 4: run_validation_phases.py — 모델 검증 (선택)

```bash
!python ${SCRIPTS}/run_validation_phases.py \
  --model_path          ${BEST_MODEL} \
  --jsonl_path          ${CN_DATASET}/train.jsonl \
  --image_root          ${CN_DATASET} \
  --roi_metadata_path   ${ROI_DIR}/roi_metadata.csv \
  --training_log_path   ${CN_TRAINING}/training_log.json \
  --output_base         ${CN_VALIDATION} \
  --phases 1 2 3 4 \
  --controlnet_conditioning_scale 0.7 \
  --resolution 512 \
  --workers 8
```

### 검증 4단계

| Phase | 내용 |
|-------|------|
| 1 | 기본 생성 품질 확인 |
| 2 | 힌트 조건화 충실도 |
| 3 | 클래스별 다양성 |
| 4 | 전체 파이프라인 통합 |

## Step 5: test_controlnet.py — 합성 이미지 생성

```bash
!python ${SCRIPTS}/test_controlnet.py \
  --model_path    ${BEST_MODEL} \
  --jsonl_path    ${CN_DATASET}/train.jsonl \
  --output_dir    ${AUG_IMAGES} \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --controlnet_conditioning_scale 0.7 \
  --num_images_per_class '{"1":2,"2":10,"3":1,"4":2}' \
  --resolution 512
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--num_inference_steps` | 30 | 디퓨전 추론 스텝 수 |
| `--guidance_scale` | 7.5 | CFG 스케일 |
| `--controlnet_conditioning_scale` | 0.7 | 학습(1.0)보다 낮게 설정 |
| `--num_images_per_class` | `{"1":2,"2":10,"3":1,"4":2}` | 클래스별 생성 이미지 수 (클래스 2 희귀하여 10개) |

### 출력

- `${AUG_IMAGES}/generated/` — 생성된 합성 결함 이미지 (512×512)
- `${AUG_IMAGES}/generation_summary.json` — 생성 요약

## 주의사항

- GPU 필수 (T4 이상)
- `--controlnet_conditioning_scale` 생성 시 0.7로 낮춤 (학습 1.0 → 생성 0.7)
- 클래스 2는 희귀 클래스로 다른 클래스보다 많이 생성
- `--skip_save_pipeline`으로 전체 파이프라인 저장 생략 (체크포인트만 저장)

## 관련 노트

[[00-INDEX]] | [[02-Pipeline-StageA]] | [[04-Pipeline-StageC]] | [[06-Scripts-Reference]]
```

- [ ] **Step 2: 커밋**

```bash
git add CASDA/03-Pipeline-StageB.md
git commit -m "docs: add Obsidian vault Stage B note"
```

---

## Task 5: 04-Pipeline-StageC.md — 후처리 + 품질관리

**Files:**
- Create: `CASDA/04-Pipeline-StageC.md`

- [ ] **Step 1: 파일 생성**

```markdown
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
```

- [ ] **Step 2: 커밋**

```bash
git add CASDA/04-Pipeline-StageC.md
git commit -m "docs: add Obsidian vault Stage C note"
```

---

## Task 6: 05-Pipeline-StageD.md — 평가

**Files:**
- Create: `CASDA/05-Pipeline-StageD.md`

- [ ] **Step 1: 파일 생성**

```markdown
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
```

- [ ] **Step 2: 커밋**

```bash
git add CASDA/05-Pipeline-StageD.md
git commit -m "docs: add Obsidian vault Stage D note"
```

---

## Task 7: 06-Scripts-Reference.md — 전체 스크립트 입출력 매핑

**Files:**
- Create: `CASDA/06-Scripts-Reference.md`

- [ ] **Step 1: 파일 생성**

```markdown
# 06 — 스크립트 전체 입출력 매핑

> **Claude 요약:** 14개 파이프라인 스크립트의 입출력 변수, Stage, CPU/GPU 여부를 한 곳에서 확인. 코드 수정·디버깅·파라미터 확인 시 이 노트를 먼저 참조.

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
```

- [ ] **Step 2: 커밋**

```bash
git add CASDA/06-Scripts-Reference.md
git commit -m "docs: add Obsidian vault scripts reference note"
```

---

## Task 8: 07-Models.md — 모델 상세

**Files:**
- Create: `CASDA/07-Models.md`

- [ ] **Step 1: 파일 생성**

```markdown
# 07 — 벤치마크 모델

> **Claude 요약:** CASDA 벤치마크에 사용된 3개 모델. YOLO-MFD와 EB-YOLOv8는 탐지(Detection), DeepLabV3+는 분할(Segmentation). 모두 YOLOv8s 또는 ResNet-101 기반.

## 모델 비교

| 모델 | 유형 | Base | 핵심 개선 | 소스 |
|------|------|------|-----------|------|
| **YOLO-MFD** | Detection | YOLOv8s | MEFE 모듈 — 멀티스케일 Sobel 엣지 + 채널 어텐션 | `src/models/yolo_mfd.py` |
| **EB-YOLOv8** | Detection | YOLOv8s | BiFPN — 양방향 특징 피라미드 + 정규화 가중치 융합 | `src/models/eb_yolov8.py` |
| **DeepLabV3+** | Segmentation | ResNet-101 | ASPP (atrous rates 6/12/18) 인코더-디코더 | `src/models/deeplabv3plus.py` |

## YOLO-MFD

**Multi-scale Feature-enhanced Defect detector**

- Base: YOLOv8s
- 핵심 모듈: **MEFE (Multi-scale Edge Feature Enhancement)**
  - 멀티스케일 Sobel 엣지 추출 (미세 결함 강조)
  - 채널 어텐션 메커니즘
  - 목적: 미세 결함(micro-defect) 탐지 성능 향상
- 평가 지표: mAP@0.5, per-class AP, precision-recall
- 소스: `src/models/yolo_mfd.py`

## EB-YOLOv8

**Edge-BiFPN YOLOv8**

- Base: YOLOv8s
- 핵심 모듈: **BiFPN (Bi-directional Feature Pyramid Network)**
  - 양방향 특징 피라미드 (top-down + bottom-up)
  - 빠른 정규화 가중치 융합 (fast normalized weighted fusion)
  - Depthwise separable convolution으로 효율화
- 평가 지표: mAP@0.5, per-class AP, precision-recall
- 소스: `src/models/eb_yolov8.py`

## DeepLabV3+

- Base: ResNet-101
- 핵심 모듈: **ASPP (Atrous Spatial Pyramid Pooling)**
  - Atrous rates: 6, 12, 18
  - Output stride: 16
  - 인코더-디코더 구조
- 평가 지표: Dice coefficient (전체 + 클래스별), IoU
- 역할: Stage D Step 12에서 최적 분할 비율 탐색의 기준 모델
- 소스: `src/models/deeplabv3plus.py`

## 벤치마크 실행 그룹

Step 13에서 3개 모델 모두, Step 14에서 YOLO-MFD만 실행:

```bash
# Step 13: 3개 모델 × 5개 그룹
--models yolo_mfd eb_yolov8 deeplabv3plus  (기본값)

# Step 14: YOLO-MFD만 × 2개 ablation 그룹
--models yolo_mfd
```

## 관련 노트

[[00-INDEX]] | [[05-Pipeline-StageD]] | [[08-Dataset-Groups]]
```

- [ ] **Step 2: 커밋**

```bash
git add CASDA/07-Models.md
git commit -m "docs: add Obsidian vault models note"
```

---

## Task 9: 08-Dataset-Groups.md — 데이터셋 그룹

**Files:**
- Create: `CASDA/08-Dataset-Groups.md`

- [ ] **Step 1: 파일 생성**

```markdown
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
```

- [ ] **Step 2: 커밋**

```bash
git add CASDA/08-Dataset-Groups.md
git commit -m "docs: add Obsidian vault dataset groups note"
```

---

## Task 10: 09-Experiments.md — 실험 기록

**Files:**
- Create: `CASDA/09-Experiments.md`

- [ ] **Step 1: 파일 생성**

```markdown
# 09 — 실험 기록

> **Claude 요약:** 완료된 실험 결과와 향후 실험 기록용 템플릿. 새 실험 시 템플릿 섹션을 복사해서 사용.

---

## 완료된 실험

### 전체 파이프라인 실험 — 2025

**목적:** CASDA 증강 프레임워크 전체 효과 검증

**설정:**
- 환경: Google Colab (T4 GPU)
- 데이터셋: Severstal Steel Defect Detection
- 모델: YOLO-MFD, EB-YOLOv8, DeepLabV3+
- 그룹: baseline_raw, baseline_trad, casda_composed, casda_composed_pruning, copypaste
- Ablation (YOLO-MFD): ablation_no_blending, ablation_no_pruning

**ControlNet 학습 설정:**
- Base: SD v1.5 + sd-controlnet-canny
- epochs: 20, batch: 1 × grad_accum 4, lr: 1e-5 (cosine)
- mixed_precision: fp16, seed: 42

**생성 설정:**
- num_inference_steps: 30, guidance_scale: 7.5
- controlnet_conditioning_scale: 0.7
- num_images_per_class: {"1":2, "2":10, "3":1, "4":2}

**결과:**
<!-- 실험 완료 후 여기에 핵심 지표 기록 -->
- FID (ROI): 
- FID (full-image): 
- YOLO-MFD mAP@0.5 (casda_composed_pruning vs baseline_raw): 
- EB-YOLOv8 mAP@0.5: 
- DeepLabV3+ Dice: 
- 최적 분할 비율: 

**관찰:**
<!-- 주요 발견 사항, 예상과 다른 결과, 다음 실험 아이디어 -->

**참조 파일:**
- `${BENCHMARK_RESULTS}/`
- `${FID_RESULTS}/`
- `${SPLIT_RESULTS}/`

---

## 새 실험 템플릿

> 아래 템플릿을 복사하여 새 실험 기록에 사용하세요.

```
### [실험명] — YYYY-MM-DD

**목적:**

**설정:**
- 환경:
- 변경 사항 (이전 실험 대비):
- 주요 파라미터:

**실행 명령 (변경된 부분만):**
\`\`\`bash

\`\`\`

**결과:**
- 지표 1:
- 지표 2:

**결론:**

**다음 실험 아이디어:**
```

---

## 실험 아이디어 목록

<!-- 향후 시도할 실험 아이디어를 여기에 메모 -->

- [ ] 

## 관련 노트

[[00-INDEX]] | [[01-Overview]] | [[06-Scripts-Reference]] | [[08-Dataset-Groups]]
```

- [ ] **Step 2: 커밋**

```bash
git add CASDA/09-Experiments.md
git commit -m "docs: add Obsidian vault experiments note"
```

---

## 최종 확인

- [ ] **10개 파일 모두 생성 확인**

```bash
ls CASDA/*.md
```

Expected output:
```
CASDA/00-INDEX.md
CASDA/01-Overview.md
CASDA/02-Pipeline-StageA.md
CASDA/03-Pipeline-StageB.md
CASDA/04-Pipeline-StageC.md
CASDA/05-Pipeline-StageD.md
CASDA/06-Scripts-Reference.md
CASDA/07-Models.md
CASDA/08-Dataset-Groups.md
CASDA/09-Experiments.md
```
