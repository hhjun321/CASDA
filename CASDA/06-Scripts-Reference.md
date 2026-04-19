# 06 — 스크립트 전체 입출력 매핑

> **Claude 요약:** 파이프라인 핵심 14개 스크립트의 입출력 변수, Stage, CPU/GPU 여부 및 각 스크립트의 핵심 계산 로직을 한 곳에서 확인. 코드 수정·디버깅·파라미터 확인 시 이 노트를 먼저 참조.

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

---

## 스크립트별 핵심 계산 로직

### Step 1 — `extract_rois.py`

**5단계 파이프라인:**
1. **배경 분석** — 64px 그리드 셀 단위로 분산·엣지 밀도 계산 → `background_type` 분류 (smooth / vertical_stripe / horizontal_stripe / textured / complex_pattern)
2. **결함 특성화** — 각 결함 마스크에서 4개 지표 산출: 면적(area), 견고도(solidity), 조밀도(compactness), 종횡비(aspect_ratio)
3. **ROI 적합도 평가** — 결함-배경 매칭 점수 계산 → `suitability_score` (0~1)
4. **256×256 패치 추출** — `roi_size=256`, `grid_size=64` 슬라이딩 윈도우
5. **메타데이터 저장** — `roi_metadata.csv` (위치, 클래스, 적합도, 배경 유형, 결함 서브타입)

**적합도 임계값:** `min_suitability=0.5` 미만 패치 제외

---

### Step 2 — `prepare_controlnet_data.py`

**멀티채널 힌트 이미지 생성 (3채널):**

| 채널 | 내용 | 방법 |
|------|------|------|
| R | 결함 마스크 형태 | 이진 마스크 (0/255) |
| G | 구조 / 엣지 | Canny 엣지 검출 (클래스 4는 `class_edge_override`로 파라미터 조정) |
| B | 텍스처 | Gabor 필터 기반 텍스처 추출 |

**텍스트 프롬프트 생성:** 결함 유형 + 배경 유형 조합 → 자연어 하이브리드 프롬프트

**클래스 균형 조정:**
- `--per_class_cap 1200` — 클래스별 최대 샘플 수 상한
- `--rare_class_threshold 200` — 이하이면 희귀 클래스로 간주, 오버샘플링

**출력:** `train.jsonl` (image, hint, prompt, negative_prompt 필드), `hints/`, `packaged_roi_metadata.csv`

---

### Step 3 — `train_controlnet.py`

**모델 구조:**
- 기반: Stable Diffusion v1.5 (VAE·텍스트 인코더·UNet — 동결)
- 학습 대상: ControlNet 가중치만 업데이트

**손실 함수:**
```
L = MSE(ε_pred, ε_target)
```
Min-SNR-γ 가중치 적용 시: `w(t) = min(SNR(t), γ) / SNR(t)`, γ=5.0

**학습 안정화:**
- ControlNet은 항상 fp32 유지 (fp16 시 NaN 기울기 방지)
- 동결 모델은 AMP 활성화 시 fp16
- 기울기 클리핑: `max_norm=1.0`
- NaN 손실 감지 시 학습률 자동 감소

**데이터 증강 (B3):**
- 수평 반전 50% 확률
- 밝기·대비 지터: 0.9–1.1 범위

**그레이스케일 강제 변환 (B1):** 생성 이미지 → 그레이스케일 → RGB 3채널 복사 (VAE RGB 아티팩트 방지)

**체크포인트:** `checkpointing_steps=500`, `checkpoints_total_limit=3`, 최적 모델 → `best_model/`

---

### Step 4 — `run_validation_phases.py`

**품질 점수 3지표 (가중 합산):**

| 지표 | 가중치 | 계산 방법 |
|------|--------|-----------|
| 색상 일관성 | 40% | LAB 공간: 채도(0.20) + 밝기 범위(0.25) + 엔트로피(0.30) + 텍스처 표준편차(0.25) |
| 아티팩트 탐지 | 30% | Sobel 기울기 이상치 비율(0.6) + 고주파 에너지 비율(0.4) |
| 선명도 | 30% | Laplacian 분산(0.5) + 엣지 대비 P90/P50(0.5) |

**주요 임계값:**

| 항목 | 정상 범위 |
|------|-----------|
| Chroma | 3.0 – 25.0 |
| L_range | 30 – 150 |
| 엔트로피 | ≥ 4.5 |
| 엣지 이상치 비율 | ≤ 0.01 |
| 고주파 비율 | ≤ 2.5 |
| Laplacian 분산 | ≥ 500 (선명) |

**FID 계산:** InceptionV3 특징 추출 → Fréchet 거리

**검증 4단계:** (1) 데이터 무결성, (2) 초기 에폭 샘플링, (3) 전체 테스트셋 평가, (4) 클래스별 분해

---

### Step 5 — `test_controlnet.py`

**추론 파이프라인:**
```
힌트 이미지 → ControlNet forward → UNet (residual conditioning) → 역방향 디퓨전 (denoise)
```

**생성 후처리:** 생성 이미지 → 그레이스케일 → RGB 3채널 (VAE RGB 아티팩트 제거)

**클래스별 생성 수:** `--num_images_per_class '{"1":2,"2":10,"3":1,"4":2}'`
- 클래스 2는 희귀하여 10배 더 생성

**conditioning_scale 조정:** 학습 1.0 → 추론 0.7 (네온 아티팩트 방지)

**출력:** `generated/` (512×512), `comparisons/` (힌트|생성1|생성2|원본 그리드), `generation_summary.json`

---

### Step 6 — `compose_casda_images.py`

**Poisson Blending 파이프라인:**
1. **배경 풀 관리** — 1600×256 배경을 엣지 분석·분산으로 5가지 유형 분류
2. **호환성 매칭** — 결함 서브타입 × 배경 유형 → 점수 행렬 (1.0=완벽, 0.2=불량)
3. **밝기 매칭** — ±30 허용 범위, 후보 부족 시 2× 확대 → 전체 허용 순으로 폴백
4. **Poisson Blending** — `seamlessClone` + 마스크 팽창 8px + Gaussian σ=7.0 스무딩
5. **위치 증강** — 위치 지터 ±100px, 멀티스케일 0.875–1.0

**`--compositions-per-roi N`:** ROI 1개당 N개의 서로 다른 배경·지터·스케일 조합 생성

**`--no-blend` (Step 8 ablation):** Poisson Blending 없이 직접 붙여넣기

---

### Step 7 — `create_copypaste_baseline.py`

**CopyPaste 알고리즘:**
- ROI 마스크 이진화 (임계값 > 127) → 배경에 직접 붙여넣기 (블렌딩 없음)
- 배경 풀 관리·밝기 매칭은 Step 6과 동일 로직 사용
- ControlNet 합성 없이 원본 ROI 사용

**목적:** ControlNet 합성의 기여도를 격리하는 비교 베이스라인

---

### Step 9 — `score_casda_quality.py`

**3지표 가중 합산 공식:**

```
color_score  = 0.20×chroma + 0.25×range + 0.30×entropy + 0.25×texture
artifact_score = 0.60×edge + 0.40×hf
sharpness_score = 0.50×laplacian + 0.50×edge_sharpness

suitability_score = 0.40×color + 0.30×artifact + 0.30×sharpness
```

**클래스별 통계:** 평균, 표준편차, P10/P25/P50/P75/P90 백분위수

**출력:** `metadata.json` (suitability_score 인플레이스 업데이트), `quality_stats.json` (클래스별 통계 + 프루닝 임계값 권장)

---

### Step 10 — `validate_augmented_quality.py`

**5지표 가중 검증 (합산 ≥ 0.70 → 통과):**

| 지표 | 가중치 | 내용 |
|------|--------|------|
| 흐림 점수 | 20% | Laplacian 분산 (≥100 → 선명) |
| 아티팩트 점수 | 20% | 고기울기 비율 (블렌딩 경계 탐지) |
| 색상 일관성 | 15% | LAB 공간 범위 (L_mean 30–200, a/b ±50 이내) |
| 결함 지표 일관성 | 25% | 마스크 분석 → 서브타입 분류 → 예상값 비교 |
| 결함 존재 비율 | 20% | 결함 면적 비율 0.1%–30% |

---

### Step 11 — `run_fid.py`

**FID 공식:**
```
FID = ||μ_real - μ_syn||² + Tr(Σ_real + Σ_syn - 2√(Σ_real·Σ_syn))
```
InceptionV3 특징 벡터 기반, 클래스별·서브타입별·cross(class×subtype) 세분화

**두 가지 평가 수준:**
- **FID-ROI:** 실제 512×512 ROI vs ControlNet 생성 이미지 (클래스당 최소 50개 필요)
- **FID-Composed:** 실제 1600×256 전체 이미지 vs Poisson Blending 합성 이미지 (suitability top-k 프루닝 후)

**최적화:** InceptionV3 특징 디스크 캐싱, DataLoader 병렬화 (`workers=12`), 배치 크기 64

---

### Step 12 — `run_split_experiment.py`

**실험 흐름:**
```
"60/20/20,70/15/15,80/10/10" 파싱
→ 각 비율별:
    1. create_dataset_split.py → split CSV 생성
    2. run_benchmark.py (DeepLabV3+ 단독, 100 에폭)
    3. test_metrics 수집 (Dice, IoU, 클래스별)
→ 비교 표 생성 (Split | 샘플 수 | Dice | 최적 에폭)
→ 최적 분할 비율 권장
```

**기준 모델:** DeepLabV3+ (분할 모델이 비율 변화에 더 민감)

---

### Steps 13–14 — `run_benchmark.py`

**모델 라우팅:**
- Detection (YOLO-MFD, EB-YOLOv8) → `UltralyticsTrainer`
- Segmentation (DeepLabV3+) → `BenchmarkTrainer`

**CASDA 주입 전략:** `casda_*` 파일을 `baseline_raw`에 주입 → 학습 → 정리 (데이터 중복 방지)

**가설 검정 (H3–H6):**
- H3: 아키텍처 독립성 — 모델 간 증강 효과 일관성
- H4: 소수 클래스(Class 2) 성능 향상
- H5: CASDA FID < CopyPaste FID
- H6: 최적 합성:실제 비율 존재

**재개 지원:** `experiment_meta.json` 완료 여부 확인 → 완료된 조합 건너뜀

---

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
