# CASDA 개선 연구 계획

> **기본 방침**: 전체 실험을 처음부터 재수행. 현재 하드코딩된 모든 임계값·비율을 Stage 0 데이터 분석 결과로 대체.  
> **총 하드코딩 교체 대상**: 19개 값 (파라미터 7 + DefectCharacterizer 6 + BackgroundAnalyzer 6)

---

## 수행 순서

```
Stage 0  → Stage A → Stage B → Stage C → Stage D
데이터분석  전처리     생성       후처리     평가

이후:
  합성 비율 탐색 (synthetic_ratio grid search)
  선택 편향 수정 (run_split_experiment.py)
  통계적 유의성 (3 seed × 전체 조건)
  SOTA Baseline (별도 결정)
```

---

## Stage 0 (신규): 데이터셋 분석

> 파이프라인 전체의 기반. 여기서 도출한 값이 Stage A~C 모든 파라미터를 결정한다.

### 신규 스크립트: `analyze_dataset.py`

**입력**: `TRAIN_IMAGES`, `TRAIN_CSV`  
**출력**: `ANALYSIS_DIR/`

| 출력 파일 | 내용 |
|-----------|------|
| `class_distribution.json` | 클래스별 결함 인스턴스 수, 이미지당 빈도 |
| `morphological_features.csv` | 전체 결함 인스턴스 × 4개 지표 (linearity, solidity, aspect_ratio, fill_ratio) |
| `background_features.csv` | 전체 이미지 패치 × 배경 특성 (variance, edge_density, high_freq_ratio 등) |
| `defect_bg_matrix_4x5.json` | 4×5 공출현 매트릭스 (아래 구조 참조) |
| `threshold_recommendations.json` | 분포 분석 기반 임계값 후보 (히스토그램 valley / 백분위수) |
| `recommended_config.yaml` | Stage A~C 파라미터 권장값 — 사람이 검토 후 확정 |

### 4×5 매트릭스

행: Class 1 / 2 / 3 / 4  
열: smooth / textured / vertical_stripe / horizontal_stripe / complex_pattern

```json
{
  "count_real": 실제 공출현 수,
  "target_synthetic": 권장 생성 목표 수 (imbalance 보정),
  "compatibility_score": 분포 기반 재산정값 (기존 1.0~0.2 하드코딩 대체)
}
```

### 교체 대상 임계값 전체 목록

#### 파이프라인 파라미터

| 위치 | 현재값 | 대체 방법 |
|------|--------|----------|
| `extract_rois.py --min_suitability` | 0.5 | suitability 분포 P25 |
| `prepare_controlnet_data.py --per_class_cap` | 1200 | 클래스 분포 기반 상한 재산정 |
| `prepare_controlnet_data.py --rare_class_threshold` | 200 | 클래스 분포 P10 |
| `test_controlnet.py --num_images_per_class` | `{"1":2,"2":10,"3":1,"4":2}` | 4×5 매트릭스 imbalance 비율에서 산출 |
| `compose_casda_images.py --compositions-per-roi` | 5 | 셀별 target_synthetic ÷ ROI 수 |
| `validate_augmented_quality.py --min_quality_score` | 0.7 | quality 분포 P25 |
| `compose_casda_images.py` 호환성 매트릭스 | 1.0~0.2 하드코딩 | 4×5 공출현 비율로 재산정 |

#### `DefectCharacterizer.classify_defect_subtype()` — `src/analysis/defect_characterization.py:213`

| 상수 | 현재값 | 용도 | 대체 방법 |
|------|--------|------|----------|
| `HIGH_LINEARITY` | 0.85 | `linear_scratch` 판단 | linearity 분포 히스토그램 valley |
| `HIGH_ASPECT_RATIO` | 5.0 | `linear_scratch`, `elongated` 판단 | aspect_ratio 분포 P75 또는 valley |
| `LOW_ASPECT_RATIO` | 2.0 | `compact_blob` 판단 | aspect_ratio 분포 P25 |
| `HIGH_SOLIDITY` | 0.9 | `compact_blob` 판단 | solidity 분포 P75 |
| `LOW_SOLIDITY` | 0.7 | `irregular` 판단 | solidity 분포 P25 |
| (인라인 0.6) | 0.6 | `elongated` linearity 하한 | linearity 분포 P50 |

#### `BackgroundAnalyzer.classify_patch()` — `src/analysis/background_characterization.py:125`

| 임계값 | 현재값 | 역할 | 대체 방법 |
|--------|--------|------|----------|
| `variance_threshold` | 100.0 | SMOOTH vs 나머지 | 전체 패치 variance 분포 valley |
| `edge_threshold` | 0.3 | v_ratio / h_ratio 지배 방향 조건 | v_ratio·h_ratio 분포 P75 |
| `total_strength < 1.0` | 1.0 | 약한 엣지 → TEXTURED 조기 판단 | edge strength 분포 P10 |
| `v_ratio > h_ratio * 1.5` | 1.5 | VERTICAL_STRIPE 지배 비율 | ratio 분포에서 적합 배율 탐색 |
| `h_ratio > v_ratio * 1.5` | 1.5 | HORIZONTAL_STRIPE 지배 비율 | 동일 |
| `high_freq_ratio > 0.3` | 0.3 | COMPLEX_PATTERN vs TEXTURED | FFT high_freq_ratio 분포 valley |

### 워크플로우

```
analyze_dataset.py 실행
  → threshold_recommendations.json + 히스토그램 시각화 검토
  → 사람이 recommended_config.yaml 수치 확정
  → Stage A~C 스크립트에 확정값 전달
```

---

## Stage A 개선: 데이터 기반 파라미터 적용

- `extract_rois.py`: Stage 0 도출 `min_suitability` 적용
- `prepare_controlnet_data.py`: `per_class_cap`, `rare_class_threshold` Stage 0 도출값 적용
- `DefectCharacterizer.classify_defect_subtype()`: 6개 상수 → `threshold_recommendations.json` 읽어 동적 적용
- `BackgroundAnalyzer.__init__()`: `variance_threshold`, `edge_threshold` → `recommended_config.yaml`에서 로드
- `BackgroundAnalyzer.classify_patch()`: 나머지 4개 임계값 동적 적용

---

## Stage B 개선: 생성 수량 데이터 기반화

- `test_controlnet.py --num_images_per_class`: 4×5 매트릭스 `target_synthetic` 합계에서 자동 산출
- (선택) 희귀 (class, bg_type) 조합 hint 우선 선택 전략

---

## Stage C 개선: 합성 전략 데이터 기반화

- `compose_casda_images.py` 호환성 매트릭스: 4×5 공출현 비율로 대체
- `--compositions-per-roi`: 셀별 target_synthetic ÷ ROI 수에서 역산
- `validate_augmented_quality.py --min_quality_score`: Stage 0 도출 임계값 적용

---

## Stage D 개선: 품질 평가 지표 확장

현재 FID만 부분 도입. 추가:

| 지표 | 측정 내용 | 측정 단위 |
|------|----------|---------|
| **KID** | 소규모 데이터셋에서 FID보다 분산 안정적 | 분포 수준 |
| **LPIPS** | image-pair 수준 perceptual similarity | 이미지 쌍 수준 |
| **Mask-to-defect alignment (IoU)** | GT mask(hint R채널) vs 생성 이미지 실제 결함 위치 일치도 | 이미지 쌍 수준 |

> Mask-IoU: FID가 높아도 Mask-IoU 낮으면 학습 레이블 오염 → benchmark 성능 저하 원인 미포착.

---

## 합성 비율 체계적 탐색

> 리뷰어 질의: "42.7% 합성 비율의 과적합 영향은? 최적 비율 결정 근거는?"

**현재 문제**: `top_k=2500`은 `benchmark_experiment.yaml`에 하드코딩. 42.7%는 임의값에서 산출된 수치이며, 비율 탐색·과적합 분석 없음.

**개선**:
- Stage 0의 4×5 매트릭스로 클래스별 imbalance 파악 → top_k 초기값 산정
- synthetic_ratio 10~70% (5% 간격) × 3 seed → 비율별 val Dice 곡선 제시
- 과적합 지표: train/val loss gap을 비율별로 비교

---

## 선택 편향 수정

> 리뷰어 질의: "테스트 세트 성능이 70/15/15 선택에 영향을 미쳤다면 선택 편향이다."

**현재 코드 문제** (`run_split_experiment.py:372`):
```python
# split 비율 선택 기준 = test Dice ← 선택 편향
best = max(results, key=lambda r: r.get('dice_mean', 0.0))
# dice_mean = seg_result['test_metrics']['dice_mean']
```

| 단계 | 사용 셋 | 문제 |
|------|--------|------|
| best epoch 선택 | val_loader | 정상 ✓ |
| split 비율 선택 | **test_metrics** | 선택 편향 ✗ |
| 최종 성능 보고 | 동일 test set | 오염 ✗ |

**수정**:

`collect_results()` — `best_metric` 추가 수집 (`benchmark_results.json`에 이미 저장됨):
```python
'val_best_metric': seg_result.get('best_metric', 0.0),  # val Dice at best epoch
```

`compare_splits()` — 선택 기준 변경:
```python
# 변경 전
best = max(results, key=lambda r: r.get('dice_mean', 0.0))
# 변경 후
best = max(results, key=lambda r: r.get('val_best_metric', 0.0))
```

> `best_metric`은 `trainer.py:434`에서 val_loader 기준 갱신 → 추가 파일 파싱 불필요.

---

## 통계적 유의성 확보

- 모든 핵심 비교 조건에 최소 **3 random seed** 적용 → mean ± std 보고
- 대상: Stage 0 파라미터 확정 후 전체 파이프라인 재실행 시 seed 추가
- **비용**: 주요 조건 × 3 seed ≈ 27+ run

---

## SOTA Baseline 비교

- 현재: Raw, Traditional Augmentation, Copy-Paste만
- 추가 필요: SOTA GAN 기반(DefectGAN 등), diffusion 기반 방법
- **개선**: 최소 1~2개 SOTA generative baseline 동일 조건 학습·평가. FID 공통 realism metric으로 비교
- *(구현 현실성 추후 결정)*

---

## 극단적 클래스 결핍 검증

- Class 2가 현재 최소 클래스. <50개 시나리오 미테스트
- Stage 0 분석 후 실제 Class 2 샘플 수 확인 → 4×5 매트릭스에서 희귀 셀 식별
- **개선**: few-shot conditioning 전략 (소수 샘플 증강 후 ControlNet fine-tuning)
