# CASDA 증강 데이터셋 비율 구성

> AROMA 연구에서 custom 예정. 본 문서는 CASDA 파이프라인의 합성 데이터 비율 설계를 정리한다.

---

## 1. 비율 정의

`casda_ratio`는 **전체 학습 데이터 대비 합성 이미지 비율**이다. 클래스별 비율이 아니다.

```
ratio = 합성 수 / (원본 수 + 합성 수)
max_samples = 원본_train수 × ratio / (1 − ratio)
```

원본 train 수 기준 예시 (Severstal ≈ 4,666장):

| ratio | max_samples (합성) | 전체 학습 수 |
|-------|--------------------|-------------|
| 10%   | ≈ 518              | ≈ 5,184     |
| 20%   | ≈ 1,167            | ≈ 5,833     |
| 30%   | ≈ 2,000            | ≈ 6,666     |
| 50%   | ≈ 4,666            | ≈ 9,332     |

---

## 2. 비율 실험 그룹 (`casda_ratio_*`)

H6 가설 검증용. 최적 합성:실제 비율 존재 여부를 탐색한다.

### 선택 방식: 전체 global top-k

- `suitability_score` 내림차순 정렬 후 상위 `max_samples`개 선택
- **클래스 비율 보정 없음** (`stratified=False`)
- 품질이 높은 클래스의 이미지가 더 많이 선택될 수 있음

### 실행 방법

```bash
# CLI
python scripts/run_benchmark.py \
  --casda-ratio 0.1 0.2 0.3 0.5 \
  --casda-ratio-source composed \
  ...

# YAML (benchmark_experiment.yaml)
casda_ratio:
  enabled: true
  source: "composed"
  ratios: [0.10, 0.20, 0.30, 0.50]
  include_baseline: true
```

### 생성되는 그룹 키

| 그룹 키           | ratio | max_samples |
|------------------|-------|-------------|
| `casda_ratio_10` | 10%   | 원본 × 0.1/0.9 |
| `casda_ratio_20` | 20%   | 원본 × 0.2/0.8 |
| `casda_ratio_30` | 30%   | 원본 × 0.3/0.7 |
| `casda_ratio_50` | 50%   | 원본 × 0.5/0.5 |

---

## 3. 클래스별 비율 보정 (`stratified top-k`)

`casda_ratio_*` 그룹은 stratified를 사용하지 않는다. 클래스 비율을 유지하는 방식은 고정 그룹에서만 사용된다.

### stratified top-k 알고리즘 (`dataset_yolo.py`)

```
(a) 전체 샘플을 class_id별 그룹화
(b) 각 클래스의 비율에 따라 할당량 계산 (최소 1개 보장)
    quota[cls] = max(1, round(cls_count / total * k))
(c) 합계 > k → 가장 큰 그룹에서 1씩 감소
(d) 합계 < k → 여유 있는 그룹에 1씩 추가
(e) 각 그룹 내 suitability_score 내림차순 → 할당량만큼 선택
```

### stratified 적용 현황

| 그룹                      | stratified | top_k  | 설명                         |
|--------------------------|------------|--------|------------------------------|
| `casda_ratio_*`          | **False**  | 비율 기반 자동 계산 | 전체 global top-k          |
| `casda_composed_pruning` | **True**   | 2,500  | 클래스 비율 유지하며 상위 선택 |
| `ablation_no_blending`   | **True**   | 2,500  | Blending 제거 ablation      |
| `ablation_no_pruning`    | False      | 5,000  | Pruning 제거 ablation       |
| `casda_composed`         | -          | 전체   | 필터링 없음                  |
| `copypaste`              | -          | 전체   | ControlNet 없는 베이스라인   |

---

## 4. AROMA 커스터마이징 포인트

아래 항목을 연구 설계에 맞게 수정한다.

### 4-1. ratio 범위 조정

```yaml
# benchmark_experiment.yaml
casda_ratio:
  enabled: true
  ratios: [0.10, 0.20, 0.30, 0.50]  # ← 탐색할 비율 목록 수정
```

### 4-2. ratio 그룹에 stratified 적용

현재 ratio 그룹은 `stratified=False`로 고정된다 (`run_benchmark.py:1642`).  
클래스별 균등 비율을 원하면 해당 로직을 수정한다:

```python
# run_benchmark.py:1644-1648 수정 예시
if group_key in casda_ratio_map:
    _, ratio_max_samples = casda_ratio_map[group_key]
    suitability_thresh = 0.0
    use_stratified = True  # False → True 로 변경
```

### 4-3. 원본 데이터 수 기준 변경

현재 `num_train_original`은 70/15/15 split의 train 수로 자동 계산된다.  
다른 split 비율 사용 시 `benchmark_experiment.yaml`의 `split` 섹션을 수정한다:

```yaml
dataset:
  split:
    train_ratio: 0.7   # ← 변경
    val_ratio: 0.15
    test_ratio: 0.15
```

### 4-4. 데이터 소스 선택

```bash
--casda-ratio-source composed   # Poisson Blending 합성 이미지 (권장)
--casda-ratio-source full       # ControlNet 생성 ROI 원본
```

---

## 5. 설계 요약

```
casda_ratio_*   : ratio = 합성/(원본+합성), 전체 global top-k, stratified=False
고정 pruning 그룹 : top_k 고정, stratified=True (클래스 비율 보정)
```

클래스 불균형이 심한 데이터셋(예: AROMA)에서 ratio 실험 시,  
`stratified=True`를 함께 적용해 희귀 클래스가 ratio 증가에 비례해 포함되도록 설계할 것을 권장한다.
