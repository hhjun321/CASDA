# 10 — Stage 0 재분석 가이드 (임계값 재도출 + 타입 구조 검증)

> **Claude 요약:** 새 임계값(Percentile P80/P20 분리 도출)으로 `analyze_dataset.py` 재실행 후, 배경 타입 축소 여부와 결함 서브타입 GMM 군집 수를 데이터로 결정하는 4단계 Colab 가이드.

---

## 목적

Stage 0 초기 실행 결과에서 발견된 두 가지 문제를 해결한다:

1. **HIGH/LOW 동일값 버그** — `LOW_ASPECT_RATIO = HIGH_ASPECT_RATIO = 8.2034` (Otsu 단일값 공유). `derive_high_low()` 교체 후 재실행으로 해소.
2. **타입 구조 미검증** — `textured` 배경(0.5%), `irregular` 서브타입(405개) 존재 의미 불명확. 재분류 결과 보고 타입 축소 여부 결정.

---

## 사전 조건

| 항목 | 확인 |
|------|------|
| `analyze_dataset.py` 수정 완료 (`derive_high_low` 포함) | ✓ 코드 반영됨 |
| Colab Pro 세션 (`approve` 브랜치 클론) | 필요 |
| `$DRIVE/analysis/` 기존 결과 존재 | 덮어쓰기 허용 |

---

## Step 0 — 환경 준비 및 사전 검증

```python
# sklearn 설치 확인 (Colab 기본 포함이나 명시 보장)
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    print("sklearn OK")
except ImportError:
    !pip install scikit-learn -q
    print("sklearn 설치 완료 — 런타임 재시작 불필요")

import os, json, yaml, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['ANALYSIS_DIR']    = f"{os.environ['DRIVE']}/analysis"
os.environ['ANALYSIS_CONFIG'] = f"{os.environ['DRIVE']}/analysis/recommended_config.yaml"
```

---

## Step 1 — `analyze_dataset.py` 재실행

### 소규모 선검증 (200장, ~3분)

```python
!python $SCRIPTS/analyze_dataset.py \
    --image_dir  $TRAIN_IMAGES \
    --train_csv  $TRAIN_CSV \
    --output_dir $DRIVE/analysis_test \
    --max_images 200 \
    --num_workers 4
```

선검증 완료 후 assert 확인:

```python
with open(f"{os.environ['DRIVE']}/analysis_test/threshold_recommendations.json") as f:
    rec = json.load(f)

high_ar = rec['HIGH_ASPECT_RATIO']['recommended']
low_ar  = rec['LOW_ASPECT_RATIO']['recommended']
high_sol = rec['HIGH_SOLIDITY']['recommended']
low_sol  = rec['LOW_SOLIDITY']['recommended']

print(f"HIGH_ASPECT_RATIO : {high_ar}  |  LOW_ASPECT_RATIO : {low_ar}")
print(f"HIGH_SOLIDITY     : {high_sol}  |  LOW_SOLIDITY     : {low_sol}")

ok = (high_ar > low_ar) and (high_sol > low_sol)
print(f"\n[판단 0] HIGH/LOW 역전 버그 해소: {'✓ OK' if ok else '✗ 여전히 역전 — 코드 확인 필요'}")
```

### 전체 실행 (~30분)

```python
!python $SCRIPTS/analyze_dataset.py \
    --image_dir  $TRAIN_IMAGES \
    --train_csv  $TRAIN_CSV \
    --output_dir $DRIVE/analysis \
    --num_workers 8
```

---

## Step 2 — 배경 타입 분포 재확인 및 축소 판단

```python
with open(f"{os.environ['ANALYSIS_DIR']}/class_distribution.json") as f:
    dist = json.load(f)

bg_counts = dist['bg_type_counts']
total = sum(bg_counts.values())

print("배경 타입 분포 (재분류 후):")
print(f"  {'타입':<22} {'셀 수':>8}  {'비율':>6}")
print("  " + "-" * 40)
for k, v in sorted(bg_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:<22} {v:>8,}  {v/total*100:>5.1f}%")

textured_ratio = bg_counts.get('textured', 0) / total
v_ratio = bg_counts.get('vertical_stripe', 0) / total
h_ratio = bg_counts.get('horizontal_stripe', 0) / total
stripe_ratio = v_ratio + h_ratio

print(f"\n[판단 1] textured 비율: {textured_ratio*100:.2f}%")
if textured_ratio < 0.02:
    print("  → textured 제거 권장 (complex_pattern 흡수) — 4타입으로 축소")
else:
    print("  → textured 유지 (2% 이상, 통계적 의미 있음)")

print(f"\n[판단 2] v_stripe + h_stripe 합계: {stripe_ratio*100:.1f}%  "
      f"(v={v_ratio*100:.1f}%, h={h_ratio*100:.1f}%)")
if stripe_ratio < 0.05:
    print("  → stripe 통합 검토 (합계 5% 미만) — 3타입으로 추가 축소 가능")
elif abs(v_ratio - h_ratio) / (stripe_ratio + 1e-9) < 0.1:
    print("  → v/h 비율 유사 — stripe 통합 검토 (방향성 신호 약함)")
else:
    print(f"  → stripe 분리 유지 (v/h 비율 차이 명확: {v_ratio/h_ratio:.2f}x)")
```

### 배경 타입 결정 기준

| 조건 | 결정 |
|------|------|
| `textured < 2%` | 5타입 → 4타입 (`textured` 제거) |
| `textured < 2%` + `stripe 합계 < 5%` | 5타입 → 3타입 (`smooth` / `stripe` / `complex`) |
| `textured ≥ 2%` | 5타입 유지 |

> **주의**: v_stripe vs h_stripe 방향성 차이(`linear_scratch` 공출현 비율에서 v=0.22, h=0.07)가 있으면 stripe 분리 유지를 권장한다.

---

## Step 3 — 결함 서브타입 GMM 클러스터링

```python
morph_df = pd.read_csv(f"{os.environ['ANALYSIS_DIR']}/morphological_features.csv")
features = morph_df[['linearity', 'aspect_ratio', 'solidity']].dropna()
X = StandardScaler().fit_transform(features)

print(f"사용 인스턴스: {len(features):,}개")
print(f"\n{'K':>3}  {'silhouette':>12}  {'BIC':>14}  판정")
print("-" * 45)

scores, bics = {}, {}
for k in [2, 3, 4]:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    labels = gmm.fit_predict(X)
    scores[k] = silhouette_score(X, labels)
    bics[k]   = gmm.bic(X)
    best_flag = ""
    print(f"  {k}  {scores[k]:>12.4f}  {bics[k]:>14.1f}  {best_flag}")

# silhouette 1차, 차이 < 0.02이면 BIC 최소로 타이브레이커
best_sil_k = max(scores, key=scores.get)
sil_vals = sorted(scores.values(), reverse=True)
if len(sil_vals) >= 2 and (sil_vals[0] - sil_vals[1]) < 0.02:
    best_k = min(bics, key=bics.get)
    reason = f"silhouette 차이 {sil_vals[0]-sil_vals[1]:.4f} < 0.02 — BIC 최소 K={best_k}"
else:
    best_k = best_sil_k
    reason = f"silhouette 최대 K={best_k}"

print(f"\n[판단 3] best_k = {best_k}  ({reason})")
if best_k <= 3:
    print("  → 3종 확정 권장: irregular → general 흡수")
    print("  → BG_TYPES/SUBTYPES 상수 업데이트 및 compatibility matrix 재산출 필요")
else:
    print("  → 4종 유지 검토 (irregular 독립 군집 존재)")
    print("  → irregular 샘플 부족(~405개) 감안, Step 4 시각화로 군집 해석 필요")
```

### Step 3b — 군집 시각화 저장

```python
gmm_final = GaussianMixture(n_components=best_k, random_state=42, n_init=5).fit(X)
labels = gmm_final.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# linearity vs aspect_ratio
axes[0].scatter(features['linearity'], features['aspect_ratio'],
                c=labels, cmap='tab10', alpha=0.4, s=8)
axes[0].set_xlabel('linearity'); axes[0].set_ylabel('aspect_ratio')
axes[0].set_title(f'GMM K={best_k}: linearity vs aspect_ratio')

# linearity vs solidity
axes[1].scatter(features['linearity'], features['solidity'],
                c=labels, cmap='tab10', alpha=0.4, s=8)
axes[1].set_xlabel('linearity'); axes[1].set_ylabel('solidity')
axes[1].set_title(f'GMM K={best_k}: linearity vs solidity')

plt.tight_layout()
save_path = f"{os.environ['ANALYSIS_DIR']}/figures/fig11_gmm_subtype_clusters.png"
plt.savefig(save_path, dpi=120)
print(f"저장: {save_path}")
plt.show()

# 군집별 특성 평균 출력
feat_with_label = features.copy()
feat_with_label['cluster'] = labels
print("\n군집별 특성 평균:")
print(feat_with_label.groupby('cluster').mean().round(3).to_string())
```

---

## Step 4 — 판단 결과 정리 및 다음 액션

Step 2~3 결과를 바탕으로 다음 중 해당하는 항목을 진행한다.

### 배경 타입 변경 시

| 결정 | 변경 대상 |
|------|---------|
| 5 → 4타입 (`textured` 제거) | `analyze_dataset.py:BG_TYPES`, `background_characterization.py` 분류 로직, `subtype_bg_matrix` 차원 |
| 5 → 3타입 (stripe 통합) | 위 + `vertical_stripe`/`horizontal_stripe` → `stripe` 병합 |

### 서브타입 변경 시

| 결정 | 변경 대상 |
|------|---------|
| 4 → 3타입 (`irregular` → `general`) | `analyze_dataset.py:SUBTYPES`, `defect_characterization.py` 분류 분기, `roi_suitability.py:MATCHING_RULES` |

### 변경 없음 시

- `recommended_config.yaml` Stage A~C에 전달
- `subtype_compatibility_matrix` → `roi_suitability.py:MATCHING_RULES` 교체

---

## 관련 노트

- [[00-INDEX]] — 경로 변수 전체 목록
- [[02-Pipeline-StageA]] — Stage A 스크립트, `--config` 인자 전달 방법
- [[06-Scripts-Reference]] — `analyze_dataset.py` 입출력 전체 매핑
