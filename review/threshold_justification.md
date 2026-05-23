# Threshold Justification — 임계값 결정의 이론적 근거
> **목적:** Reviewer 지적 "Q > 0.7, 분류 임계값의 이론적 근거는?" 에 대한 데이터 수집 및 논문 서술 근거 마련  
> **브랜치:** `feature/statistical-robustness` (robustness_experiments.md와 동일 브랜치)

---

## 대상 임계값 목록

| 임계값 | 값 | 위치 |
|--------|-----|------|
| Q-score 프루닝 | ≥ 0.7 | `validate_augmented_quality.py` |
| Q 가중치 (color) | 0.40 | `score_casda_quality.py` |
| Q 가중치 (artifact) | 0.30 | `score_casda_quality.py` |
| Q 가중치 (blur) | 0.30 | `score_casda_quality.py` |
| 배경: smooth | σ² < 200 AND ed < 2% | `background_characterization.py` |
| 배경: vertical_stripe | Sobel_x 비율 > 65% | `background_characterization.py` |
| 배경: horizontal_stripe | Sobel_y 비율 > 65% | `background_characterization.py` |
| 배경: complex | ed > 15% | `background_characterization.py` |
| 배경: textured | σ² > 500 OR ed > 5% | `background_characterization.py` |
| 결함 서브타입: θ_e | elongation 기준 | `defect_characterization.py` |
| 결함 서브타입: θ_s | solidity 기준 | `defect_characterization.py` |

---

## 분석 1 — Q-score 분포 및 0.7 근거

### 목적

전체 합성 이미지의 Q 점수 분포에서 0.7이 **자연 분리점(natural valley)** 임을 확인한다.
Valley가 존재하면 → *"자연 분리점으로 결정"* 서술 가능.
Valley가 불분명하면 → 민감도 분석(분석 2)으로 보완.

### Colab 스크립트

```python
# ─────────────────────────────────────────────────────────────
# [분석 1-A] Q 점수 히스토그램 + 누적 분포
# 실행: Colab 셀에 붙여넣기
# ─────────────────────────────────────────────────────────────
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

META_COMPOSED = Path("/content/drive/MyDrive/data/Severstal/augmented_dataset/casda_composed/metadata.json")
OUT_DIR       = Path("/content/CASDA/review/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(META_COMPOSED) as f:
    meta = json.load(f)

scores = np.array([e["suitability_score"] for e in meta
                   if e.get("suitability_score") is not None
                   and e["suitability_score"] > 0])

print(f"Total samples : {len(scores)}")
print(f"Mean ± Std    : {scores.mean():.3f} ± {scores.std():.3f}")
print(f"Median        : {np.median(scores):.3f}")
print(f"Q ≥ 0.7 ratio : {(scores >= 0.7).mean()*100:.1f}%  ({(scores >= 0.7).sum()} samples)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white")

# 히스토그램
ax = axes[0]
ax.hist(scores, bins=60, color="#1565C0", alpha=0.75, edgecolor="white", linewidth=0.4)
ax.axvline(0.7, color="#C62828", linewidth=2.0, linestyle="--", label="threshold = 0.7")
ax.set_xlabel("Q score", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Q-score Distribution", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# 누적 분포 (ECDF)
ax2 = axes[1]
sorted_s = np.sort(scores)
ecdf     = np.arange(1, len(sorted_s)+1) / len(sorted_s)
ax2.plot(sorted_s, ecdf, color="#1565C0", linewidth=1.8)
ax2.axvline(0.7, color="#C62828", linewidth=2.0, linestyle="--", label="threshold = 0.7")
pct_above = (scores >= 0.7).mean() * 100
ax2.axhline(1 - pct_above/100, color="#C62828", linewidth=1.0, linestyle=":")
ax2.text(0.71, 1 - pct_above/100 + 0.02, f"{pct_above:.1f}% retained",
         color="#C62828", fontsize=10)
ax2.set_xlabel("Q score", fontsize=12)
ax2.set_ylabel("Cumulative proportion", fontsize=12)
ax2.set_title("Empirical CDF", fontsize=13, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "q_score_distribution.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved → {OUT_DIR}/q_score_distribution.png")
```

```python
# ─────────────────────────────────────────────────────────────
# [분석 1-B] 클래스별 Q 점수 분포 비교
# ─────────────────────────────────────────────────────────────
CLASS_COLOR = {1: "#2196F3", 2: "#F44336", 3: "#4CAF50", 4: "#FF9800"}

fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor="white", sharey=True)

for ax, cls_id in zip(axes, [1, 2, 3, 4]):
    s = np.array([e["suitability_score"] for e in meta
                  if e.get("class_id") == cls_id
                  and e.get("suitability_score") is not None
                  and e["suitability_score"] > 0])
    ax.hist(s, bins=40, color=CLASS_COLOR[cls_id], alpha=0.75,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0.7, color="#C62828", linewidth=1.8, linestyle="--")
    ax.set_title(f"Class {cls_id}  (n={len(s)})", fontsize=12, fontweight="bold",
                 color=CLASS_COLOR[cls_id])
    ax.set_xlabel("Q score", fontsize=11)
    retained = (s >= 0.7).mean() * 100
    ax.text(0.05, 0.95, f"{retained:.1f}% ≥ 0.7",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(fc="white", alpha=0.7, boxstyle="round,pad=0.3"))
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Count", fontsize=11)
plt.suptitle("Q-score Distribution by Defect Class", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "q_score_by_class.png", dpi=200, bbox_inches="tight")
plt.show()
```

---

## 분석 2 — Q 임계값 민감도 분석 (Elbow Point)

### 목적

Q 임계값을 0.3 ~ 0.9 범위로 변화시키며 **잔류 샘플 수 vs 기대 성능**의 관계를 시각화한다.
Elbow (성능 포화 + 샘플 급감 교차점)가 0.7 부근이면 → *"elbow point로 결정"* 서술 가능.

> 실제 mAP 수치는 P1 multi-seed 실험 완료 후 overlaying 가능.
> 그 전에는 잔류 샘플 수 곡선만으로 1차 시각화한다.

```python
# ─────────────────────────────────────────────────────────────
# [분석 2-A] 임계값별 잔류 샘플 수 곡선
# ─────────────────────────────────────────────────────────────
thresholds = np.arange(0.3, 0.95, 0.05)
retain_counts = [(t, int((scores >= t).sum())) for t in thresholds]

fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
ts, ns = zip(*retain_counts)
ax.plot(ts, ns, "o-", color="#1565C0", linewidth=2.0, markersize=6)
ax.axvline(0.7, color="#C62828", linewidth=2.0, linestyle="--", label="threshold = 0.7")
ax.axhline(ns[list(ts).index(min(ts, key=lambda x: abs(x-0.7)))],
           color="#C62828", linewidth=1.0, linestyle=":", alpha=0.7)
ax.set_xlabel("Q-score threshold", fontsize=12)
ax.set_ylabel("Retained samples", fontsize=12)
ax.set_title("Retained Sample Count vs Q Threshold", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# 각 지점 수치 표기
for t, n in retain_counts:
    ax.annotate(str(n), (t, n), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="#546E7A")

plt.tight_layout()
plt.savefig(OUT_DIR / "q_threshold_sensitivity.png", dpi=200, bbox_inches="tight")
plt.show()

print("\nRetained samples per threshold:")
for t, n in retain_counts:
    bar = "█" * int(n / max(ns) * 40)
    marker = " ← selected" if abs(t - 0.7) < 0.01 else ""
    print(f"  Q ≥ {t:.2f}:  {n:4d}  {bar}{marker}")
```

```python
# ─────────────────────────────────────────────────────────────
# [분석 2-B] P1 결과 완료 후: threshold vs mAP overlay
# (multi-seed 실험 후 수치를 채워서 실행)
# ─────────────────────────────────────────────────────────────
# 아래 map_by_threshold 딕셔너리를 P1 결과로 채운 뒤 실행
map_by_threshold = {
    # 0.3: 0.000,   # ← P1 실험 후 채울 것
    # 0.4: 0.000,
    # 0.5: 0.000,
    # 0.6: 0.000,
    # 0.7: 0.000,
    # 0.8: 0.000,
    # 0.9: 0.000,
}

if map_by_threshold:
    fig, ax1 = plt.subplots(figsize=(9, 4.5), facecolor="white")
    ax2 = ax1.twinx()

    ts_m = sorted(map_by_threshold.keys())
    maps = [map_by_threshold[t] for t in ts_m]
    ns_m = [(scores >= t).sum() for t in ts_m]

    ax1.plot(ts_m, maps, "s-", color="#1565C0", linewidth=2.0,
             markersize=7, label="mAP@0.5")
    ax2.plot(ts_m, ns_m, "o--", color="#78909C", linewidth=1.5,
             markersize=5, label="Retained samples")

    ax1.axvline(0.7, color="#C62828", linewidth=2.0, linestyle="--",
                label="threshold = 0.7")
    ax1.set_xlabel("Q-score threshold", fontsize=12)
    ax1.set_ylabel("mAP@0.5 (YOLO-MFD)", fontsize=12, color="#1565C0")
    ax2.set_ylabel("Retained samples", fontsize=12, color="#78909C")
    ax1.set_title("Detection Performance vs Q Threshold (Elbow Analysis)",
                  fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower left")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "q_threshold_elbow.png", dpi=200, bbox_inches="tight")
    plt.show()
```

---

## 분석 3 — 배경 분류 임계값 근거

### 목적

Severstal 전체 학습 이미지의 σ², edge density, Sobel 방향 비율 분포를 측정하여
각 임계값(200, 2%, 5%, 15%, 65%)이 **실제 데이터 분포 내에서 어디에 위치하는지** 확인한다.

```python
# ─────────────────────────────────────────────────────────────
# [분석 3-A] 전체 이미지 배경 통계 측정
# 시간 소요: ~10분 (1만 장 기준, workers 활용)
# ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json

TRAIN_IMAGES = Path("/content/drive/MyDrive/data/Severstal/train_images")
ROI_META     = Path("/content/drive/MyDrive/data/Severstal/roi_patches_v5.1/roi_metadata.csv")
OUT_DIR      = Path("/content/CASDA/review/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_roi = pd.read_csv(ROI_META, sep=None, engine="python")
df_roi.columns = df_roi.columns.str.strip().str.lower().str.replace(" ", "_")

def _measure_bg_stats(img_path_str):
    img = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    patch = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)

    var  = float(np.var(patch))
    edges = cv2.Canny(patch.astype(np.uint8), 50, 150)
    ed   = float(np.count_nonzero(edges) / edges.size)

    sx   = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sy   = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    ex, ey = float(np.mean(np.abs(sx))), float(np.mean(np.abs(sy)))
    tot  = ex + ey + 1e-7
    sx_ratio = ex / tot

    return {"path": img_path_str, "var": var, "ed": ed, "sx_ratio": sx_ratio}

img_paths = sorted(TRAIN_IMAGES.glob("*.jpg"))[:5000]  # 5천 장 샘플링
print(f"Measuring stats for {len(img_paths)} images ...")

results = []
with ProcessPoolExecutor(max_workers=8) as pool:
    for r in pool.map(_measure_bg_stats, [str(p) for p in img_paths]):
        if r:
            results.append(r)

df_bg = pd.DataFrame(results)
df_bg.to_csv(OUT_DIR / "bg_stats.csv", index=False)
print(f"Saved {len(df_bg)} rows → bg_stats.csv")
print(df_bg[["var", "ed", "sx_ratio"]].describe())
```

```python
# ─────────────────────────────────────────────────────────────
# [분석 3-B] 배경 분류 임계값 위치 시각화
# ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/content/CASDA/review/figures")
df_bg   = pd.read_csv(OUT_DIR / "bg_stats.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")

specs = [
    ("var",      "Pixel Variance (σ²)",
     [(200, "#1565C0", "smooth / other"),
      (500, "#2E7D32", "other / textured")]),
    ("ed",       "Edge Density",
     [(0.02, "#1565C0", "smooth threshold"),
      (0.05, "#2E7D32", "textured threshold"),
      (0.15, "#C62828", "complex threshold")]),
    ("sx_ratio", "Sobel-X Ratio  (Sx / (Sx+Sy))",
     [(0.65, "#C62828", "vertical_stripe threshold")]),
]

for ax, (col, xlabel, vlines) in zip(axes, specs):
    data = df_bg[col].dropna().values
    ax.hist(data, bins=80, color="#546E7A", alpha=0.6,
            edgecolor="white", linewidth=0.3)
    for val, color, label in vlines:
        ax.axvline(val, color=color, linewidth=2.0, linestyle="--", label=f"{label} ({val})")
        pct = (data < val).mean() * 100
        ax.text(val * 1.02, ax.get_ylim()[1] * 0.85,
                f"{pct:.0f}%tile", color=color, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(xlabel, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Background Classification Threshold Positions in Data Distribution",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "bg_threshold_distribution.png", dpi=200, bbox_inches="tight")
plt.show()
```

---

## 분석 4 — 결함 서브타입 θ_e, θ_s 근거

### 목적

roi_metadata.csv의 실제 elongation, solidity 분포에서 **쌍봉 분리점(bimodal valley)** 을 확인한다.
분리점이 θ_e, θ_s와 일치하면 → *"분포의 자연 분리점으로 결정"* 서술 가능.

```python
# ─────────────────────────────────────────────────────────────
# [분석 4-A] elongation / solidity 분포 + 분리점 확인
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROI_META = Path("/content/drive/MyDrive/data/Severstal/roi_patches_v5.1/roi_metadata.csv")
OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROI_META, sep=None, engine="python")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

CLASS_COLOR = {1: "#2196F3", 2: "#F44336", 3: "#4CAF50", 4: "#FF9800"}

fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")

for idx, (col, xlabel, thresholds) in enumerate([
    ("elongation", "Elongation",  [("θ_e", None, "#C62828")]),
    ("solidity",   "Solidity",    [("θ_s", None, "#1565C0")]),
]):
    # 전체 분포
    ax_full = axes[0, idx]
    data    = df[col].dropna().values
    ax_full.hist(data, bins=60, color="#546E7A", alpha=0.65,
                 edgecolor="white", linewidth=0.3)

    # θ 값 자동 탐지: KDE valley
    from scipy.signal import find_peaks
    from scipy.stats  import gaussian_kde
    kde_x = np.linspace(data.min(), data.max(), 500)
    kde_y = gaussian_kde(data)(kde_x)
    valleys, _ = find_peaks(-kde_y, prominence=kde_y.max()*0.05)
    for v in valleys:
        ax_full.axvline(kde_x[v], color="#FF7043", linewidth=1.5,
                        linestyle=":", alpha=0.8,
                        label=f"KDE valley ≈ {kde_x[v]:.2f}")

    ax_full.set_title(f"{xlabel} Distribution (all classes)", fontsize=12, fontweight="bold")
    ax_full.set_xlabel(xlabel, fontsize=11)
    ax_full.set_ylabel("Count", fontsize=11)
    ax_full.legend(fontsize=9)
    ax_full.grid(axis="y", alpha=0.3)

    # 클래스별 분포
    ax_cls = axes[1, idx]
    for cls_id in [1, 2, 3, 4]:
        s = df[df["class_id"] == cls_id][col].dropna().values
        if len(s) > 5:
            ax_cls.hist(s, bins=40, alpha=0.45,
                        label=f"Class {cls_id} (n={len(s)})",
                        color=CLASS_COLOR[cls_id], edgecolor="white", linewidth=0.3)

    ax_cls.set_title(f"{xlabel} by Defect Class", fontsize=12, fontweight="bold")
    ax_cls.set_xlabel(xlabel, fontsize=11)
    ax_cls.set_ylabel("Count", fontsize=11)
    ax_cls.legend(fontsize=9)
    ax_cls.grid(axis="y", alpha=0.3)

plt.suptitle("Defect Subtype Threshold Analysis (θ_e, θ_s)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "subtype_threshold_distribution.png", dpi=200, bbox_inches="tight")
plt.show()
```

```python
# ─────────────────────────────────────────────────────────────
# [분석 4-B] 실제 θ_e, θ_s 값 확인 및 적합도 검증
# ─────────────────────────────────────────────────────────────
if "defect_subtype" in df.columns and "elongation" in df.columns:
    print("=== Elongation statistics by defect_subtype ===")
    print(df.groupby("defect_subtype")["elongation"].describe().round(3))
    print()
    print("=== Solidity statistics by defect_subtype ===")
    print(df.groupby("defect_subtype")["solidity"].describe().round(3))
    print()

    # θ_e: linear_scratch vs others의 elongation 분리점
    lin  = df[df["defect_subtype"] == "linear_scratch"]["elongation"].dropna()
    rest = df[df["defect_subtype"] != "linear_scratch"]["elongation"].dropna()
    print(f"linear_scratch  elongation: mean={lin.mean():.3f}, std={lin.std():.3f}")
    print(f"others          elongation: mean={rest.mean():.3f}, std={rest.std():.3f}")

    # θ_s: compact_blob vs irregular의 solidity 분리점
    blob = df[df["defect_subtype"] == "compact_blob"]["solidity"].dropna()
    irr  = df[df["defect_subtype"] == "irregular"]["solidity"].dropna()
    print(f"\ncompact_blob  solidity: mean={blob.mean():.3f}, std={blob.std():.3f}")
    print(f"irregular     solidity: mean={irr.mean():.3f}, std={irr.std():.3f}")
```

---

## 예상 결과 및 논문 서술 전략

### Q > 0.7

| 결과 패턴 | 논문 서술 |
|-----------|-----------|
| 히스토그램에 valley 존재 | *"The threshold Q = 0.7 was selected at the natural valley of the quality score distribution, cleanly separating high-fidelity composites from artifact-prone samples."* |
| Valley 불분명, elbow 확인 | *"Q = 0.7 was identified as the elbow point at which further tightening of the threshold yields diminishing returns in detection performance while substantially reducing the augmented dataset size."* |
| 두 근거 모두 약할 경우 | sensitivity analysis 표를 보여주며 *"Q = 0.7 demonstrated the best balance between data volume and downstream mAP across three model architectures."* |

### 배경 분류 임계값

| 근거 확인 시 | 논문 서술 |
|--------------|-----------|
| 퍼센타일로 정당화 | *"The smooth threshold (σ² < 200) corresponds to the lowest 30th percentile of pixel variance in the Severstal training set, capturing images with near-uniform surface finish."* |

### 결함 서브타입 θ_e, θ_s

| 근거 확인 시 | 논문 서술 |
|--------------|-----------|
| 분포 valley 확인 | *"The elongation threshold θ_e was set at the KDE valley between linear-scratch and non-linear defect distributions (θ_e ≈ X.X), providing clean morphological separation."* |

---

## 출력 파일 목록

```
review/figures/
  q_score_distribution.png          # 분석 1-A: Q 히스토그램 + ECDF
  q_score_by_class.png              # 분석 1-B: 클래스별 Q 분포
  q_threshold_sensitivity.png       # 분석 2-A: 임계값 vs 잔류 샘플 수
  q_threshold_elbow.png             # 분석 2-B: 임계값 vs mAP (P1 완료 후)
  bg_stats.csv                      # 분석 3-A: 배경 통계 원시 데이터
  bg_threshold_distribution.png     # 분석 3-B: 배경 임계값 위치
  subtype_threshold_distribution.png # 분석 4-A: elongation/solidity 분포
```

---

## 작업 순서

- [ ] 1. Colab에서 분석 3-A 실행 (배경 통계 측정, ~10분)
- [ ] 2. Colab에서 분석 1-A/1-B 실행 (Q 분포)
- [ ] 3. Colab에서 분석 2-A 실행 (임계값 민감도 — 잔류 샘플)
- [ ] 4. Colab에서 분석 3-B, 4-A/4-B 실행 (배경/서브타입 분포)
- [ ] 5. 히스토그램에서 valley / elbow 존재 여부 확인
- [ ] 6. 위 결과에 따라 논문 서술 전략 선택 (위 표 참조)
- [ ] 7. P1 multi-seed 완료 후 분석 2-B (elbow 그래프) 실행
- [ ] 8. 논문에 그림 + 서술 추가

---

## 참고

- Q-score 가중치 코드: `scripts/score_casda_quality.py` L120–174
- 배경 분류 로직: `src/analysis/background_characterization.py`
- 결함 서브타입 로직: `src/analysis/defect_characterization.py`
- 관련 실험: `review/robustness_experiments.md` — P1 multi-seed (분석 2-B 연계)
