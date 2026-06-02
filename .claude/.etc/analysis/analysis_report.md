# CASDA Dataset Analysis Report (Stage 0)

> Images analyzed: **12,568** | Defect instances: **19,958** | BG grid cells: **1,256,800**

---

## 1. Class Distribution

| Class | Instances | % of total |
|-------|-----------|------------|
| Class 1 | 3,082 | 15.4% |
| Class 2 | 321 | 1.6% |
| Class 3 | 14,648 | 73.4% |
| Class 4 | 1,907 | 9.6% |

![Class Distribution](figures/fig1_class_distribution.png)
![Subtype Distribution](figures/fig2_subtype_distribution.png)

---

## 2. Morphological Feature Distributions

![Morphological Features](figures/fig3_morph_features_hist.png)

---

## 3. Background Feature Distributions

![Variance](figures/fig4_bg_variance_hist.png)
![Edge](figures/fig5_bg_edge_hist.png)
![Frequency](figures/fig6_bg_freq_hist.png)
![BG Type](figures/fig7_bg_type_distribution.png)

---

## 4. Defect–Background Co-occurrence (4×5 Matrix)

![Heatmap](figures/fig8_defect_bg_heatmap.png)
![Area](figures/fig9_per_class_area_box.png)

---

## 4b. Subtype × Background Compatibility (5×5) — MATCHING_RULES 교체 근거

좌: 데이터 기반 (실측 공출현 비율)  |  우: 논문 prior (수동 설정)

![Compat Heatmap](figures/fig10_subtype_bg_compat_heatmap.png)

> `subtype_bg_matrix_5x5.json` + `recommended_config.yaml:subtype_compatibility_matrix`에 저장.
> `roi_suitability.py:MATCHING_RULES`를 이 값으로 교체할 것.

---

## 5. Recommended Thresholds

| Parameter | Current | Recommended | Method | Derivable |
|-----------|---------|-------------|--------|-----------|
| `HIGH_LINEARITY` | 0.85 | 0.7012 | otsu_1d | ✓ |
| `HIGH_ASPECT_RATIO` | 5.0 | 8.2034 | otsu_1d | ✓ |
| `LOW_ASPECT_RATIO` | 2.0 | 8.2034 | otsu_1d | ✓ |
| `HIGH_SOLIDITY` | 0.9 | 0.8405 | otsu_1d | ✓ |
| `LOW_SOLIDITY` | 0.7 | 0.8405 | otsu_1d | ✓ |
| `variance_threshold` | 100.0 | 14.0894 | log_otsu | ✓ |
| `edge_threshold` | 0.3 | 0.607 | otsu_1d | ✓ |
| `total_strength` | 1.0 | 49.4964 | otsu_1d | ✓ |
| `stripe_ratio_v` | 1.5 | 1.1088 | otsu_1d | ✓ |
| `stripe_ratio_h` | 1.5 | 1.1088 | otsu_1d | ✓ |
| `high_freq_ratio` | 0.3 | 0.4668 | otsu_1d | ✓ |
| `min_suitability` | 0.5 | 0.5 | hardcoded(not_derivable_in_stage0) | ✗ Stage A 이후 |
| `min_quality_score` | 0.5 | 0.5 | hardcoded(not_derivable_in_stage0) | ✗ Stage A 이후 |

---

## 6. Generation Plan

| Parameter | Recommended |
|-----------|-------------|
| `num_images_per_class` | {'1': 5, '2': 46, '3': 1, '4': 8} |
| `compositions_per_roi` | 6 |
| `per_class_cap` | 11178 |
| `rare_class_threshold` | 1510 |

---

## 7. Artifacts

All outputs in: `/content/drive/MyDrive/data/Severstal/analysis`

- `class_distribution.json`
- `morphological_features.csv`
- `background_features.csv`
- `defect_bg_matrix_4x5.json`
- `threshold_recommendations.json`
- `recommended_config.yaml`  ← apply to Stage A~C before re-running
- `figures/` (9 PNG files)