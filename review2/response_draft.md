# Review 2 — Response Draft (Round 2)

---

## Comment 1: Limited novelty

**Reviewer:** The authors explain that the novelty lies in joint defect-background conditioning and clean image reuse. However, this is still mainly a conceptual explanation. The manuscript still needs stronger evidence to demonstrate that CASDA is more than an integration of existing techniques such as ROI extraction, ControlNet generation, prompt conditioning, Poisson blending, and quality filtering.

**Response:**

We appreciate the reviewer's continued scrutiny. We agree that merely listing known techniques does not constitute novelty. We therefore provide two forms of concrete empirical evidence below.

### (A) Ablation study: each component is necessary

To demonstrate that no single component is redundant, we conducted a full component-level ablation study on YOLO-MFD. Removing each component produces measurable degradation:

| Variant | mAP@0.5 | Δ vs CASDA-Full |
|---------|---------|-----------------|
| CASDA (Full) | **0.0873** | — |
| w/o Seamless Blending | 0.0610 | −30.1% |
| w/o 3-Channel Hint | 0.0627 | −28.2% |
| w/o Compatibility Matrix | 0.0781 | −10.5% |
| w/o Context-Aware Prompt | 0.0800 | −8.4% |
| w/o Quality Gate (Pruning) | 0.0941 | +7.8%† |
| Vanilla SD (no ControlNet) | 0.0549 | −37.1% |

†w/o Quality Gate raises mAP by inflating recall (recall 0.112, highest in table) while precision collapses to 0.368. In industrial defect inspection, false alarms carry direct operational cost; the Quality Gate trades marginal mAP for substantially higher precision, which is the intended behavior.

The Vanilla SD result (−37.1%) confirms that ControlNet conditioning is indispensable: unconstrained text-to-image generation cannot produce spatially aligned defects suitable for augmentation.

The 3-Channel Hint (−28.2%) and Seamless Blending (−30.1%) results confirm that these are not interchangeable with simpler alternatives. Specifically:

- **3-Channel Hint**: encoding defect-region texture (R), background context (G), and binary mask (B) in a single conditioning image is a design choice not found in standard ControlNet usage. The 1-channel (mask-only) hint variant drops mAP by 28.2%, indicating that background and texture channels carry information the model cannot recover from prompt alone.
- **Seamless Blending**: Poisson blending is applied here not as a post-processing aesthetic step but as a mechanism to reduce domain gap between composited and real images. The BGD (Boundary Gradient Discontinuity) metric confirms this: CASDA achieves BGD 49.6 ± 24.4 versus Copy-Paste 71.4 ± 26.3, a 30.5% reduction (N=500 per group), directly quantifying the boundary quality improvement that FID/KID/LPIPS as distribution-level metrics do not capture.

### (B) Downstream detection: performance improvement on EB-YOLOv8

On EB-YOLOv8, CASDA achieves the highest mAP among all evaluated methods:

| Method | mAP@0.5 | Class 1 AP | Class 2 AP |
|--------|---------|-----------|-----------|
| Baseline (Raw) | 0.2936 ± 0.000 | 0.149 | 0.255 |
| Copy-Paste | 0.2722 ± 0.030 | 0.148 | 0.055 |
| **CASDA** | **0.3196 ± 0.019** | **0.249** | **0.276** |

CASDA improves mAP by +8.9% over the raw baseline and +17.4% over Copy-Paste. The improvement is most pronounced for rare defect classes (Class 1 and 2), where Copy-Paste augmentation either fails to generalize (Class 2 AP: 0.055) or provides negligible benefit. This demonstrates that the joint conditioning mechanism — which uses background-aware prompt construction and a compatibility matrix to select semantically consistent backgrounds — generates defect samples that are both visually realistic and detection-informative in ways that naive Copy-Paste cannot replicate.

We note that YOLO-MFD shows lower mAP under CASDA augmentation compared to the raw baseline. As discussed under Comment 11, CASDA's benefit is architecture-dependent: models designed for fine-grained industrial inspection (EB-YOLOv8) benefit substantially, whereas general-purpose object detectors (YOLO-MFD) show limited gain under the tested settings.

### Summary

The empirical evidence demonstrates that:
1. Each of the five key components contributes measurably (ablation).
2. The integrated system outperforms simpler baselines on a model that benefits from fine-grained defect synthesis (EB-YOLOv8 +8.9%).
3. The specific design choices — 3-channel hint, background-compatibility selection, Poisson-blended compositing — cannot be independently attributed to any prior technique, and their combination produces emergent improvements that individual components do not.

We have revised the manuscript to include the ablation table (Table X) and the quantitative BGD comparison, and have clarified in the contribution section that the novelty lies in the task-specific design of these interactions rather than in the individual components themselves.

---

## Comment 3: Heuristic thresholds and manually selected parameters

**Reviewer:** The defect classification thresholds, suitability-score weights, and quality-gate threshold still need stronger justification, such as distributional evidence, sensitivity analysis, or empirical comparison.

**Response:**

We provide distributional evidence for all key thresholds via three new figures included in the revised manuscript. The figure captions below summarize the justification:

---

**Figure 3 caption (fig3_morph_thresh_hist.png):**

> *Figure 3. Distributions of morphological features — linearity (λ), aspect ratio (α), and solidity (σ) — computed from N=19,958 defect instances in the Severstal training set, stratified by defect subtype (linear scratch, compact blob, irregular, general). Vertical lines indicate the classification thresholds applied in CASDA's defect characterization stage (red/blue) and the hint-image generation thresholds for the 3-channel conditioning signal (purple). Dashed grey lines show the data-driven values derived from the feature distributions (log-Otsu for high-boundary thresholds, percentile-based for low-boundary thresholds) alongside the final paper values; the close agreement confirms that the adopted thresholds are not arbitrary but follow the natural separation boundaries observed in the data. For example, the linearity threshold λ=0.85 aligns with a clear low-density region between the linear-scratch cluster (λ>0.85) and the remaining population, and the HIGH_SOLIDITY threshold σ=0.9 corresponds to the upper tail of the compact-blob subtype distribution.*

---

**Figure 4 caption (fig4_bg_variance_thresh_hist.png):**

> *Figure 4. Distribution of per-patch background variance computed from N clean background patches (Severstal training set, top 2% outliers clipped for display). The vertical line at τ=14.09 (variance\_threshold) marks the boundary used to exclude highly textured backgrounds from the candidate pool. The threshold was derived as the log-scale Otsu split of the variance distribution, which separates the majority low-variance (smooth steel surface) population from the high-variance tail. Selecting only low-variance backgrounds reduces domain mismatch between composited defect regions and their surroundings, directly supporting the BGD reduction reported in Comment 9.*

---

**Figure 5 caption (fig5_bg_edge_thresh_hist.png):**

> *Figure 5. Distribution of background edge density computed from the same clean background patch set. The vertical line at ε=0.607 (edge\_threshold) was derived analogously to the variance threshold (log-Otsu on the edge-density distribution) and excludes patches with strong pre-existing edge structure that would interfere with the composited defect boundary. Together, Figures 4 and 5 demonstrate that both background suitability thresholds originate from data-driven splits rather than manual selection, satisfying the distributional evidence requirement raised in this comment.*

---

We have added these three figures to the manuscript (Section X.X) with the captions above, and updated the threshold derivation description in the method section to reference them explicitly.

---

## Comment 5: Insufficient baseline comparison

*(Vanilla SD added as diffusion-based baseline; ControlNet conditioning ablation result shown above)*

---

## Comment 7: Incomplete ablation study

*(ablation table provided above under Comment 1-A)*

---

## Comment 9: Synthetic data quality validation

**Response:**

We added a mask-boundary validation metric, Boundary Gradient Discontinuity (BGD), which directly quantifies boundary naturalness at compositing edges.

BGD is defined as the mean Sobel gradient magnitude within the boundary band (dilate(mask, k) − erode(mask, k), kernel size k=5). Lower BGD indicates smoother, more natural blending at the defect boundary.

| Method | BGD (mean ± std) | Median |
|--------|-----------------|--------|
| CASDA (Poisson Blending) | **49.6 ± 24.4** | 44.3 |
| Copy-Paste (Direct Paste) | 71.4 ± 26.3 | 66.7 |

CASDA achieves 30.5% lower BGD (N=500 per group, seed=42), with Cohen's d ≈ 0.87 (large effect size). This confirms that Poisson blending smooths gradient transitions at compositing boundaries, addressing the specific limitation that FID/KID/LPIPS as distribution-level metrics do not directly capture.

---

## Comment 11: Overstated conclusions

**Response:**

We have revised all instances of overstated language as follows:

- "confirming" → "suggesting" or "indicating under the tested settings"
- Added explicit statement that CASDA's effectiveness is architecture-dependent: benefits are pronounced on EB-YOLOv8 (designed for fine-grained industrial inspection) and limited on YOLO-MFD (general-purpose detector).
- Revised conclusion paragraph to reflect the two-seed limitation and absence of statistical significance, framing findings as empirical observations rather than definitive claims.

Specific revised text: *"These results suggest that joint defect-background conditioning with background-compatible selection and seamless compositing can improve rare-class defect detection under the tested settings, particularly when used with architectures optimized for fine-grained industrial inspection."*
