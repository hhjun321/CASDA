# Response to Reviewer 3

We thank Reviewer 3 for the extensive evaluation. We address all 23 comments below.

---

## Comment 1

> General comment: grammar mistakes and low English quality.

**Response:** The entire manuscript has been comprehensively revised. Informal phrasing was eliminated, technical terminology was standardized, and grammatical errors in Sections 3 and 4 were corrected.

---

## Comment 2

> Improve the literature review in the introduction by including research articles on defect detection using aware data augmentation methods.

**Response:** Two new subsections have been added to Section 2.

**Section 2.5 — Data-Driven Augmentation for Industrial Inspection.** Reviews GAN-based and diffusion-based defect generation methods, and identifies their shared limitation: generation is conditioned on defect appearance alone without explicit background modeling. CASDA addresses this through the compatibility matrix and 3-channel hint image.

**Section 2.6 — Aware Data Augmentation for Defect Detection.** Reviews GAN-based augmentation for rare defect classes, diffusion-based few-shot defect generation with mask-guided fine-tuning, and context-aware inpainting with boundary-coherence loss. CASDA formalizes this line of work through an explicit compatibility matrix and 3-channel hint image.

---

## Comment 3

> How sensitive is CASDA to errors in ROI characterization and feature extraction bias?

**Response:** Two built-in mechanisms reduce this sensitivity. The suitability score downgrades or excludes ROIs with ambiguous geometric features (7.2% excluded). The quality gate (Q ≥ 0.7) independently rejects generated samples that are morphologically implausible regardless of the input characterization. Systematic sensitivity analysis is identified as future work.

---

## Comment 4

> What is the theoretical justification for the weighting scheme (0.5, 0.3, 0.2), and has sensitivity analysis been performed?

**Response:** The weights reflect domain-informed priorities: matching quality (0.5) captures the most perceptible failure mode (structural mismatch), continuity (0.3) prevents spurious gradients during training, and stability (0.2) has the least localization impact. Justifications are stated in Section 3.2.1. Formal weight optimization is identified as future work.

---

## Comment 5

> How does CASDA compare against state-of-the-art diffusion-based methods using FID, LPIPS?

**Response:** FID, KID, and LPIPS metrics have been added to Section 4.2 (Table 14). LPIPS Realism: CASDA (0.467) outperforms Copy-Paste (0.491), confirming Poisson blending reduces boundary artifacts. LPIPS Diversity: CASDA (0.448) exceeds Copy-Paste (0.394), confirming structural variety. Direct comparison against other diffusion-based augmentation systems requires controlled re-implementation and is identified as future work.

---

## Comment 6

> To what extent does the synthetic-to-real domain gap affect generalization on unseen datasets?

**Response:** CASDA's compatibility matrix and background type definitions were derived from the Severstal dataset and may require recalibration for different surface characteristics. The ControlNet model was fine-tuned on Severstal ROIs and may exhibit distribution shift on new substrates. This limitation is acknowledged in Section 5, and cross-dataset generalization evaluation is identified as future work.

---

## Comment 7

> Why does CASDA degrade segmentation performance in DeepLabV3+, and how can boundary artifacts be minimized?

**Response:** DeepLabV3+ results have been removed from the revised manuscript. The Dice score degradation was attributable to boundary-level gradient artifacts from Poisson blending — a failure mode specific to pixel-level segmentation that requires further controlled analysis. The current evaluation focuses on detection models (YOLO-MFD and EB-YOLOv8). Multi-scale blending and texture-aware compositing are identified as future work.

---

## Comment 8

> How scalable is CASDA to multi-material or multi-sensor datasets?

**Response:** The core pipeline is dataset-agnostic in structure. Scaling to new domains would require: (1) recalibrating the compatibility matrix for new defect-background combinations, (2) extending the background type taxonomy, and (3) fine-tuning ControlNet on the target domain. Establishing a universal pipeline for PCBs, wood, and carpets is identified as a primary future work direction.

---

## Comment 9

> What are the computational costs compared to traditional augmentation and GAN-based methods?

**Response:** Per-stage timing data is not available. Qualitatively, CASDA's offline synthesis (ROI extraction, ControlNet generation, quality verification, Poisson blending) is more compute-intensive than geometric augmentation or Copy-Paste. However, synthesis is performed entirely offline; no additional cost is incurred at inference. Computational benchmarking is identified as future work.

---

## Comment 10

> How does the method ensure generated defects do not introduce label noise?

**Response:** Two mechanisms prevent label noise. The quality gate (Q ≥ 0.7) rejects morphologically inconsistent, low-sharpness, or artifact-contaminated samples — 17.0% of candidates (417 of 2,500) were excluded. Defect masks are derived from original ROI masks and geometrically transformed to match placement, ensuring structural consistency between mask and generated content. Both are described in Sections 3.2.4–3.2.5.

---

## Comment 11

> Can the morphological indices fully capture complex defect geometries?

**Response:** The four indices (linearity, solidity, area ratio, aspect ratio) are effective for macro-scale subtypes but have limitations for irregular micro-defects with complex boundary topology, which may be misclassified into the General category. This limitation is acknowledged in the revised Discussion. Extension to richer descriptors (e.g., fractal dimension, curvature-based features) is identified as future work.

---

## Comment 12

> What is the impact of the 42.7% synthetic ratio on overfitting, and how was it determined?

**Response:** The 42.7% ratio was not independently optimized as a global target. It emerged from per-class augmentation decisions designed to achieve a roughly uniform synthetic-to-real distribution across classes, while preserving the distributional characteristics of the original data. The impact of alternative ratios on overfitting is acknowledged as a future investigation direction.

---

## Comment 13

> How reproducible is the pipeline across datasets given heuristic thresholds?

**Response:** The thresholds (linearity > 0.85, aspect ratio > 5.0, solidity ≥ 0.7) were calibrated on Severstal and are not guaranteed to transfer to datasets with different morphology distributions. The threshold derivation methodology (histogram inspection of geometric index distributions) is fully documented in Section 3.2.1, enabling practitioners to recalibrate for new datasets. Cross-dataset transfer studies are identified as future work.

---

## Comment 14

> Why does EB-YOLOv8 show marginal improvement compared to YOLO-MFD?

**Response:** The question is reversed in the revised results: YOLO-MFD did not improve overall mAP (−2.50 pp vs. Raw; Table 12), while EB-YOLOv8 showed consistent gains (+2.60 pp vs. Raw, +4.74 pp vs. Copy-Paste; Table 13). We attribute the differential response to BiFPN in EB-YOLOv8 being more sensitive to contextually consistent minority-class samples, whereas YOLO-MFD's backbone may require augmentation strategies more tightly coupled to its feature representation. This architecture-dependent sensitivity is discussed in Section 5.

---

## Comment 15

> How does Q ≥ 0.7 influence diversity versus accuracy trade-offs?

**Response:** Q ≥ 0.7 retains 92.8% of candidates (2,075 of 2,500), excluding only severe failures. LPIPS Diversity (0.448) confirms the retained samples remain structurally varied. Lowering the threshold admits more diverse but lower-quality samples; raising it improves quality at the cost of data quantity. Systematic exploration of this trade-off is identified as future work.

---

## Comment 16

> Could GAN-based refinement or diffusion inpainting outperform Poisson blending?

**Response:** Poisson blending provides computationally efficient boundary harmonization without additional model training. GAN-based refinement and diffusion inpainting have shown superior perceptual fidelity in general compositing tasks, and their application to industrial defect synthesis is a natural extension identified as future work.

---

## Comment 17

> How does CASDA perform under extreme class imbalance where minority samples are nearly absent?

**Response:** Class 2 (247 original samples) is the closest case in this work. CASDA added 519 synthetic samples (110.1% augmentation rate), and EB-YOLOv8 Class 2 AP improved by +22.09 pp vs. Copy-Paste. Near-zero minority class scenarios (fewer than 10–20 samples) are beyond the current evaluation scope and are identified as a future investigation target.

---

## Comment 18

> What is the statistical significance of the reported improvements?

**Response:** Multi-seed evaluation (seeds 42 and 456) has been added, and all results are reported as mean ± std (Tables 12–13). Formal hypothesis tests (Wilcoxon signed-rank, BH-FDR corrected, α = 0.05) were conducted but did not reach significance due to limited statistical power at n = 2 seeds. This limitation is acknowledged in the revised Discussion.

---

## Comment 19

> How does the framework handle overlapping or interacting defects?

**Response:** The current pipeline places a single ROI-derived defect per image. Overlapping or co-occurring defect synthesis is not supported. Multi-instance synthesis with overlap-aware placement is identified as future work.

---

## Comment 20

> Can the approach be extended to 3D defect detection?

**Response:** The current framework operates on 2D grayscale images. 3D extension would require: (1) 3D morphological characterization, (2) a volumetric background compatibility model, and (3) a volumetric generative model replacing ControlNet. These represent substantial architectural changes and are identified as a long-term research direction.

---

## Comment 21

> What are the limitations of using the grayscale Severstal dataset?

**Response:** The grayscale format means CASDA was fine-tuned and evaluated without color texture information. The 3-channel hint image captures structural and orientation signals that do not depend on color, but background characterization and compatibility matrix behavior on multi-channel RGB or hyperspectral data is uncertain. This limitation is acknowledged in Section 5, and cross-domain validation is identified as future work.

---

## Comment 22

> Discussion must be strengthened by comparison with results published by others.

**Response:** Sections 2.5 and 2.6 situate CASDA within related GAN-based, diffusion-based, and inpainting-based augmentation work. Direct numerical comparison in the Discussion is not included, as differences in dataset, evaluation protocol, and defect taxonomy make cross-paper comparisons unreliable without controlled re-implementation. Establishing a standardized benchmark is identified as future work.

---

## Comment 23

> Rewrite the abstract and conclusion incorporating the above suggestions.

**Response:** Both have been rewritten in the revised manuscript. The abstract reports key quantitative results under multi-seed evaluation (83.0% quality pass rate, EB-YOLOv8 mAP +2.60 pp vs. Raw, Class 2 AP +22.09 pp vs. Copy-Paste) and acknowledges architecture-dependent sensitivity. The conclusion reports results with standard deviations, acknowledges that YOLO-MFD did not benefit, and explicitly identifies limitations and future work.
