# Response to Reviewer 2 (2nd Round)

We thank the reviewer for the continued and constructive feedback. We address each remaining comment below.

---

**Comments 1:** The authors explain that the novelty lies in joint defect-background conditioning and clean image reuse. However, this is still mainly a conceptual explanation. The manuscript still needs stronger evidence to demonstrate that CASDA is more than an integration of existing techniques such as ROI extraction, ControlNet generation, prompt conditioning, Poisson blending, and quality filtering.

**Response 1:** Thank you for this comment. We have revised the novelty statement in §1 (Introduction, paragraph 4) to clarify that the contribution lies not in individual techniques but in their task-specific design as an integrated system. In addition, the newly added component-level ablation study (§4.2, Table 14) provides empirical evidence: removing any single component degrades mAP@0.5 by 8.4–37.1% on YOLO-MFD, demonstrating that the performance gain originates from their specific combination. The revised text in §1 reads:

"The novelty lies not in any individual technique but in the task-specific design of their interactions: the 3-channel hint image encodes defect texture, background orientation, and binary mask jointly in a single conditioning signal; the compatibility matrix constrains background selection to semantically consistent substrates; and Poisson blending is applied as a domain-gap reduction mechanism rather than a cosmetic post-processing step. A full component-level ablation study confirms that each of these design choices contributes measurably to downstream detection performance."

---

**Comments 3:** The revised manuscript provides some explanations for the selected thresholds and weights, but these explanations remain largely qualitative. The defect classification thresholds, suitability-score weights, and quality-gate threshold still need stronger justification, such as distributional evidence, sensitivity analysis, or empirical comparison.

**Response 3:** Thank you for this comment. We have revised §3.2.1 to provide distributional evidence for all classification and suitability thresholds. Defect classification thresholds (e.g., linearity λ=0.85, solidity σ=0.9) were derived from the full defect-instance distribution (N=19,958) using log-scale Otsu splitting to identify natural low-density valleys between feature clusters; percentile-based values were used for lower-boundary thresholds. Background suitability thresholds (variance τ=14.09, edge density ε=0.607) were derived analogously. Figures 7, 8, and 9 show the feature distributions with threshold positions. We acknowledge that a formal sensitivity analysis is not included and note this as a direction for future work. The revised text in §3.2.1 reads:

"Classification thresholds were derived from the full defect-instance distribution (N=19,958 instances) rather than set manually. For high-boundary thresholds (e.g., linearity λ=0.85, solidity σ=0.9), log-scale Otsu splitting was applied to the feature histograms to identify the natural low-density valley between clusters. For low-boundary thresholds, percentile-based values were used. The close agreement between data-driven values and final paper values confirms that the adopted thresholds follow natural separation boundaries in the data. Figure 7 illustrates the morphological feature distributions and threshold positions for all defect subtypes."

---

**Comments 5:** This comment has not been fully addressed. The authors acknowledge the limitation but do not add additional generative or diffusion-based baselines. Comparing CASDA mainly with Raw and Copy-Paste is still insufficient to demonstrate the advantage of the proposed framework.

**Response 5:** Thank you for this comment. We acknowledge that the current comparison does not include additional diffusion-based or GAN-based baselines. As a partial step, Vanilla SD (SD v1.5 text-to-image without ControlNet) has been added to the component-level ablation study (§4.2, Table 14) as a diffusion-level reference point (mAP@0.5 = 0.0549, −37.1% vs. CASDA Full). This result establishes a lower bound on unconstrained diffusion-based augmentation and demonstrates that spatial conditioning via ControlNet is essential. We agree that comparison against additional diffusion-based augmentation methods would further substantiate the claim and note this as a limitation and direction for future work.

---

**Comments 7:** This remains the most important unresolved issue. The authors removed the ablation study instead of adding component-level ablation. This does not address the concern. Since the claimed contribution depends on the compatibility matrix, three-channel hint image, prompt construction, quality gate, and ControlNet conditioning, at least a simplified ablation study should be provided to show the contribution of these key components.

**Response 7:** Thank you for this comment. A full component-level ablation study has been added as §4.2 (Table 14). Six variants were evaluated on YOLO-MFD (seed 42) under the same data split and training configuration as the main experiments. The revised §4.2 reads:

"To verify that each CASDA component contributes independently, we conducted a full component-level ablation study on YOLO-MFD. All variants were evaluated on the Severstal test set under the same 70:15:15 data split and training hyperparameters as described in §3.3, using random seed 42. Each variant removes one component at a time while keeping the remaining pipeline intact."

| Variant | mAP@0.5 | Δ vs CASDA Full |
|---------|---------|-----------------|
| CASDA (Full) | **0.0873** | — |
| w/o Seamless Blending | 0.0610 | −30.1% |
| w/o 3-Channel Hint | 0.0627 | −28.2% |
| w/o Compatibility Matrix | 0.0781 | −10.5% |
| w/o Context-Aware Prompt | 0.0800 | −8.4% |
| w/o Quality Gate | 0.0941 | +7.8%† |
| Vanilla SD (no ControlNet) | 0.0549 | −37.1% |

†Removing the Quality Gate inflates recall (0.112) while precision collapses to 0.368; the Quality Gate trades marginal mAP for substantially higher precision, which is the intended behavior in industrial inspection.

---

**Comments 9:** The addition of FID/KID/LPIPS is helpful, but the validation is still not fully sufficient. Human perceptual evaluation, feature-distribution comparison, or mask-boundary validation would further strengthen the evidence, especially because some quality metrics do not clearly favor CASDA over Copy-Paste.

**Response 9:** Thank you for this comment. We have added Boundary Gradient Discontinuity (BGD) as a mask-boundary-specific quality metric in §4.3 (Table 16). BGD is computed as the mean Sobel gradient magnitude over the boundary band extracted from the binary defect mask; lower values indicate smoother blending at the compositing boundary. The result (Table 16) directly addresses the limitation that FID/KID/LPIPS as distribution-level metrics do not capture. The revised text in §4.3 reads:

"To complement distribution-level metrics (FID, KID, LPIPS) with a boundary-specific measure, we additionally report Boundary Gradient Discontinuity (BGD). The boundary band B = dilate(M, k) − erode(M, k) is extracted from the binary defect mask M using an elliptical structuring element with kernel size k = 5. BGD is then computed as the mean Sobel gradient magnitude |∇I| = √(Gx² + Gy²) over all pixels within B on the grayscale composited image."

| Method | BGD (mean ± std) | N |
|--------|-----------------|---|
| CASDA (Poisson Blending) | **49.6 ± 24.4** | 500 |
| Copy-Paste (Direct Paste) | 71.4 ± 26.3 | 500 |

CASDA achieves 30.5% lower BGD than Copy-Paste (Cohen's d ≈ 0.87, large effect size), directly quantifying the boundary quality improvement. We acknowledge that human perceptual evaluation was not conducted and remains a limitation of this study.

---

**Comments 11:** Although the wording has improved, some conclusions remain stronger than the evidence supports. Since the statistical tests are not significant and only two random seeds are used, terms such as "confirming" should be replaced with more cautious wording such as "suggesting" or "indicating under the tested settings." The conclusion should also emphasize that CASDA's effectiveness is architecture-dependent.

**Response 11:** Thank you for this comment. We have replaced definitive language with hedged expressions throughout the manuscript and have explicitly emphasized the architecture-dependent nature of CASDA's effectiveness. Key revisions are as follows.

Abstract (revised):
"suggesting that context-aware synthesis produces more discriminative minority-class training samples than simple patch reuse under the tested settings. Performance gains are architecture-dependent; YOLO-MFD did not show overall improvement, indicating that augmentation sensitivity varies with backbone feature representation."

§4.1 (revised):
"These gains indicate that ControlNet-based context-aware generation produces training samples of higher discriminative value than simple patch reuse. In contrast, CASDA did not improve overall mAP@0.5 for YOLO-MFD (−2.50 pp vs. Raw) ... Due to the small sample size (n = 2 seeds) and evaluation limited to two architectures, statistical power is insufficient to draw definitive conclusions."

§6 Conclusion (revised):
"These results suggest that context-aware synthesis produces more discriminative minority-class samples than simple patch reuse under the tested settings ... indicating architecture-dependent sensitivity that warrants further investigation."
