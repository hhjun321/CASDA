# Response to Reviewer 2

We thank Reviewer 2 for the rigorous and detailed evaluation. We have revised the manuscript substantially and address all eleven comments below.

---

## Comment 1

> The major components of CASDA are individually familiar concepts in the literature. The manuscript should more clearly explain what constitutes the core methodological novelty of CASDA beyond combining these existing elements.

**Response:** We agree the novelty was not articulated clearly. Two contributions distinguish CASDA from prior work.

**Joint defect-background conditioning.** Prior diffusion-based defect synthesis conditions generation on defect geometry or class label alone. CASDA's 3-channel hint image simultaneously encodes defect pixel-level geometry (R), background grain orientation (G), and surface roughness (B), combined with a 5×4 defect-background compatibility matrix that encodes physical co-occurrence plausibility. This joint conditioning scheme is not present in prior industrial defect synthesis work.

**Clean image reuse.** CASDA synthesizes defects onto defect-free images selected via the compatibility matrix, converting the large pool of otherwise unusable clean images into labeled training samples. Copy-Paste augmentation can only redistribute defects among existing defect images and provides no mechanism to leverage clean images.

These two contributions are now foregrounded in the revised Introduction and Section 2 (Related Work).

---

## Comment 2

> Several critical parts of the framework are described only conceptually and cannot be reproduced reliably.

**Response:** All missing implementation details have been added to the revised manuscript.

**Compatibility matrix (Table 6, Section 3.2.3).** The 5×4 matrix (5 background types × 4 defect subtypes, scores in [0.2, 1.0]) was defined by empirical calibration based on visual identifiability of each defect-background pair and domain-informed co-occurrence intuition from the Severstal training set. The complete matrix is presented in Table 6.

**Background category definitions (Section 3.2.1).** Five types are defined: Smooth (uniform, no directional pattern), Textured (grainy surface), Vertical Stripe (directional vertical pattern), Horizontal Stripe (directional horizontal pattern), Complex (multi-directional, high edge density). Classification is performed via Canny edge density and Sobel direction maps.

**Prompt construction (Section 3.2.2).** Four semantic fields concatenated in a fixed template (F1: defect type, F2: surface description, F3: directional texture, F4: stability condition). Example: *"a linear scratch defect on vertical striped metal surface with directional texture (pristine), class 2."*

**ControlNet settings (Section 3.2.2).** Base model: runwayml/stable-diffusion-v1-5 with lllyasviel/sd-controlnet-canny adapter. Fine-tuned up to 20 epochs, AdamW (lr = 1×10⁻⁵, cosine scheduler, 50 warmup steps), min-SNR weighting (γ = 5.0), effective batch size 4, fp16 mixed precision, Google Colab T4 GPU. Inference: 30 denoising steps, guidance scale 7.5, ControlNet conditioning scale 0.7.

**Benchmark training settings (Section 3.3.2).** YOLO-MFD and EB-YOLOv8: AdamW (lr = 0.001, weight decay = 0.0005), batch 16, input 640×640, cosine LR, 300 epochs, warmup 10 epochs, early stopping patience 30.

---

## Comment 3

> Several key decisions appear manually chosen without sufficient justification, including defect classification thresholds, suitability score weights, and quality gate threshold.

**Response:** Justifications have been added to the revised manuscript for each parameter.

**Morphological thresholds.** Linearity > 0.85 and aspect ratio > 5.0 for Linear Scratch; solidity ≥ 0.7 for Compact Blob. Values were determined by examining the bimodal distributions of geometric indices computed over all 3,247 training ROIs, where natural breakpoints between subtype clusters were visually unambiguous. Documented in Section 3.2.1.

**Suitability score weights (0.5 / 0.3 / 0.2).** Components: matching quality (defect-background structural compatibility), spatial continuity, background stability. The 0.5 weight on matching reflects that structural mismatch is the most immediately perceptible failure mode; continuity (0.3) introduces spurious gradients during training; stability (0.2) has the least impact on localization accuracy. Documented in Section 3.2.1.

**Quality gate (Q ≥ 0.7).** This threshold retains 92.8% of ROIs (61.3% high-quality + 31.5% acceptable) while excluding only the 7.2% with severe artifacts. Selection logic is stated in the Table 6 footnote.

---

## Comment 4

> For EB-YOLOv8, the overall improvement is minimal and some metrics are worse than baselines. For DeepLabV3+, the Dice score decreases. Statements claiming consistent performance improvement should be revised.

**Response:** We agree and have revised all relevant sections accordingly.

**DeepLabV3+ removed.** In the revised manuscript, DeepLabV3+ results have been removed. The segmentation performance degradation (−0.58 pp Dice) was attributable to boundary-level artifacts introduced by Poisson blending, a failure mode specific to pixel-level segmentation that requires further controlled analysis before reliable conclusions can be drawn. This limitation is acknowledged in the Discussion.

**YOLO-MFD.** CASDA did not improve overall mAP@0.5 (−2.50 pp vs. Raw; Table 12). Class 2 AP also decreased vs. Raw (−4.59 pp), though it remained superior to Copy-Paste (+5.61 pp). Results are now reported as mean ± std over two seeds, and all claims of broad superiority for YOLO-MFD have been removed.

**EB-YOLOv8.** CASDA improved overall mAP@0.5 (+2.60 pp vs. Raw, +4.74 pp vs. Copy-Paste; Table 13), with a large Class 1 gain (+10.03 pp vs. Raw) and Class 2 AP gain (+22.09 pp vs. Copy-Paste). Architecture-dependent sensitivity is now discussed explicitly in Section 5.

---

## Comment 5

> The manuscript compares CASDA mainly with Raw data and Copy-Paste. This is not enough to establish state-of-the-art relevance.

**Response:** We acknowledge this limitation. The current baselines were chosen to isolate CASDA's specific contributions relative to no-augmentation and the most directly comparable instance-pasting method. Comparison against additional generative baselines (GAN-based, diffusion-based augmentation) is identified as an important direction and is explicitly listed as future work in the revised Discussion.

---

## Comment 6

> Without repeated runs, standard deviations, or significance testing, it is unclear whether reported gains are meaningful.

**Response:** Multi-seed evaluation (seeds 42 and 456) has been added, and all results are now reported as mean ± std (Tables 12–13). Formal hypothesis tests (Wilcoxon signed-rank, BH-FDR corrected, α = 0.05) were conducted but did not reach significance, primarily due to limited statistical power at n = 2 seeds. This limitation is acknowledged explicitly, and descriptive statistics are presented as the primary evidence.

---

## Comment 7

> The ablation study is incomplete. The truly central components — compatibility matrix, 3-channel hint, prompt fields, quality gate — are not independently tested.

**Response:** The ablation study has been removed from the revised manuscript. A full component-level ablation isolating each of the central mechanisms requires training multiple additional experimental configurations at full scale, which necessitates more controlled experimental design than was feasible in this revision. This is identified as an important direction for future work.

---

## Comment 8

> If test-set performance influenced the split ratio selection, there may be selection bias. The authors should clarify whether evaluation data was excluded from synthesis.

**Response:** Data isolation is now documented explicitly in Section 3.3.1.

- **Split ratio selection** was based solely on validation-set performance with the test partition held out entirely.
- **ROI extraction and background patches** were performed exclusively on training-set images (70% split). No validation or test images were used as sources.
- **Synthetic generation** is conditioned only on training-set ROIs and background patches. Generated samples were added only to the training partition.

---

## Comment 9

> Synthetic samples are evaluated only using a heuristic quality score. More convincing evidence is needed, such as FID / KID / LPIPS.

**Response:** FID, KID, and LPIPS metrics have been added to Section 4.2 (Table 14) of the revised manuscript, providing distribution-level and perceptual realism evaluation independent of the heuristic quality score. Human perceptual evaluation and systematic mask boundary validation are identified as future work.

---

## Comment 10

> YOLO-MFD, EB-YOLOv8 (detection) and DeepLabV3+ (segmentation) are analyzed together too broadly.

**Response:** DeepLabV3+ has been removed from the revised manuscript (see Comment 4). The remaining evaluation covers two detection models (YOLO-MFD and EB-YOLOv8) with AP-based metrics only. Task-type mixing is no longer a concern in the revised paper.

---

## Comment 11

> Presentation issues: inconsistent table numbering, informal expressions, overstated claims, limited discussion of limitations.

**Response:** The following changes have been made throughout the revised manuscript.

- Table numbering has been verified and corrected sequentially.
- Informal expressions (e.g., "a whopping 15.06 pp") have been replaced with neutral phrasing.
- Overstated claims (e.g., "solves the data sparsity problem") have been replaced with qualified language (e.g., "mitigates data sparsity for minority defect classes").
- A dedicated Limitations paragraph in Section 5 now covers: boundary artifacts in pixel-level synthesis, architecture-dependent augmentation sensitivity, limited statistical power at n = 2 seeds, and the incomplete component-level ablation.
