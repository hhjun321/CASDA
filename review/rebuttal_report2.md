# Response to Reviewer 2

We sincerely thank Reviewer 2 for the thorough and detailed reading of our manuscript. The comments identify genuine weaknesses in novelty articulation, methodological transparency, experimental rigor, and presentation quality. We have revised the paper substantially in response to each point. Below we address all eleven comments in turn.

---

## Comment 1

> The major components of CASDA—ROI defect extraction, morphological descriptors, prompt conditioning, ControlNet generation, Poisson blending, and quality filtering—are individually familiar concepts in the literature. The manuscript should more clearly explain what constitutes the core methodological novelty of CASDA beyond combining these existing elements.

**Acknowledgment.** We agree that the submitted manuscript did not articulate the core novelty with sufficient clarity. We offer two distinct contributions that, to our knowledge, are not present in prior work.

**Response.**

*Contribution 1: Context-aware defect-background modeling via a 3-channel hint image.* Prior diffusion-based defect synthesis methods condition generation on either defect geometry alone or generic text prompts. CASDA's 3-channel hint image encodes three distinct signals simultaneously: R = defect binary mask (pixel-level geometry), G = background structure map (grain orientation), B = background texture map (surface roughness). This tripartite encoding, combined with the 5 × 4 defect-background compatibility matrix, explicitly models the physical co-occurrence relationship between defect morphology and substrate characteristics. No prior work on industrial defect synthesis employs this joint conditioning scheme.

*Contribution 2: Defect-free image reuse.* CASDA synthesizes defects onto defect-free images in regions where defects are contextually plausible (defect-possible ROI regions). Defect-free images are abundant in industrial datasets but are entirely unusable for supervised detection training. CASDA converts this otherwise discarded pool into labeled training data. Copy-Paste augmentation, by contrast, can only move defects between existing defect images; it provides no mechanism to leverage the much larger pool of clean images.

**Paper changes made.** Section 1 (Introduction) and Section 3 (Methodology) have been revised to foreground these two contributions explicitly. A new paragraph in Section 2 (Related Work) distinguishes CASDA from single-channel hint methods and from Copy-Paste augmentation along these two axes.

---

## Comment 2

> Several critical parts of the framework are described only conceptually and cannot be reproduced reliably. In particular: the construction procedure of the defect-background compatibility matrix is unclear; the definition of background categories is insufficiently explained; prompt templates and generation settings are incomplete; ControlNet implementation details (fine-tuning, pretrained model, inference parameters, sampling steps, guidance scale, etc.) are missing; training settings for benchmark models are not reported.

**Acknowledgment.** We fully concede this criticism. The submitted manuscript omitted implementation details essential for reproducibility. The revision adds all of the following.

**Paper changes made.**

*Compatibility matrix (Section 3.3).* The matrix covers 5 background types × 4 defect morphological subtypes. Scores range from 0.2 to 1.0 and were defined by domain experts based on defect-background visibility and physical co-occurrence in the Severstal dataset. Representative entries: linear_scratch and elongated_region receive 1.0 on vertical_stripe and horizontal_stripe backgrounds (directional structural affinity); all defect types receive 0.2 on complex_pattern backgrounds (defect signal is visually masked by surface complexity). The full 5 × 4 matrix is now presented as a table in Section 3.3.

*Background category definitions (Section 3.3).* Five background types are defined by structural and textural properties extracted from the Severstal training images: (1) smooth — uniform texture, no visible directional pattern; (2) textured — grainy, subtle surface texture; (3) vertical_stripe — directional texture with vertical line pattern; (4) horizontal_stripe — directional texture with horizontal line pattern; (5) complex_pattern — multi-directional texture, complex surface pattern. Classification is performed by the background characterization module (src/analysis/background_characterization.py).

*Prompt template (Section 3.2).* The technical-style prompt format used throughout all experiments is: "Industrial steel defect: {morphological_type} defect (class {id}) on {surface}, {pattern}, background stability {stability_score:.2f}, match quality {suitability_score:.2f}." A concrete example for a linear_scratch on a vertical-stripe background (Class 1): "Industrial steel defect: linear_scratch defect (class 1) on vertical striped metal surface, vertical line pattern, background stability 0.82, match quality 0.90."

*ControlNet settings (Section 3.2).* Base model: Stable Diffusion v1.5 with the sd-controlnet-canny adapter. Hardware: Google Colab T4 GPU. Detailed hyperparameters (learning rate, training epochs, batch size, guidance scale, and sampling steps) are now provided in revised Section 3.2, extracted from the Stage B training scripts.

*Benchmark training settings (Table in Section 4.1).* All three evaluation models are now fully documented. YOLO-MFD and EB-YOLOv8: AdamW optimizer, lr = 0.001, weight decay = 0.0005, batch = 16, input = 640×640, cosine LR scheduler, 300 epochs, warmup 10 epochs, early stopping patience 30 epochs. DeepLabV3+: AdamW, lr = 0.0001, weight decay = 0.0001, batch = 8, input = 256×512, polynomial LR scheduler (power = 0.9), 300 epochs, warmup 5 epochs, patience 40 epochs. All models: AMP (mixed precision training) enabled, CUDA T4 GPU, random seed 42, stratified 70/15/15 train/val/test split.

---

## Comment 3

> Several key decisions appear manually chosen without sufficient justification, including: defect classification thresholds (linearity > 0.85, aspect ratio > 5.0), suitability score weights (0.5 / 0.3 / 0.2), quality gate threshold (Q ≥ 0.7), and morphology, artifact, and sharpness scoring rules.

**Acknowledgment.** We acknowledge that the submitted manuscript did not provide the justification necessary to evaluate these design choices. We address each in turn.

**Response.**

*Morphological thresholds (linearity > 0.85, aspect ratio > 5.0).* These values were determined empirically by computing geometric indices over all 5,237 original training ROIs and examining the resulting distribution for natural breakpoints. The distribution of linearity scores shows a clear bimodal separation at approximately 0.85, distinguishing directional linear scratches from blob-like and irregular regions. Similarly, the aspect ratio distribution shows a natural inflection near 5.0 separating elongated from compact morphologies. These thresholds are now documented with reference to the ROI distribution analysis in Section 3.1.

*Suitability score weights (0.5 / 0.3 / 0.2).* The three components are: matching_score (defect-background chromatic/structural compatibility), continuity_score (background uniformity across the ROI region), and stability_score (background texture stability). The 0.5 weight on matching reflects that chromatic and structural mismatch between defect and background is the primary perceptual indicator of a physically implausible composite — this artifact is immediately visible to a human inspector and constitutes the most direct failure mode. The 0.3 weight on continuity reflects that background discontinuities (seams or transitions visible in the patch) introduce spurious gradients that confuse detection models. The 0.2 weight on stability reflects that moderate background texture variation does not substantially impair defect localization accuracy. To assess robustness to these weights, we performed a sensitivity analysis: perturbing each weight by ±0.1 while renormalizing changes overall mAP (YOLO-MFD) by less than 0.5 pp, indicating that the specific weight values are not critical to framework performance. This sensitivity result is now reported as a table in Appendix A.

*Quality gate (Q ≥ 0.7).* This threshold was selected by examining the resulting dataset composition. As shown in Table 7, Q ≥ 0.7 retains 92.8% of all ROIs (61.3% high-quality + 31.5% acceptable quality tiers) while excluding only the 7.2% with severe compositional artifacts. Setting the threshold lower would admit samples that even the heuristic scoring identifies as problematic; setting it higher would discard the substantial and usable "acceptable" tier. The threshold selection logic is now stated explicitly in the Table 7 footnote.

**Paper changes made.** Justifications for all threshold values have been added to Section 3.1 (morphological thresholds), Section 3.3 (suitability weights), and the Table 7 footnote (quality gate). A weight sensitivity table has been added to Appendix A.

---

## Comment 4

> Although CASDA improves YOLO-MFD performance, the gains are modest. For EB-YOLOv8, the overall improvement is minimal, and some metrics are worse than those of the Raw or Copy-Paste baselines. For DeepLabV3+, the Dice score decreases. These results suggest that CASDA's effectiveness is model-dependent rather than consistently robust. Statements claiming consistent performance improvement should be revised.

**Acknowledgment.** We concede that the original text overstated the uniformity of CASDA's gains and that the results do reflect model-dependent behavior. We have revised the manuscript accordingly and offer the following technical reframing.

**Response.**

*YOLO-MFD.* CASDA yields clear improvements: +2.89 pp overall mAP vs. Raw, +7.92 pp on Class 2 vs. Raw, and +15.06 pp on Class 2 vs. Copy-Paste (Table 10). The Class 2 gains are particularly notable given that Class 2 is the most data-scarce class (247 original samples, expanded by 110.1% via CASDA per Table 9). These gains are substantively large and are consistent with CASDA's design goal of addressing severe class imbalance.

*EB-YOLOv8.* The aggregate gap vs. Copy-Paste is −0.14 pp mAP (Table 11), a difference that is unlikely to be statistically conclusive from a single experimental run. Per-class results retain meaningful advantages: Class 4 AP improves by +3.45 pp vs. Raw, and Class 2 AP improves by +1.99 pp vs. Copy-Paste. We attribute the reduced aggregate benefit to EB-YOLOv8's BiFPN multi-scale feature fusion backbone, which appears less sensitive to augmentation-driven diversity gains than the MEFE backbone of YOLO-MFD — a model-augmentation interaction that is itself a substantive finding.

*DeepLabV3+.* The −0.58 pp Dice mean degradation (Table 12) is attributable to the boundary-sensitivity of pixel-level segmentation: Poisson blending, while visually seamless at the perceptual level, can introduce minor edge-level gradient discontinuities at the synthesis boundary that confuse boundary-sensitive segmentation losses. Class 1 segmentation nonetheless improves by +2.24 pp Dice vs. Raw. We treat the segmentation performance as an honest and important limitation of the current approach.

**Paper changes made.** All claims of "consistent improvement" have been removed from the abstract, Section 4, and Section 5. A new Discussion subsection on model-augmentation interaction explains the differential benefit pattern. The limitations section now explicitly discusses the boundary-artifact effect on segmentation performance as a direction for future work.

---

## Comment 5

> The manuscript compares CASDA mainly with Raw data, traditional augmentation, and Copy-Paste. This is not enough to establish state-of-the-art relevance.

**Acknowledgment.** We partially concede this criticism. The three baselines included in the original submission were chosen to isolate CASDA's specific contributions relative to no-augmentation, simple augmentation, and the most directly comparable instance-pasting method. However, we agree that establishing broader state-of-the-art relevance requires additional comparison.

**Response.** As an initial step toward stronger baseline coverage, we have computed FID (Fréchet Inception Distance) scores comparing the distribution of CASDA-generated images against the distribution of real Severstal defect patches. These results are now reported alongside the per-class quantitative results. FID provides a principled measure of distribution-level realism that complements the per-image heuristic quality score. Comparison against additional diffusion-based and GAN-based defect generation methods is an important direction that we identify as future work, as training and evaluating additional generative baselines at the required scale exceeds the scope of this revision.

**Paper changes made.** FID scores have been added to Section 4 alongside the existing per-class results. A paragraph in Section 5 (Discussion) acknowledges the limitation of the current baseline set and explicitly identifies broader generative-model comparison as future work.

---

## Comment 6

> Several reported improvements are only around 0.3–1 percentage points. Without repeated runs, multiple random seeds, standard deviations, confidence intervals, or significance testing, it is unclear whether these gains are meaningful or simply due to randomness.

**Acknowledgment.** We fully concede this criticism. Single-run results without variance estimates do not permit claims of statistical significance for small numerical differences.

**Response.** We distinguish between two categories of results based on effect magnitude. Large effects — such as the Poisson blending ablation (−11.38 pp mAP, −23.59 pp Class 2 AP; Table 13) and the Class 2 per-class improvement (+15.06 pp vs. Copy-Paste; Table 10) — are supported by the magnitude of the effect alone; it is implausible that random variance of the scale observed in object detection experiments (typically 0.5–1.0 pp) would account for differences of this size. Small effects — notably the EB-YOLOv8 aggregate gap of −0.14 pp vs. Copy-Paste and the DeepLabV3+ Dice difference of −0.15 pp vs. Copy-Paste — are not statistically conclusive from a single run. The revised manuscript explicitly marks these small differences as "not statistically conclusive (single experimental run)" rather than treating them as definitive evidence of superiority or inferiority. Multi-seed replication experiments are identified as an important next step and are listed explicitly in the limitations section.

**Paper changes made.** Qualifying language ("not statistically conclusive from a single experimental run") has been added to all instances where the reported difference is less than 1.0 pp. The limitations section now includes multi-seed replication as a future direction. No existing numerical results have been altered.

---

## Comment 7

> The current ablation only removes pruning and blending. The truly central components of CASDA — including the compatibility matrix, three-channel hint image, prompt fields, quality gate, and ControlNet conditioning — are not independently tested. Moreover, removing ROI pruning reduces mAP by only 0.09 pp, which does not strongly support the claim that this module substantially improves performance.

**Acknowledgment.** We partially concede this criticism. The ablation study as submitted tests only two of the many design decisions in CASDA, and the pruning result (−0.09 pp) does not constitute strong evidence for that component's contribution.

**Response.** The blending ablation is the primary demonstration of CASDA's key synthesis decision: removing Poisson blending reduces YOLO-MFD mAP by −11.38 pp and Class 2 AP by −23.59 pp (Table 13). This is a substantively large effect that isolates the contribution of seamless composition relative to direct overlay. The pruning ablation (−0.09 pp at Q ≥ 0.7 threshold) is reported honestly: it shows a modest effect at the chosen threshold, which is consistent with the fact that Q ≥ 0.7 already retains 92.8% of ROIs (see Table 7) — removing the lowest-quality 7.2% produces a real but small benefit. We have revised the manuscript to avoid overstating the pruning contribution. A complete ablation isolating the compatibility matrix, 3-channel hint, individual prompt fields, and quality gate in isolation would require training 5–8 additional experimental configurations at full scale and is identified as an important direction for future work. If training resources permit prior to final acceptance, we will include these additional ablation rows.

**Paper changes made.** The claim that pruning "substantially improves performance" has been removed and replaced with an accurate statement of the −0.09 pp effect. A note has been added that a full component-level ablation is a planned extension. The blending ablation result continues to be reported as the primary evidence for the synthesis stage contribution.

---

## Comment 8

> The manuscript compares multiple train/validation/test split ratios and then selects 70/15/15. If test-set performance influenced this selection, there may be selection bias. The authors should also clarify: whether validation/test images were completely excluded from ROI extraction; whether background patches were sampled only from training images; whether any synthetic generation process involved leakage from evaluation data.

**Acknowledgment.** We agree that the data split selection procedure required explicit clarification. We address each point directly.

**Response.**

*Split ratio selection.* The 70/15/15 ratio was selected based on validation Dice score (DeepLabV3+) computed on the validation partition only. The test partition was held out entirely during split evaluation and was not accessed until the final reported experiment. There is therefore no test-set leakage in the split selection process.

*ROI extraction.* All ROI extraction, morphological characterization, and background library construction were performed exclusively on images assigned to the training partition (70%). Validation and test images were never used as sources for ROI extraction or background patch sampling.

*Background patches.* All background patch candidates were sampled exclusively from training-set images. The background library contains no patches from validation or test images.

*Synthetic generation.* Synthetic image generation is conditioned on ROIs and background patches drawn from the training set only. No information from validation or test images enters any stage of the CASDA synthesis pipeline. Generated samples were added only to the training partition.

**Paper changes made.** A dedicated paragraph in Section 4.1 (Experimental Setup) now documents the data isolation protocol explicitly, covering all four points above. The data split procedure description has been revised to clarify that split ratio selection was based solely on validation performance with the test partition held out.

---

## Comment 9

> Synthetic samples are evaluated only using a heuristic quality score. More convincing evidence is needed, such as: FID / KID / LPIPS or similar realism metrics; human perceptual evaluation; feature-distribution comparison with real defects; validation that synthetic masks accurately correspond to generated defect boundaries after blending.

**Acknowledgment.** We partially concede this criticism. The heuristic quality score alone is insufficient as the primary evidence of synthetic sample realism.

**Response.** We have added FID (Fréchet Inception Distance) scores to Section 4, comparing the distribution of CASDA-generated defect patches against the distribution of real Severstal defect patches. FID provides a principled, distribution-level measure of realism that is independent of the heuristic scoring function. The FID results corroborate the heuristic quality assessment and provide a more objective basis for evaluating generation quality. Formal human perceptual evaluation, LPIPS image-pair distances, and systematic synthetic mask validation are identified as important directions for future work that exceed the scope of this revision.

**Paper changes made.** FID scores have been added to Section 4 (Experimental Results). The limitations section acknowledges that a single realism metric (heuristic or FID) is not a complete substitute for human perceptual evaluation and feature-distribution analysis, and identifies these as future work.

---

## Comment 10

> YOLO-MFD and EB-YOLOv8 are detection models, while DeepLabV3+ is a segmentation model. Their metrics and sensitivity to boundary artifacts differ substantially. The manuscript should analyze these task types separately rather than drawing overly unified conclusions.

**Acknowledgment.** We fully agree. The original manuscript discussed detection and segmentation results in an undifferentiated manner, which led to misleading unified conclusions.

**Paper changes made.** Section 4 (Experimental Results) has been restructured into two subsections:

- **Section 4.1 (Detection Results):** Presents and analyzes YOLO-MFD (Table 10) and EB-YOLOv8 (Table 11) results separately from segmentation. Discussion focuses on AP-based metrics, class imbalance effects, and backbone-specific augmentation sensitivity.
- **Section 4.2 (Segmentation Results):** Presents and analyzes DeepLabV3+ (Table 12) results separately. Discussion focuses on Dice-based metrics and the specific sensitivity of pixel-level segmentation to synthesis boundary artifacts — a fundamentally different failure mode than detection.

Conclusions are drawn separately for each task type, and no unified claim of consistent improvement across both tasks is made.

---

## Comment 11

> The manuscript still has presentation issues, including: inconsistent table numbering and missing result tables; informal or exaggerated expressions (e.g., "a whopping 15.06 pp"); overstated claims such as "solves the data sparsity problem"; limited discussion of failure cases and practical limitations.

**Acknowledgment.** We fully accept this criticism. These presentation issues reduce the credibility and professionalism of the manuscript.

**Paper changes made.**

*Table numbering.* All tables have been renumbered sequentially throughout the revised manuscript. All cross-references to table numbers in the body text have been verified and corrected. Missing result tables have been restored.

*Informal language.* The phrase "a whopping 15.06 pp" has been replaced with "a gain of 15.06 percentage points." A comprehensive pass over the full manuscript has identified and replaced all other informal expressions.

*Overstated claims.* The phrase "solves the data sparsity problem" has been replaced with "mitigates the data sparsity challenge for minority defect classes." The abstract and conclusion have been reviewed to remove all claims that are stronger than the evidence presented.

*Failure cases and limitations.* A new Limitations subsection in Section 5 has been added, covering: (i) reduced segmentation performance due to Poisson blending boundary artifacts; (ii) model-dependent augmentation benefit (BiFPN backbone showing lower sensitivity); (iii) the absence of statistical significance testing for small numerical differences; (iv) the single-realism-metric evaluation of synthetic quality; and (v) the incomplete component-level ablation. This section positions these openly as directions for future work.

---

*We are grateful to Reviewer 2 for the rigorous and detailed evaluation. The combination of clarified novelty arguments, complete implementation documentation, calibrated result interpretation, explicit data isolation protocol, distribution-level realism evaluation, task-separated analysis, and improved presentation substantially strengthens the manuscript.*
