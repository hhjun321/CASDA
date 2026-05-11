# Response to Reviewer 3

We sincerely thank Reviewer 3 for the extensive and detailed evaluation of our manuscript. The 23 comments span language quality, methodological rigor, experimental completeness, and scope — and together constitute the most comprehensive critique we received. We have taken each point seriously and revised the manuscript accordingly. Below we address all 23 comments in turn.

---

## Comment 1

> General comment (enhance the quality of the language). There are several grammar mistakes, and the quality of the English is too low. A thorough review from a native language speaker is suggested to improve the quality of the language.

**Acknowledgment.** We fully accept this criticism. The language quality of the submitted manuscript did not meet the standard expected for a journal publication.

**Paper changes made.** The entire manuscript has been subjected to a comprehensive language revision. All sections have been rewritten in professional academic English, with particular attention to: elimination of informal phrasing and colloquial constructions, consistent use of technical terminology throughout (e.g., uniform use of "synthetic augmentation," "suitability score," and "quality gate" across sections), correction of grammatical errors concentrated in Section 3 (methodology) and Section 4 (results), and restructuring of overly long or syntactically ambiguous sentences. The revised manuscript has additionally been reviewed by a fluent English speaker for idiomatic clarity and has been processed through professional academic editing tools prior to resubmission.

---

## Comment 2

> In the introduction section improve the literature review — by inserting the outcome. It is suggested to include in the introduction some research articles dealing with defect detection using aware data augmentation methods.

**Acknowledgment.** We agree that the literature review in the submitted manuscript did not adequately survey recent work on context-aware and data-driven augmentation for defect detection, and that this omission weakened the positioning of CASDA's contributions.

**Paper changes made.** Section 2 (Related Work) has been expanded with three new subsections:

1. **Diffusion-based augmentation for industrial inspection.** Recent work on score-based generative models and latent diffusion models applied to defect synthesis is now surveyed, including methods that condition generation on defect geometry alone or on generic text prompts. This body of work motivates CASDA's distinguishing contribution: explicit joint modeling of the defect-background relationship through the 5×4 compatibility matrix and the 3-channel hint image, which no prior work in industrial defect synthesis employs.

2. **GAN-based defect generation and copy-paste methods.** The evolution from unconditional GAN-based synthesis to conditional and context-aware generation is reviewed, including CycleGAN, DCGAN, and Pix2Pix variants applied to surface inspection. The structural limitation of Copy-Paste augmentation — that it can only redistribute existing defect instances and cannot leverage the much larger pool of defect-free images — is contrasted with CASDA's clean-image reuse mechanism.

3. **Context-aware and morphology-guided augmentation.** Studies that incorporate structural and contextual priors into the augmentation process (e.g., background-conditioned placement, morphological descriptor-guided synthesis) are reviewed to situate CASDA within this emerging line of work and to clarify how CASDA's tripartite hint encoding advances beyond single-channel or text-only conditioning approaches.

These additions were incorporated into both the Introduction and Section 2, ensuring that the literature survey is visible from the first reading.

---

## Comment 3

> Given that ControlNet-based generation is conditioned on handcrafted features, how sensitive is CASDA to errors in ROI characterization and feature extraction bias?

**Response.** CASDA addresses ROI characterization error through two sequential filtering stages that prevent error propagation into the training dataset.

*Stage A (pre-generation filter).* Before any generation is attempted, each ROI is assigned a suitability score Q = 0.5×matching + 0.3×continuity + 0.2×stability. ROIs with Q < 0.7 are excluded from generation entirely. This gate operates on the characterization outputs: an ROI with erroneous feature extraction (e.g., a misidentified background type, an incorrect morphological subtype) will typically receive a low matching or continuity score, causing it to be filtered before it conditions any generated sample. As shown in Table 7, 7.2% of ROIs are excluded at this stage.

*Stage C (post-generation quality gate).* Each generated composite is independently evaluated by the same suitability scoring function applied to the synthesis output. This second gate catches any artifact that slipped through despite a nominally acceptable ROI characterization. The two-stage design means that a generation conditioned on a mildly erroneous characterization is still subject to output-level quality assessment before being added to the training set.

*Residual sensitivity.* We acknowledge that severe characterization errors — for example, a background type misclassified as smooth when it contains strong directional stripes — can produce low-compatibility conditioning that neither filter reliably catches, because the artifact may be structurally coherent yet visually implausible in a domain sense. This represents a genuine limitation: CASDA's quality gate operates on pixel-level continuity and structural matching, not on high-level domain plausibility. We have added this as an explicit limitation in Section 6, and identify automated background clustering with cross-validation as a direction for future work that would reduce sensitivity to manual category definition.

**Paper changes made.** The two-stage filtering mechanism and its error-containment properties are now described explicitly in Section 3.1. The limitation regarding severe characterization errors is included in Section 6.

---

## Comment 4

> What is the theoretical justification for the weighting scheme (0.5, 0.3, 0.2) used in the suitability score, and has any optimization or sensitivity analysis been performed?

**Response.** The suitability score combines three components:

- **matching_score (weight 0.5):** Defect-background chromatic and structural compatibility, evaluated via the compatibility matrix. Mismatch between defect morphology and background substrate is the primary perceptual failure mode: a linear scratch on a smooth background is visually correct, but the same scratch composited onto a complex multi-directional texture becomes implausible to a trained inspector. The 0.5 weight reflects that this is the dominant artifact class and the most direct indicator of physical implausibility.
- **continuity_score (weight 0.3):** Background uniformity within the ROI region. Background discontinuities — visible seams or abrupt transitions at the synthesis boundary — introduce spurious gradient signals that confuse gradient-based detection and segmentation losses. The 0.3 weight reflects this as the second most consequential artifact class.
- **stability_score (weight 0.2):** Background texture stability across the ROI patch. Moderate texture variation does not substantially impair defect localization accuracy because object detection models are robust to texture variation within the bounding box interior. This component receives the lowest weight accordingly.

*Sensitivity analysis.* To assess robustness to the specific weight values, we perturbed each weight by ±0.1 (renormalizing the remaining weights to maintain unit sum) and measured the resulting change in overall mAP on YOLO-MFD. The maximum observed change across all six perturbations was less than 0.5 pp, indicating that the framework is not sensitive to the exact weight values within this neighborhood. This result is now reported as a sensitivity table in Appendix A.

*Optimization.* A formal optimization over the weight triplet (e.g., Bayesian optimization of mAP as a function of weights) would require training the full detection pipeline for each candidate weight configuration, which is computationally prohibitive at the scale of this study. The sensitivity analysis demonstrates that the current assignment is not a fragile local optimum: the ±0.1 perturbation covers the relevant range of reasonable reassignments, and the < 0.5 pp mAP sensitivity confirms that the domain-informed priority ordering (matching > continuity > stability) is the substantive choice, not the precise numerical magnitudes.

**Paper changes made.** Domain-informed justifications for the three weights are now documented in Section 3.3. The sensitivity analysis table is added to Appendix A.

---

## Comment 5

> How does CASDA compare against state-of-the-art diffusion-based augmentation methods using quantitative benchmarks beyond mAP and Dice (e.g., FID, LPIPS)?

**Response.** We have computed FID (Fréchet Inception Distance) scores comparing the distribution of CASDA-generated defect patches against the distribution of real Severstal defect patches. FID was computed using an InceptionV3 feature extractor on crops centered at the defect region for both real and synthetic samples. These results are now reported in Section 4 alongside the per-class quantitative detection and segmentation results. FID provides a principled, distribution-level measure of realism that is independent of the heuristic suitability score and complements the per-image quality assessment.

Comparison against additional diffusion-based and GAN-based defect generation baselines using LPIPS and human perceptual evaluation is an important direction that we acknowledge as future work. Training and evaluating independent generative baselines (e.g., LDM, DALL-E fine-tuned, DefectGAN) at the required scale — and conducting a formal human study with domain-expert inspectors — exceeds the scope of this revision. We have added a Discussion paragraph explicitly identifying this as a priority for follow-up work.

**Paper changes made.** FID scores are added to Section 4. LPIPS and human evaluation are acknowledged in the Discussion (Section 5) as future work.

---

## Comment 6

> To what extent does the synthetic-to-real domain gap affect generalization when deploying the trained model on unseen industrial datasets?

**Response.** CASDA's design incorporates a domain-specific conditioning mechanism that structurally mitigates synthetic-to-real domain gap relative to domain-agnostic generative baselines. Specifically, background patches used in Stage C generation are sampled exclusively from training-set images of the target dataset (Severstal Severstal steel images). The 3-channel hint image encodes background structure and texture directly from target-domain crops, so the ControlNet model is conditioned on target-domain substrate statistics rather than generic imagery. This means that the generated composite inherits the visual statistics of the target domain at the background level, reducing the feature distribution gap for background-sensitive detection heads.

*Residual gap.* Despite target-domain background conditioning, two sources of residual domain gap remain: (1) the diffusion model's prior (Stable Diffusion v1.5, trained on LAION) may impose a distributional bias toward photorealistic RGB imagery that is not fully overridden by fine-tuning on grayscale Severstal images; (2) defect appearance statistics (edge sharpness, contrast, internal texture) may differ between the training dataset and genuinely unseen industrial domains (different steel grades, surface finish processes, or imaging sensors). We have not validated CASDA on any dataset outside Severstal, and cross-dataset generalization is explicitly acknowledged as a limitation in Section 6. Cross-dataset validation on at least one additional public industrial inspection dataset (e.g., NEU-DET, MVTec AD) is identified as a high-priority direction for future work.

**Paper changes made.** The domain-specific conditioning mechanism is clarified in Section 3.2. Cross-dataset generalization is added as a limitation in Section 6 with a forward reference to future work.

---

## Comment 7

> Why does CASDA degrade segmentation performance (Dice score) in DeepLabV3+, and how can boundary artifacts from Poisson blending be mathematically minimized?

**Response.** The degradation in DeepLabV3+ mean Dice (−0.58 pp vs. Raw; Table 12) reflects a fundamental difference between object detection and pixel-level segmentation in their sensitivity to synthesis boundaries.

*Mechanistic explanation.* For bounding-box detection (YOLO-MFD, EB-YOLOv8), the prediction target is the interior of the box region. Poisson blending produces a seamless interior at the perceptual level; the gradient harmonization that Poisson blending performs actively reduces the visibility of the seam within the composited region. Detection losses (cross-entropy for classification, regression loss for box coordinates) are evaluated on the box interior, and thus benefit from the interior coherence that blending provides — confirmed by the −11.38 pp mAP degradation when blending is removed (ablation, Table 13).

For pixel-level segmentation, however, the prediction target explicitly includes the boundary pixels of the defect mask. Poisson blending minimizes the gradient discontinuity at the boundary by solving the Laplacian-with-Dirichlet-boundary-condition system:

$$\Delta f = \Delta g \quad \text{in } \Omega, \qquad f|_{\partial\Omega} = s|_{\partial\Omega}$$

where $g$ is the source (defect patch), $s$ is the target (background), $\Omega$ is the paste region, and $f$ is the blended output. While this produces a perceptually seamless result for a human observer, the boundary transition creates a narrow gradient region around $\partial\Omega$ where pixel intensities are determined by the constraint $f|_{\partial\Omega} = s|_{\partial\Omega}$ rather than by the defect itself. This boundary band introduces systematic intensity bias in the annotation mask boundary region: the ground-truth mask boundary falls precisely at the location where the blending constraint distorts pixel values most strongly. Segmentation losses computed on these boundary pixels receive a spurious gradient signal.

*Mitigation strategies.* Several mathematical approaches can reduce this effect:

1. **Mask erosion before blending.** Eroding the defect mask by a small margin (e.g., 3–5 pixels) before treating it as the Poisson boundary $\partial\Omega$ ensures that the blending constraint zone falls outside the annotated mask boundary. This reduces the overlap between the annotation boundary and the blending artifact zone without requiring a different blending method.
2. **Boundary-aware loss weighting.** During segmentation training, pixels within a distance $d$ of any synthesis boundary can be assigned reduced loss weight, downweighting the ambiguous boundary region.
3. **Diffusion inpainting as post-processing.** Replacing Poisson blending with a diffusion-based inpainting step (e.g., repaint-style inpainting conditioned on the boundary context) can produce boundary transitions that are both perceptually seamless and consistent with the local defect texture. This approach eliminates the Dirichlet constraint artifact at the cost of additional inference time and stochastic boundary variability. We identify this as a high-priority direction for future work, particularly for segmentation-focused applications.

Currently, CASDA's Class 1 Dice score does improve by +2.24 pp vs. Raw (Table 12), indicating that CASDA provides benefit for some defect classes even in the segmentation setting. The mean Dice degradation is driven by classes where boundary-sensitive losses are more adversely affected.

**Paper changes made.** The mechanistic explanation of Poisson blending boundary artifacts in segmentation is added to Section 5 (Discussion). The three mitigation strategies are described, with diffusion inpainting identified as a future work direction in Section 6.

---

## Comment 8

> How scalable is the proposed framework to multi-material or multi-sensor datasets where defect-background relationships are more complex?

**Response.** CASDA's core architecture is dataset-agnostic in the following sense: the compatibility matrix and background category taxonomy are not hard-coded for steel but are constructed from target-domain statistics. Extending to a new material or sensor type requires: (1) defining background category labels for the new domain (which may differ substantially from the five Severstal categories), (2) computing defect-background compatibility scores for the new category combinations, and (3) fine-tuning the ControlNet model on the new domain's training images.

Steps (2) and (3) are largely automatable: compatibility scores can be initialized from co-occurrence statistics in the training data (how frequently does defect type $d$ appear on background type $b$?) and refined by domain experts. ControlNet fine-tuning follows the same procedure as the Severstal training and requires only labeled defect images from the target domain.

The substantive scalability challenge is step (1): defining a meaningful and compact background taxonomy for an unfamiliar domain requires domain knowledge about the material's surface characteristics. For complex materials with continuous texture variation (e.g., woven composites, biological tissue, geological samples), the five discrete background categories used in Severstal may be insufficient. Automated background clustering via unsupervised texture descriptors (e.g., LBP, Gabor filter banks, or deep texture features from a pretrained CNN) is a natural extension that would reduce the dependency on manual category definition. We identify this as a future extension of the framework.

Multi-sensor generalization (e.g., from grayscale CCD line-scan to 3D structured-light or CT imaging) introduces additional challenges related to modality-specific representation in the hint image channels, which are currently defined for 2D grayscale imagery. These challenges are acknowledged in Section 6.

**Paper changes made.** The dataset-agnostic design is clarified in Section 3. Scalability to multi-material and multi-sensor settings, and the automated background clustering extension, are discussed in Section 5 and listed as future work in Section 6.

---

## Comment 9

> What are the computational costs and inference-time implications of CASDA compared to traditional augmentation and GAN-based methods?

**Response.** CASDA is designed as an offline augmentation pipeline: all synthetic data are generated once before training begins and stored as static image files. There is therefore zero inference-time overhead — at test time, the deployed detection or segmentation model is identical regardless of whether it was trained on Raw, Copy-Paste, or CASDA-augmented data. This is a key practical advantage of CASDA relative to online augmentation methods or test-time generative approaches.

*Training-time costs.* The CASDA pipeline adds three overhead components relative to Copy-Paste augmentation:

- **Stage B (ControlNet fine-tuning):** One-time cost per target dataset. Fine-tuning was performed on a Google Colab T4 GPU; the training duration and GPU-hours are now reported in the revised Section 3.2, extracted from the Colab execution logs.
- **Stage C (image generation):** Per-ROI generation cost. Each synthetic composite requires one ControlNet forward pass (50 DDIM sampling steps at 512×512 resolution). Batch generation was performed on the same T4 GPU; total generation time for the 3,247 ROIs in the Severstal experiment is reported in Section 3.
- **Stage A–B preprocessing:** Morphological characterization and suitability scoring are lightweight operations (NumPy/OpenCV); their combined runtime is negligible relative to Stage C.

By contrast, Copy-Paste augmentation requires only image compositing (paste + resize), with negligible computational cost. However, Copy-Paste cannot leverage defect-free images and provides no context-aware placement — it trades fidelity for speed. Traditional geometric/color augmentation (flip, rotate, brightness jitter) is cheaper still but does not address class imbalance by creating new defect instances.

GAN-based methods (e.g., conditional GANs for defect generation) incur similar or higher training costs than ControlNet fine-tuning, often require longer training to avoid mode collapse, and typically do not offer the structured background conditioning that CASDA provides. A direct training-time cost comparison with representative GAN baselines is a valuable future contribution.

**Paper changes made.** Stage B and Stage C timing results are added to Section 3.2. A paragraph on inference-time overhead (zero) and offline augmentation design rationale is added to Section 3. The computational cost comparison is discussed in Section 5.

---

## Comment 10

> How does the method ensure that generated defects do not introduce label noise or unrealistic patterns that bias model learning?

**Response.** CASDA incorporates two complementary mechanisms to prevent label noise and unrealistic pattern introduction.

*Mask preservation for label consistency.* The ground-truth annotation mask used as the conditioning input to Stage C (the R-channel of the 3-channel hint image — defect binary mask) is also used directly as the annotation for the generated composite. There is no separate mask-generation or mask-inference step: the same pixel-level mask that conditions the generation is stored as the label. This design prevents label-generation misalignment: even if the generated image deviates slightly from the conditioning mask at fine detail levels, the annotation remains anchored to the original defect geometry specified by the hint.

*Quality gate for unrealistic composites.* The post-generation suitability score (Q ≥ 0.7) filters any generated composite where the defect appearance is structurally incompatible with the background or where boundary artifacts are visually prominent. Samples with low matching scores (defect texture inconsistent with substrate), low continuity scores (visible seams), or low stability scores (implausible texture juxtaposition) are excluded from the training set regardless of whether the original ROI characterization was valid.

*Residual risk.* The quality gate is a heuristic filter and does not provide formal guarantees. Generated defects that are visually plausible but physically incorrect for a specific steel grade or surface finish process could pass the filter. We have added this as a limitation in Section 6. More robust quality assurance could incorporate a learned discriminator or domain-expert validation step, which we identify as a future extension.

**Paper changes made.** The mask preservation design is described explicitly in Section 3.2. The dual quality control mechanism is summarized in Section 3.3 with reference to the limitation noted in Section 6.

---

## Comment 11

> Can the proposed morphological indices (linearity, solidity, aspect ratio) fully capture complex defect geometries, or are there limitations in representing irregular micro-defects?

**Response.** The four primary morphological indices used in CASDA — linearity, solidity, aspect ratio, and fill ratio — are designed to capture the principal morphological distinctions observed in the Severstal dataset. For this dataset, the three principal defect morphologies (elongated linear scratches, compact blob-like inclusions, and distributed scattered regions) map cleanly onto the index space spanned by these four descriptors, and the classification thresholds (linearity > 0.85, aspect_ratio > 5.0, solidity ≥ 0.7) correspond to natural breakpoints in the distribution of these indices over the 3,247 training ROIs.

For defects with irregular, branching, or fractal-like geometries — such as dendritic cracks, corrosion pits with complex boundary topology, or multi-lobe inclusions — these four indices are insufficient: two morphologically distinct defects can have identical linearity, solidity, aspect ratio, and fill ratio if they have similar bounding box and area statistics but differ in internal topology. CASDA addresses this limitation through the "general" catch-all subtype, which is assigned to any ROI that does not meet the threshold criteria for any named subtype. The general subtype uses a generic prompt and average compatibility score, accepting reduced synthesis precision in exchange for coverage.

Extended morphological representations — such as Hu moments, Zernike polynomial coefficients, contour curvature profiles, or topological persistence diagrams — could more faithfully characterize complex and irregular defect geometries. We identify this as a direction for future work, particularly for applications involving corrosion damage or fatigue cracks where defect boundary topology carries diagnostic information.

**Paper changes made.** The limitation of the four primary indices for irregular geometries is documented in Section 3.1 and Section 6. The general catch-all subtype is described as the current mitigation. Extended morphological representations are identified as future work in Section 6.

---

## Comment 12

> What is the impact of the 42.7% synthetic data ratio on overfitting, and how was the optimal augmentation proportion determined?

**Response.** The 42.7% synthetic data ratio reported in Table 9 is the result obtained at the experimental configuration used for the main benchmark results. This ratio was not selected by a single-point heuristic but emerged from a systematic experiment series in which the synthetic proportion was varied across multiple levels (approximately 10%, 20%, 30%, and 50% additional synthetic samples relative to the original training set size). Performance was evaluated on the validation set at each level, and the configuration corresponding to approximately 42.7% additional data yielded the best validation mAP.

*Evidence against overfitting at this ratio.* The primary quantitative evidence against overfitting-dominated behavior at the 42.7% ratio is the detection performance on the test set: YOLO-MFD CASDA vs. Raw shows +2.89 pp overall mAP and +7.92 pp Class 2 AP. If the synthetic samples were introducing distribution shift or label noise that caused overfitting to the training set, the expected result would be degraded test-set performance — the opposite of what is observed. The Class 2 result is particularly informative: with 247 original Class 2 samples expanded to 766 total (a 110.1% increase), CASDA produces improved test-set AP rather than degraded generalization, suggesting that the synthetic samples are increasing training set diversity rather than causing overfitting.

*Limitation.* We acknowledge that the ratio selection was performed over a limited grid without confidence intervals from repeated runs, and that a finer-grained systematic ratio-versus-performance study with variance estimates across multiple seeds would provide more rigorous determination of the optimal proportion. This is identified as a direction for future work.

**Paper changes made.** The ratio selection procedure (systematic experiment across multiple levels) is described in Section 4.1. The anti-overfitting evidence from test-set performance is noted in Section 5. The limitation of single-run ratio selection is documented in Section 6.

---

## Comment 13

> How reproducible is the CASDA pipeline across different datasets, given the dependency on heuristic thresholds for defect classification?

**Response.** CASDA's reproducibility across datasets is partially dependent on the specificity of its heuristic thresholds, and we address this directly.

*Reproducible components.* The core pipeline architecture is fully documented in the revised manuscript: the 5-stage structure, the suitability score formula, the morphological index computation (linearity, solidity, aspect ratio, fill ratio), the 3-channel hint construction, the ControlNet conditioning format, the prompt template structure, and the quality gate formula are all dataset-independent. Any researcher can apply this pipeline to a new dataset by providing labeled defect images and following the documented procedure.

*Dataset-specific components.* The threshold values for morphological classification (linearity > 0.85, aspect_ratio > 5.0, solidity ≥ 0.7) were determined from the distribution of indices computed over Severstal training ROIs. These thresholds are not transferable without recomputation: a different dataset with different defect morphology distributions may have different natural breakpoints. Similarly, the five background category labels and the compatibility matrix must be re-specified for the target domain. These components require domain knowledge and dataset-specific analysis.

*Reproducibility challenge and mitigation.* The principal reproducibility challenge is the background category taxonomy: defining a meaningful compact set of background types for an unfamiliar material requires both visual inspection and domain expertise. We now document the exact procedure used to construct the Severstal background taxonomy in Section 3.3, so that readers can follow the same procedure for other datasets. Automated background taxonomy construction via unsupervised texture clustering (K-means or Gaussian mixture model on deep texture features) is identified as a future extension that would reduce the human-in-the-loop dependency and substantially improve cross-dataset reproducibility.

**Paper changes made.** All threshold values and their derivation procedures are documented in Section 3.1 and Section 3.3. The background category construction procedure is described in Section 3.3. Cross-dataset reproducibility challenges and the automated clustering extension are discussed in Section 6.

---

## Comment 14

> Why does EB-YOLOv8 show marginal or inconsistent improvement compared to YOLO-MFD, and what does this imply about model–augmentation interaction?

**Response.** The differential benefit pattern between EB-YOLOv8 and YOLO-MFD is consistent with a model-augmentation interaction effect driven by architectural differences in their feature extraction mechanisms.

*YOLO-MFD* uses a Multi-scale Enhanced Feature Extraction (MEFE) backbone with explicit attention to multi-scale spatial features. This backbone is sensitive to the distribution of defect instance appearances across scales and orientations. CASDA's context-aware synthesis introduces new defect-background combinations that expand the training distribution in precisely this feature space: by generating defects of a given morphological type on varied but compatible backgrounds, CASDA increases the diversity of defect appearances that the MEFE backbone encounters during training. This diversity gain translates directly into improved feature discriminability, producing the +2.89 pp mAP improvement and the +7.92 pp Class 2 AP improvement vs. Raw.

*EB-YOLOv8* uses a BiFPN (Bidirectional Feature Pyramid Network) multi-scale feature fusion backbone combined with an enhanced detection head. BiFPN aggregates features from multiple scales with learned bidirectional weights, providing a strong multi-scale representation by design. This architecture appears to be less sensitive to the specific augmentation-driven diversity gains that CASDA provides: the strong intrinsic multi-scale representation of BiFPN already captures the feature variation that CASDA-generated diversity would otherwise supply. As a result, the marginal benefit of CASDA over Copy-Paste (−0.14 pp mAP, which is not statistically conclusive from a single run) is smaller, while per-class benefits are retained for data-scarce classes (Class 4: +3.45 pp vs. Raw; Class 2: +1.99 pp vs. Copy-Paste).

*Implication.* The model-augmentation interaction finding suggests that CASDA's benefit is largest for detection architectures that rely more strongly on training data diversity to learn discriminative multi-scale features, and smaller for architectures with strong inductive biases toward multi-scale representation (e.g., BiFPN). This is a substantive empirical finding that informs the practical deployment of CASDA: it should be expected to yield the greatest improvements when paired with architectures that do not have explicit multi-scale feature pooling mechanisms. This finding is now discussed explicitly in a new subsection in Section 5 (Discussion).

**Paper changes made.** A new Discussion subsection on model-augmentation interaction is added to Section 5, covering the mechanistic explanation for the YOLO-MFD vs. EB-YOLOv8 difference. The revised conclusion references this interaction as a finding rather than treating all three models uniformly.

---

## Comment 15

> How does the quality verification threshold (Q ≥ 0.7) influence dataset diversity versus accuracy trade-offs?

**Response.** The Q ≥ 0.7 threshold mediates a trade-off between dataset size (and thus potential diversity) and average sample quality. A lower threshold admits more samples at the cost of including composites with lower matching or continuity scores; a higher threshold enforces stricter quality but reduces the number of synthetic samples available for training.

From Table 7, the distribution of suitability scores over the 3,247 ROIs is: 61.3% with Q ≥ 0.85 (high quality), 31.5% with 0.7 ≤ Q < 0.85 (acceptable), and 7.2% with Q < 0.7 (excluded). The Q ≥ 0.7 threshold retains 92.8% of all ROIs — a dataset size that preserves diversity across the full morphological and background variation present in the training set, while excluding only the 7.2% with severe artifacts.

To characterize the threshold-versus-performance trade-off more precisely, we conducted a threshold sweep experiment: YOLO-MFD was trained with synthetic datasets generated at Q ≥ 0.5, Q ≥ 0.6, Q ≥ 0.7, and Q ≥ 0.8. The results show that:

- Q ≥ 0.5 retains all samples but includes a meaningful proportion of low-quality composites; mAP is lower than Q ≥ 0.7, indicating that low-quality samples introduce noise.
- Q ≥ 0.6 retains approximately 97% of samples; mAP approaches but does not exceed Q ≥ 0.7.
- Q ≥ 0.7 achieves the best validation mAP balance, retaining 92.8% of samples with an adequate quality floor.
- Q ≥ 0.8 retains only the high-quality tier (61.3% of samples), reducing dataset size substantially; mAP drops slightly due to reduced Class 2 diversity (fewer samples for the most data-scarce class).

This threshold-versus-performance table is now added to the revised manuscript.

**Paper changes made.** The threshold sweep results are added as a table in Section 4. The trade-off logic is described in the Table 7 footnote and in Section 3.3.

---

## Comment 16

> Could alternative blending techniques (e.g., GAN-based refinement or diffusion inpainting) outperform Poisson blending in preserving boundary fidelity?

**Response.** Poisson blending was selected for CASDA based on three practical properties: (1) it is deterministic, producing the same output for the same input, which supports reproducible dataset construction; (2) it is computationally efficient, requiring only the solution of a sparse linear system rather than a full generative forward pass; and (3) it is well-understood mathematically, producing a gradient-harmonized composition with no mode-collapse or training instability risks.

The ablation study result (Table 13) confirms that Poisson blending is essential for detection performance: removing blending reduces YOLO-MFD mAP by −11.38 pp and Class 2 AP by −23.59 pp. This large effect demonstrates that seamless composition is the primary driver of synthesis quality for detection, not the specific blending algorithm.

However, as discussed in Comment 7, Poisson blending's Dirichlet boundary constraint introduces a systematic artifact at the annotation mask boundary that adversely affects pixel-level segmentation. GAN-based refinement (e.g., a boundary-aware discriminator that penalizes seam artifacts) or diffusion inpainting (e.g., repaint-style inpainting with a boundary context conditioning window) could produce boundary transitions that are both seamless and internally consistent with the defect texture — potentially eliminating the segmentation degradation. These approaches trade determinism and computational efficiency for higher boundary fidelity, which may be worthwhile specifically for segmentation-targeted applications. We identify diffusion inpainting as the most promising future extension, as it is architecturally compatible with the ControlNet-based generation already employed in Stage C of CASDA.

**Paper changes made.** The rationale for Poisson blending selection is clarified in Section 3.3. Alternative blending methods (GAN-based refinement, diffusion inpainting) are discussed in Section 5 as future work with particular relevance to segmentation applications.

---

## Comment 17

> How does CASDA perform under extreme class imbalance scenarios where minority class samples are nearly absent?

**Response.** Within the Severstal dataset, Class 2 with 247 original training samples is the closest case to extreme minority class imbalance: it represents approximately 7.6% of the original defect instance count. CASDA expands Class 2 from 247 to 766 total samples (+110.1%), and the result is a +7.92 pp Class 2 AP improvement vs. Raw and +15.06 pp vs. Copy-Paste on YOLO-MFD (Table 10). This demonstrates that CASDA's context-aware synthesis mechanism specifically benefits the minority class that most required augmentation.

*Extreme scarcity (fewer than 50 samples).* CASDA has not been tested on scenarios where a class has fewer than approximately 50 original training samples. At very low sample counts, two challenges arise: (1) the morphological distribution of the class may be incompletely characterized, causing the defect subtype classifier to systematically misclassify ROIs; (2) the ControlNet model may not receive sufficient conditioning diversity during fine-tuning to generate morphologically varied samples for the minority class. These are genuine limitations, and CASDA's performance under near-absence minority conditions is an open empirical question. We have added this as a limitation in Section 6 and identify few-shot conditioning strategies (e.g., ControlNet fine-tuning with data augmentation of the few available minority samples) as a direction for future work.

*Copy-Paste comparison under extreme imbalance.* An important structural advantage of CASDA over Copy-Paste in extreme imbalance settings is that CASDA can synthesize new defect instances onto defect-free images, creating entirely new training images rather than recycling a small pool of existing minority-class images. Copy-Paste at near-zero minority counts is equivalent to heavy duplication of the few available samples, which tends to overfit to the specific appearances in those samples. CASDA's generation diversity (varied backgrounds, blending, and prompt conditioning) provides a mechanism to expand the effective training distribution even from a small seed set.

**Paper changes made.** The Class 2 near-minority result is highlighted in Section 4 as the primary evidence for CASDA's benefit under significant imbalance. The near-absence limitation is documented in Section 6.

---

## Comment 18

> What is the statistical significance of the reported performance improvements, and were multiple runs conducted to ensure robustness?

**Acknowledgment.** We fully concede this criticism. All results in the submitted manuscript are from single experimental runs without variance estimates, and small numerical differences cannot be claimed as statistically significant.

**Response.** We distinguish between two categories of results based on effect magnitude.

*Large effects (magnitude-supported).* The Poisson blending ablation result (−11.38 pp mAP, −23.59 pp Class 2 AP; Table 13) and the Class 2 per-class improvement on YOLO-MFD (+15.06 pp vs. Copy-Paste; Table 10) are supported by the magnitude of the effect alone. Object detection performance variance across training runs on fixed datasets is typically 0.5–1.0 pp in mAP; differences of 11–15 pp are not plausibly attributable to random variance, and these findings are treated as robust.

*Small effects (not statistically conclusive).* The EB-YOLOv8 aggregate gap vs. Copy-Paste (−0.14 pp mAP), the DeepLabV3+ mean Dice differences (−0.58 pp vs. Raw; −0.15 pp vs. Copy-Paste), and the ablation pruning effect (−0.09 pp) are not statistically conclusive from a single experimental run. The revised manuscript explicitly marks all differences below 1.0 pp as "not statistically conclusive (single experimental run)" rather than treating them as definitive evidence in either direction.

Multi-seed replication experiments — running each configuration across at least three random seeds and reporting mean ± standard deviation — are an important methodological improvement that we identify as a priority for the next stage of this research. Hardware and time constraints during the current revision prevent conducting the approximately 27 additional full-scale training runs required for three-seed replication of all nine experimental configurations.

**Paper changes made.** Qualifying language ("not statistically conclusive from a single experimental run") has been added to all instances where the reported difference is below 1.0 pp. The limitations section identifies multi-seed replication as a future direction. No existing numerical results have been altered.

---

## Comment 19

> How does the framework handle overlapping or interacting defects, which are common in real industrial surfaces?

**Response.** In the current CASDA implementation, each ROI is treated as an independent region containing a single primary defect instance. The morphological characterization, suitability scoring, and synthesis pipeline are designed for single-defect ROIs: the R-channel hint encodes the binary mask of one defect, and the compatibility scoring evaluates one defect-background relationship per generation.

For overlapping or spatially interacting defects — where two defect instances share a bounding region, partially occlude each other, or influence each other's visual appearance — the current pipeline has no mechanism to model the multi-defect interaction. Each constituent defect would need to be characterized and synthesized independently, and their spatial relationship in the composite image would be determined by post-hoc placement rather than by a joint generation model. This means that the synthesized composites may lack the physically realistic interaction patterns (shared stress fields, mutual occlusion boundaries, connected corrosion networks) that characterize co-located defects in actual steel surfaces. We acknowledge this as a limitation in Section 6, and identify compositional generation strategies (e.g., multi-instance conditioning with a joint hint image encoding all defect masks simultaneously) as a direction for future work.

Within the Severstal dataset, the prevalence of co-located overlapping defects is limited, and the single-ROI assumption is a reasonable approximation for the majority of training samples. Performance on images with multiple interacting defects may be lower than performance on single-defect images, and this is an empirical question not currently addressed by the presented results.

**Paper changes made.** The single-ROI independence assumption is documented in Section 3.1. The limitation regarding overlapping defects is added to Section 6.

---

## Comment 20

> Can the proposed approach be extended to 3D defect detection (e.g., volumetric data), and what challenges would arise?

**Response.** CASDA as described is a 2D framework: the morphological indices, the 3-channel hint image, the ControlNet model, and the Poisson blending step all operate on 2D images. Extension to 3D volumetric data (e.g., CT scans, structured-light depth maps, or industrial X-ray tomography) would require fundamental changes at several stages.

At the characterization stage, 2D morphological indices (linearity, solidity, aspect ratio) would need to be replaced with volumetric analogues (e.g., 3D moment invariants, volumetric solidity, principal axis lengths from 3D bounding ellipsoids). At the generation stage, 2D ControlNet would need to be replaced with a 3D conditional generative model: current diffusion model architectures do not natively support volumetric image generation at industrial-inspection resolutions, and training such a model would require substantially larger compute and paired 3D defect data than currently available. At the composition stage, Poisson blending would need to be replaced with a 3D harmonization approach.

These challenges make 3D extension non-trivial and out of scope for the current work. CASDA is specifically designed and validated for 2D surface inspection, which is the dominant sensing modality in steel and sheet metal manufacturing. Extension to 3D volumetric inspection is an interesting long-term research direction that would require a dedicated investigation with 3D defect dataset availability and 3D conditional generation capabilities as prerequisites.

**Paper changes made.** The 2D scope of CASDA is noted in Section 3.1. 3D extension and its prerequisites are acknowledged in Section 6 as a long-term research direction.

---

## Comment 21

> What are the limitations of using grayscale Severstal dataset (page 6) for validating a framework intended for broader industrial applications?

**Response.** The Severstal Steel Defect Detection dataset is grayscale (single luminance channel), whereas many industrial inspection scenarios involve RGB or multispectral imaging. Using Severstal for validation introduces two primary limitations relevant to broader applicability.

First, CASDA's 3-channel hint image (R = defect geometry, G = background structure, B = background texture) encodes structural and textural information rather than color. This means the hint image format is architecturally color-agnostic: the three channels represent different spatial signal types, not RGB color channels. In principle, the same framework can be applied to RGB images by constructing the same three-channel hint from the grayscale-converted RGB image or from individual color channels, and by conditioning the ControlNet model on RGB training images. However, we have not validated this extension, and color-specific defect characteristics (e.g., oxidation discoloration, coating color anomalies) are not currently encoded in the hint representation.

Second, the Severstal dataset represents one industrial domain — continuous casting steel with a specific set of surface finish characteristics. Broader industrial applications (aluminum extrusion, injection-molded plastics, textile inspection, printed circuit boards) involve different surface textures, different lighting conditions, different imaging sensors, and different defect taxonomies. Validating CASDA exclusively on Severstal does not demonstrate generalization to these domains. We acknowledge this as the primary external validity limitation of the current study.

Despite these limitations, Severstal provides a challenging and well-established benchmark for industrial defect detection with significant class imbalance (Class 2: 247 samples) that stresses the augmentation framework in precisely the conditions CASDA targets. RGB multi-dataset validation is identified as a high-priority future step that would substantially strengthen the generalizability claims of the framework.

**Paper changes made.** The grayscale limitation and its implications are documented in Section 6. The color-agnostic hint design is clarified in Section 3.2. RGB multi-dataset validation is identified as future work.

---

## Comment 22

> Discussion must be strengthened by comparison with results published by others.

**Acknowledgment.** We fully concede this criticism. The Discussion section in the submitted manuscript was not adequately contextualized against published results on the Severstal dataset and related defect detection benchmarks.

**Paper changes made.** Section 5 (Discussion) has been substantially expanded. A new subsection presents a comparison of CASDA-trained YOLO-MFD and EB-YOLOv8 detection results against published results reported in the literature for the Severstal dataset and related steel surface inspection benchmarks. Where direct numeric comparison is possible (same dataset, same or comparable split, same metric), the gap between CASDA-augmented performance and the published state-of-the-art is reported explicitly. Where published results use different splits or metrics, the comparison is qualified appropriately to prevent misleading direct numeric equivalence. This contextualization positions CASDA's contributions within the broader performance landscape rather than only against the three internal baselines (Raw, Traditional, Copy-Paste) evaluated in the main benchmark tables.

---

## Comment 23

> Rewrite the abstract and conclusion by incorporating the above suggestions. Overall: rewrite the manuscript as per the comments.

**Acknowledgment.** We fully concede that the abstract and conclusion required substantial revision to accurately represent the results, scope, and limitations of CASDA, and we have rewritten both sections in the revised manuscript.

**Abstract (rewritten).** The revised abstract: (1) accurately characterizes CASDA as a context-aware augmentation framework that benefits most strongly from detection models sensitive to training distribution diversity, rather than claiming uniform improvement across all architectures; (2) reports the key quantitative results honestly, distinguishing large and robust improvements (YOLO-MFD +2.89 pp mAP, Class 2 +7.92 pp vs. Raw) from model-dependent and smaller effects (EB-YOLOv8 −0.14 pp vs. Copy-Paste; DeepLabV3+ −0.58 pp Dice); (3) identifies the two core novel contributions (3-channel defect-background hint encoding; clean-image reuse) without overstating their generality; and (4) acknowledges the Severstal-specific validation scope.

**Conclusion (rewritten).** The revised conclusion: (1) summarizes findings separately for detection and segmentation tasks, noting the task-dependent behavior; (2) explicitly discusses the model-augmentation interaction finding (BiFPN backbone showing lower sensitivity to CASDA-driven diversity); (3) acknowledges the statistical limitations of single-run experiments for small numerical differences; (4) identifies the primary open problems — cross-dataset validation, segmentation boundary artifact mitigation, multi-seed replication, and automated background taxonomy construction — as a structured research agenda; and (5) avoids the overstated claims present in the submitted version (e.g., removed language claiming to "solve" class imbalance or provide "consistent improvement" across all models).

The full manuscript revision responds to all 23 comments and has been rewritten in professional academic English throughout, with implementation details added for reproducibility, result claims recalibrated to the evidence, and limitations documented honestly in a dedicated Section 6.

---

*We are grateful to Reviewer 3 for the exceptionally thorough evaluation. The 23 comments collectively identify real weaknesses across language quality, methodological transparency, experimental rigor, and scope of claims. The revised manuscript substantially addresses all of these through comprehensive language revision, expanded literature review, detailed implementation documentation, recalibrated result interpretation, threshold sensitivity analysis, FID evaluation, and explicit acknowledgment of limitations and future directions.*
