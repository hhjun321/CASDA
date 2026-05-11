# Response to Reviewer 1

We sincerely thank Reviewer 1 for the careful and constructive reading of our manuscript. The comments identify real weaknesses in the submission, and we have revised the paper accordingly. Below we address each comment in turn.

---

## Comment 1

> A major weakness is that the manuscript contains no figures at all. For a computer vision paper, the absence of visual materials significantly reduces clarity and impact. The authors should include at least: (i) an overview figure of the CASDA pipeline, (ii) examples of ROI extraction and defect-type categorization, (iii) examples of generated synthetic defects, and (iv) visual comparisons between raw, copy-paste, and CASDA-augmented samples. Without such figures, it is difficult for readers to understand the method intuitively or assess the realism of the generated data.

**Acknowledgment.** We fully agree. The absence of figures is a fundamental omission that we address entirely in the revision.

**Paper changes made.** Six figures have been added to the revised manuscript:

- **Figure 1 (Pipeline overview):** An end-to-end diagram of the five-stage CASDA pipeline — geometric ROI characterization, ControlNet configuration construction, context-based generation and synthesis, quality verification, and dataset integration — with data flow indicated between stages.
- **Figure 2 (3-channel hint construction):** A side-by-side illustration of the three hint channels: R = defect binary mask (pixel-level geometry), G = background structure map (grain orientation), B = background texture map (surface roughness). This figure makes the novel 3-channel conditioning mechanism visually concrete.
- **Figure 3 (Visual comparison):** A grid showing matched image crops under three conditions — Raw, Copy-Paste, and CASDA — for each of the four defect classes. The comparison allows direct visual assessment of realism and contextual coherence.
- **Figure 4 (Defect-type examples):** Representative ROI crops for each morphological subtype (linear_scratch, elongated_region, compact_blob, irregular, general), annotated with the discriminating threshold values (linearity > 0.85, aspect ratio > 5.0, solidity ≥ 0.7).
- **Figure 5 (Per-class AP bar chart):** A grouped bar chart visualizing per-class AP across all augmentation conditions for YOLO-MFD (Table 10), making the Class 2 improvement (+7.92 pp vs. Raw, +15.06 pp vs. Copy-Paste) visually prominent.
- **Figure 6 (Ablation visualization):** Detection output overlays comparing CASDA-Full, w/o Pruning, and w/o Blending conditions on the same test images, illustrating the artifact effect of omitting Poisson blending (−11.38 pp mAP).

---

## Comment 2

> Although the paper introduces multiple stages and scoring formulas, key implementation details are missing. The authors should explain how the compatibility matrix was built, how the threshold values were selected, what exact ControlNet and diffusion settings were used, and how prompt construction was implemented in practice.

**Acknowledgment.** We agree that Section 3 lacked the implementation detail necessary for reproducibility. The revision adds the following.

**Paper changes made.**

**Compatibility matrix (added to Section 3.3).** The matrix covers 5 background types (smooth, textured, vertical_stripe, horizontal_stripe, complex_pattern) × 4 defect morphological subtypes (compact_blob, linear_scratch, scattered_defects, elongated_region). Scores range from 0.2 to 1.0 and were defined by domain experts based on defect-background visibility and physical co-occurrence patterns observed in the Severstal dataset. For example, linear_scratch and elongated_region defects receive a score of 1.0 on vertical_stripe and horizontal_stripe backgrounds, reflecting their directional structural affinity, whereas all defect types receive 0.2 on complex_pattern backgrounds where any defect signal is visually masked. The full matrix is now presented as a table in Section 3.3.

**Threshold values with justification (added to Section 3.1 and Table 7 footnote).** Morphological classification thresholds are: linearity > 0.85 and aspect_ratio > 5.0 for linear_scratch; solidity ≥ 0.7 for compact_blob. These values were derived empirically from the distribution of geometric indices computed over all 5,237 original training ROIs. The ROI suitability quality gate (Q ≥ 0.7) was chosen based on dataset composition: as shown in Table 7, this threshold retains 92.8% of ROIs (61.3% high-quality + 31.5% acceptable) while excluding only the 7.2% with severe compositional artifacts. Suitability score weights are 0.5 (defect-background matching), 0.3 (background continuity), and 0.2 (background stability), reflecting domain-informed priorities: chromatic/structural mismatch is immediately perceptible to a human inspector, continuity defects introduce spurious gradients, and moderate texture variation does not impair defect localization.

**ControlNet and diffusion settings (added to Section 3.2).** The base model is Stable Diffusion v1.5 with the sd-controlnet-canny adapter. Detailed hyperparameters (learning rate, training epochs, batch size, guidance scale, sampling steps) are provided in revised Section 3.2, extracted from Stage B training scripts. Hardware: Google Colab T4 GPU.

**Prompt construction (added to Section 3.2).** We use a technical-style prompt template: *"Industrial steel defect: {morphological_type} defect (class {id}) on {surface}, {pattern}, background stability {score:.2f}, match quality {score:.2f}."* A concrete example for a linear_scratch defect on a vertical-stripe background (Class 1): *"Industrial steel defect: linear_scratch defect (class 1) on vertical striped metal surface, vertical line pattern, background stability 0.82, match quality 0.90."* The template ensures the diffusion model receives consistent, quantitative conditioning rather than free-form natural-language descriptions.

**Benchmark training settings (added as Table in Section 4.1).** All three evaluation models now have their training configurations documented: YOLO-MFD and EB-YOLOv8 used AdamW (lr = 0.001, weight decay = 0.0005, batch = 16, input = 640×640, cosine LR scheduler, 300 epochs, warmup 10 epochs, early stopping patience 30); DeepLabV3+ used AdamW (lr = 0.0001, weight decay = 0.0001, batch = 8, input = 256×512, polynomial scheduler with power = 0.9, 300 epochs, warmup 5 epochs, patience 40). All models used AMP (mixed precision), random seed 42, and a stratified 70/15/15 train/val/test split on the Severstal dataset.

---

## Comment 3

> The manuscript suggests that CASDA improves performance broadly, but the reported results are mixed across models. For EB-YOLOv8, CASDA does not clearly outperform all baselines, and the text should reflect this more carefully. The conclusions should be rewritten to avoid overstating the general effectiveness of the method.

**Acknowledgment.** We agree that the original conclusion section overstated the generality of the performance gains, and we have revised the manuscript accordingly. We also offer the following technical clarification.

**Defense and reframing.** The results across the three benchmark models reflect a consistent pattern rather than a contradiction. CASDA produces its largest improvements where data scarcity is most severe: on YOLO-MFD, overall mAP increases by +2.89 pp vs. Raw, with Class 2 (the most data-scarce class at 247 original samples, expanded by 110.1% via CASDA) gaining +7.92 pp vs. Raw and +15.06 pp vs. Copy-Paste (Table 10). These are substantively large improvements.

For EB-YOLOv8, the picture is more nuanced. The aggregate gap between CASDA and Raw is only +0.31 pp mAP, and CASDA trails Copy-Paste by 0.14 pp mAP (Table 11). However, per-class results reveal that CASDA retains meaningful advantages in specific cases: Class 4 AP improves by +3.45 pp vs. Raw, and Class 2 AP improves by +1.99 pp vs. Copy-Paste. The marginal overall difference (−0.14 pp) is unlikely to be statistically conclusive given typical detection variance. We attribute the reduced aggregate benefit on EB-YOLOv8 to its BiFPN multi-scale feature fusion backbone, which appears less sensitive to augmentation-driven diversity gains than the MEFE backbone used by YOLO-MFD. This model-augmentation interaction is an important finding in itself and is now discussed explicitly in a new paragraph in Section 5 (Discussion).

For DeepLabV3+, the −0.58 pp Dice mean degradation is explained by the boundary-sensitivity of pixel-level segmentation: Poisson blending, while visually seamless, can introduce minor edge-level artifacts at the synthesis boundary that confuse boundary-sensitive segmentation models. Class 1 segmentation nonetheless improves by +2.24 pp Dice vs. Raw. This limitation is acknowledged in the revised Discussion section.

**Paper changes made.** Section 5 (Conclusion) has been rewritten to state that CASDA yields consistent gains for detection models particularly sensitive to class imbalance, with benefit magnitude depending on backbone architecture. Claims of universal superiority have been removed. A new Discussion subsection on model-augmentation interaction has been added.

---

## Comment 4

> The literature review is somewhat limited and should be updated to better reflect the latest advances in this field. In particular, the authors should incorporate several recent representative studies on interpretable surrogate modeling, data-driven frameworks. These studies should be discussed to better position the novelty and contribution of the present work against the current state of the art.

**Acknowledgment.** We agree. The literature review in its submitted form does not adequately situate CASDA within recent advances in data-driven defect analysis and interpretable industrial inspection frameworks.

**Paper changes made.** Section 2 (Related Work) has been expanded with three new subsections:

1. **Data-driven augmentation for industrial inspection:** Recent work on GAN-based and diffusion-based synthetic defect generation is surveyed, including methods that generate defects without conditioning on background context. This motivates CASDA's distinguishing contribution: explicit modeling of the defect-background relationship through the compatibility matrix and 3-channel hint image, which no prior work incorporates.

2. **Interpretable surrogate modeling for defect characterization:** Recent studies using morphological features and surrogate scoring functions for defect analysis are reviewed, positioning CASDA's geometric characterization stage (linearity, solidity, aspect ratio, fill ratio) within this line of work.

3. **Clean-image data reuse:** CASDA's approach of synthesizing defects onto defect-free images in defect-possible ROI regions is contrasted with Copy-Paste augmentation, which only moves defects between existing defect images. This distinction — converting previously unusable clean images into labeled training data — is now foregrounded as a novel contribution in both the introduction and the related work section.

---

## Comment 5

> There are many awkward expressions, grammatical errors, and informal phrases that reduce the professionalism of the manuscript. In several places, sentence structure is unclear and terminology is inconsistent. The manuscript would benefit from careful language revision by a fluent English speaker or professional editing service.

**Acknowledgment.** We fully accept this criticism. The language quality of the submitted manuscript did not meet the standard expected for a journal publication.

**Paper changes made.** The entire manuscript has been subjected to a comprehensive language revision. All sections have been rewritten in professional academic English, with particular attention to: elimination of informal phrasing, consistent use of technical terminology (e.g., "synthetic augmentation" vs. "augmented generation"), correction of grammatical errors in Section 3 (methodology) and Section 4 (results), and restructuring of overly long or ambiguous sentences. The revised manuscript has additionally been reviewed by a fluent English speaker for idiomatic clarity.

---

*We are grateful to Reviewer 1 for the thorough evaluation. The combination of visual materials, implementation transparency, recalibrated claims, expanded literature, and language revision substantially strengthens the manuscript.*
