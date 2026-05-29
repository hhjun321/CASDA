# Response to Reviewer 1

We thank Reviewer 1 for the constructive comments. We have revised the manuscript accordingly and address each point below.

---

## Comment 1

> A major weakness is that the manuscript contains no figures at all. For a computer vision paper, the absence of visual materials significantly reduces clarity and impact. The authors should include at least: (i) an overview figure of the CASDA pipeline, (ii) examples of ROI extraction and defect-type categorization, (iii) examples of generated synthetic defects, and (iv) visual comparisons between raw, copy-paste, and CASDA-augmented samples.

**Response:** We agree. Six figures have been added to the revised manuscript:

- **Figure 1** — CASDA pipeline overview (addresses (i))
- **Figure 2** — Geometric ROI characterization pipeline: four-step processing (crop → binary mask → geometric overlay → classified 256×256 ROI) for each defect subtype (addresses (ii))
- **Figure 3** — Background texture classification pipeline: processing sequence (crop → Canny edge density map → Sobel direction map → classified ROI) for each background type (addresses (ii))
- **Figure 4** — Representative classified ROI samples for each defect subtype (top) and background type (bottom) (addresses (ii))
- **Figure 5** — Three-channel hint image construction: R (defect geometry), G (surface orientation via Sobel), B (surface roughness via local variance), and final composite (addresses (iii))
- **Figure 6** — ControlNet-generated defect samples compared with the original ground-truth ROI under identical hint conditioning (addresses (iii))

Regarding (iv), direct cross-condition visual comparisons are provided quantitatively in Tables 12 and 13; a dedicated visual comparison figure is planned for future work.

---

## Comment 2

> Although the paper introduces multiple stages and scoring formulas, key implementation details are missing. The authors should explain how the compatibility matrix was built, how the threshold values were selected, what exact ControlNet and diffusion settings were used, and how prompt construction was implemented in practice.

**Response:** The following details have been added to the revised manuscript.

**Compatibility matrix (Table 6, Section 3.2.3).** The 5 × 4 matrix (5 background types × 4 defect subtypes, scores in [0.2, 1.0]) was defined by empirical calibration based on the visual identifiability of each defect–background pair and domain-informed co-occurrence intuition from the Severstal training set. The complete matrix is presented in Table 6.

**Classification thresholds (Section 3.2.1).** Linearity > 0.85 and aspect ratio > 5.0 → Linear Scratch; solidity ≥ 0.7 → Compact Blob. Thresholds were set from histogram inspection of geometric indices computed over all 3,247 training ROIs. The quality gate (Q ≥ 0.7) retains 92.8% of ROIs while excluding 7.2% with severe artifacts.

**ControlNet and diffusion settings (Section 3.2.2).** Base model: runwayml/stable-diffusion-v1-5 with lllyasviel/sd-controlnet-canny adapter. Fine-tuned for up to 20 epochs, AdamW (lr = 1×10⁻⁵, cosine scheduler, 50 warmup steps), min-SNR weighting (γ = 5.0), effective batch size 4, fp16 mixed precision, Google Colab T4 GPU. Inference: 30 denoising steps, guidance scale 7.5, ControlNet conditioning scale 0.7.

**Prompt construction (Section 3.2.2).** Prompts concatenate four semantic fields (F1: defect type, F2: background type, F3: directional texture, F4: surface stability). Example: *"a linear scratch defect on vertical striped metal surface with directional texture (pristine), class 2."*

---

## Comment 3

> The manuscript suggests that CASDA improves performance broadly, but the reported results are mixed across models. For EB-YOLOv8, CASDA does not clearly outperform all baselines, and the text should reflect this more carefully. The conclusions should be rewritten to avoid overstating the general effectiveness of the method.

**Response:** We agree and have revised the Results, Discussion, and Conclusion sections accordingly. Results are now reported as mean ± std over two independent seeds (42 and 456).

**YOLO-MFD.** CASDA did not improve overall mAP@0.5 (−2.50 pp vs. Raw; Table 12), and Class 2 AP also decreased vs. Raw (−4.59 pp), though it remained superior to Copy-Paste (+5.61 pp). The high standard deviation in YOLO-MFD Class 2 AP (±0.079) indicates instability at n = 2 seeds. These findings are reported transparently, and claims of broad superiority for YOLO-MFD have been removed.

**EB-YOLOv8.** CASDA improved overall mAP@0.5 (+2.60 pp vs. Raw, +4.74 pp vs. Copy-Paste; Table 13), with a large Class 1 gain (+10.03 pp) and substantial Class 2 improvement (+22.09 pp vs. Copy-Paste).

**Architecture-dependent sensitivity** is now discussed explicitly in Section 5. Formal hypothesis tests did not reach significance (α = 0.05) at n = 2, and this limitation is acknowledged. The ablation study from the original submission has been removed, as further controlled analysis is required before these results can be reported with sufficient rigor.

---

## Comment 4

> The literature review is somewhat limited and should be updated to better reflect the latest advances in this field. In particular, the authors should incorporate several recent representative studies on interpretable surrogate modeling, data-driven frameworks.

**Response:** Two new subsections have been added to Section 2.

**Section 2.5 — Data-Driven Augmentation for Industrial Inspection.** Reviews GAN-based and diffusion-based defect generation methods, and identifies their shared limitation: generation is conditioned on defect appearance alone without explicit background modeling, causing synthesized defects to exhibit texture distributions incompatible with the target substrate. CASDA directly addresses this gap through the compatibility matrix and 3-channel hint image.

**Section 2.6 — Aware Data Augmentation for Defect Detection.** Reviews GAN-based synthetic augmentation for rare defect classes, diffusion-based few-shot defect generation with mask-guided fine-tuning, and context-aware inpainting with boundary-coherence loss. These works confirm that defect-background awareness consistently improves downstream detection utility. CASDA formalizes this relationship through an explicit compatibility matrix and a 3-channel hint image that generalizes across defect subtypes and background categories.

---

## Comment 5

> There are many awkward expressions, grammatical errors, and informal phrases that reduce the professionalism of the manuscript.

**Response:** The entire manuscript has been comprehensively revised for language quality. Informal phrasing was eliminated, technical terminology was standardized throughout, grammatical errors in Sections 3 and 4 were corrected, and ambiguous sentence structures were rewritten.
