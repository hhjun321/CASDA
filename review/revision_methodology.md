# Methodology Section — Additions for Paper Revision

> **Instructions for the author:** Insert the passages below at the indicated locations in the Method section. Table placeholder numbers (e.g., [Table N]) must be renumbered to match the final manuscript. Figure placeholders (e.g., [Figure N]) must be assigned once the corresponding figures are inserted.

---

## Addition 1: Threshold Justification

**Insert:** After the paragraph that introduces the defect subtype classification (linear_scratch, elongated_region, compact_blob, irregular, general), immediately following the definition of the morphological indices.

---

The classification thresholds linearity > 0.85 and aspect ratio > 5.0 were not selected arbitrarily but were derived from the empirical distribution of morphological indices computed across all 3,247 Severstal training ROIs. Plotting the frequency histograms of both indices ([Figure N]) reveals bimodal distributions with a clear separation valley: linearity clusters around 0.4–0.6 for non-directional defects and above 0.85 for scratch-type defects, while aspect ratio exhibits a similar gap between compact blobs (< 3.0) and elongated features (> 5.0). These natural breakpoints in the training data ensure that the classification boundary aligns with the morphological structure of the defect population rather than with an ad hoc cutoff. The discriminative contribution of aspect ratio was further confirmed by the feature importance analysis reported in [Table 4], where aspect ratio accounts for 41.1% of the total morphological discriminative power across defect classes.

---

## Addition 2: Compatibility Matrix Construction

**Insert:** Before the paragraph describing Poisson blending (Stage 3 of the synthesis pipeline), after the paragraph introducing background type enumeration.

---

The synthesis target selection is governed by a compatibility matrix **M** ∈ ℝ^(4×5), where rows index defect types and columns index background types. Each entry M[d, b] ∈ [0.2, 1.0] encodes the visual compatibility between defect type *d* and background type *b*, reflecting both the expected co-occurrence frequency in real industrial images and the perceptual saliency of the defect against each background. Scores were assigned by domain experts based on two criteria: (i) defect-background visibility, i.e., whether the defect is clearly distinguishable against the background texture, and (ii) physical co-occurrence plausibility, i.e., whether the combination corresponds to realistic surface conditions observed on the Severstal production line.

The five background types are defined as follows:

| Background Type | Description |
|---|---|
| smooth | Uniform surface, no directional pattern, low spatial frequency |
| textured | Grainy surface with subtle isotropic texture |
| vertical_stripe | Directional pattern dominated by vertical lines (rolling direction) |
| horizontal_stripe | Directional pattern dominated by horizontal lines (transverse direction) |
| complex_pattern | Multi-directional overlapping patterns, high spatial frequency |

The full compatibility matrix is:

| | smooth | textured | vertical_stripe | horizontal_stripe | complex_pattern |
|---|---|---|---|---|---|
| linear_scratch | 0.8 | 0.6 | 1.0 | 0.4 | 0.2 |
| elongated_region | 0.6 | 0.8 | 0.6 | 0.6 | 0.2 |
| compact_blob | 1.0 | 0.8 | 0.4 | 0.4 | 0.2 |
| irregular | 0.6 | 0.6 | 0.4 | 0.4 | 0.2 |

Expert scores range from 0.2 (low compatibility; complex_pattern uniformly suppressed for all defect types due to masking effects) to 1.0 (optimal compatibility; smooth background for compact blobs, and vertical_stripe for linear scratches which are co-directional with rolling-direction defects). During synthesis, the background type is sampled with probability proportional to M[d, b] for defect type *d*, ensuring that high-compatibility pairings are generated more frequently while retaining diversity through low-probability samples.

---

## Addition 3: Prompt Template Description

**Insert:** In Stage 2 (ControlNet conditioning / prompt construction), after the sentence introducing the text conditioning mechanism.

---

Text prompts for the ControlNet generator follow a structured template designed to convey both the defect semantics and the surface context to the diffusion model:

```
"Industrial steel defect: {subtype_descriptor} defect (class {class_id}) on {surface_description},
{pattern_description}, background stability {stability_score:.2f}, match quality {match_score:.2f}"
```

The five defect subtype descriptors and five background descriptions used to instantiate the template are:

**Defect subtype → descriptor:**

| Subtype | Descriptor |
|---|---|
| linear_scratch | linear scratch |
| elongated_region | elongated region |
| compact_blob | compact blob |
| irregular | irregular-shaped |
| general | general surface defect |

**Background type → description:**

| Background Type | Description String |
|---|---|
| smooth | smooth uniform steel surface |
| textured | grainy textured steel surface |
| vertical_stripe | steel surface with vertical stripe pattern |
| horizontal_stripe | steel surface with horizontal stripe pattern |
| complex_pattern | steel surface with complex overlapping pattern |

The `stability_score` and `match_score` fields are populated with the numeric values computed during the suitability scoring stage (see Section [N]), providing the model with quantitative quality context that steers generation toward high-fidelity textures. For example, a representative prompt for a Class 2 linear scratch on a vertical-stripe background with high suitability is:

> "Industrial steel defect: linear scratch defect (class 2) on steel surface with vertical stripe pattern, vertical stripe pattern, background stability 0.91, match quality 0.95"

The 3-channel hint image accompanying this text prompt encodes: (R channel) the defect geometry mask derived from the original ROI annotation; (G channel) the background structure and grain orientation map extracted from the target patch; (B channel) the background texture roughness map. This multi-channel conditioning enables the ControlNet to simultaneously respect defect shape, surface directionality, and texture characteristics.

---

## Addition 4: ControlNet Training Settings

**Insert:** At the end of the Stage B / ControlNet fine-tuning subsection, before the transition to Stage C.

---

[Table N] summarizes the training configuration used for ControlNet fine-tuning.

**[Table N] ControlNet Fine-Tuning Configuration**

| Parameter | Value |
|---|---|
| Base diffusion model | Stable Diffusion v1.5 |
| Control adapter | sd-controlnet-canny |
| Hardware | Google Colab (NVIDIA T4 GPU) |
| Optimizer | AdamW |
| Learning rate | (see footnote *) |
| Batch size | (see footnote *) |
| Training epochs | (see footnote *) |
| Mixed precision | AMP enabled |
| Random seed | 42 |

(*) Detailed hyperparameters (learning rate, batch size, epochs) are extracted from the Stage B training scripts. Values are reported in the supplementary material / script repository. The base model and adapter selection follow the standard ControlNet canny-edge conditioning approach [cite HED/canny ControlNet paper].

---

## Addition 5: Benchmark Model Training Settings

**Insert:** At the beginning of Section 4 (Experiments) or at the end of Section 3 (Implementation Details), as a unified configuration table.

---

To ensure reproducibility, [Table N] provides the complete training configuration for all three benchmark models evaluated in this study.

**[Table N] Training Configurations for Benchmark Detection and Segmentation Models**

| Parameter | YOLO-MFD | EB-YOLOv8 | DeepLabV3+ |
|---|---|---|---|
| Base architecture | YOLOv8s + MEFE module | YOLOv8s + BiFPN | ResNet-101 encoder |
| Task | Object detection | Object detection | Semantic segmentation |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 0.001 | 0.001 | 0.0001 |
| LR scheduler | Cosine annealing | Cosine annealing | Polynomial decay (power = 0.9) |
| Epochs | 300 | 300 | 300 |
| Batch size | 16 | 16 | 8 |
| Input resolution | 640 × 640 | 640 × 640 | 256 × 512 |
| Mixed precision | AMP enabled | AMP enabled | AMP enabled |
| Hardware | Google Colab (T4) | Google Colab (T4) | Google Colab (T4) |
| Random seed | 42 | 42 | 42 |
| Data split | 70 / 15 / 15 (stratified) | 70 / 15 / 15 (stratified) | 70 / 15 / 15 (stratified) |

All models were trained from scratch on the Severstal steel defect dataset under identical data split conditions to ensure comparability across augmentation conditions (Raw, Copy-Paste, CASDA).

---

## Addition 6: Data Leakage Clarification

**Insert:** In the Data Preparation subsection (Section 3 or Section 4.0), as a dedicated paragraph after the description of the train/val/test split procedure.

---

To prevent data leakage, the following strict data isolation protocol was applied throughout the pipeline. ROI extraction for background-defect synthesis was performed exclusively on images belonging to the training split; no validation or test images were used as a source for ROI collection. Background patches used in the Poisson blending stage were sampled solely from training-split images. The train/validation/test split ratio (70/15/15) was determined prior to any augmentation step; the test split was held out completely until final evaluation and was not consulted for split ratio selection, threshold tuning, or suitability score calibration. This ensures that all reported evaluation metrics reflect performance on held-out data that was strictly unseen during both training and augmentation design.
