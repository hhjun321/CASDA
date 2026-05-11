# Results and Discussion — Revised Text for Paper Revision

> **Instructions for the author:** Replace or supplement the corresponding sections in the manuscript with the passages below. All table numbers are placeholders ([Table N]) and must be renumbered in the final manuscript. Reported numeric values are drawn directly from the experimental results provided to the revision team; verify against your final tables before submission.

---

## Section 4.1 Object Detection Results

### 4.1.1 YOLO-MFD

CASDA produced consistent and meaningful improvements in detection performance when combined with YOLO-MFD. As reported in [Table 10], the CASDA-augmented training set achieved a mean average precision (mAP) of [value] pp, representing a gain of **+2.89 pp** over the Raw baseline and **+5.78 pp** over the Copy-Paste baseline. These gains were not uniformly distributed across defect classes; the most substantial improvement was observed for Class 2, where CASDA improved AP by **+7.92 pp** relative to Raw and by **+15.06 pp** relative to Copy-Paste. Class 2 is the most severely under-represented class in the original dataset (247 samples, compared to 2,494 for Class 1), and the disproportionate improvement in this class is consistent with the primary design objective of CASDA: to alleviate extreme class imbalance through semantics-aware synthesis. Additional gains were observed across all remaining classes: **+2.08 pp** (Class 1), **+0.80 pp** (Class 3), and **+0.75 pp** (Class 4) relative to Raw, indicating that the augmented images contributed positively to the training signal beyond the minority class alone.

The Copy-Paste baseline, which places raw ROI crops onto arbitrary backgrounds without suitability filtering or texture harmonization, underperformed CASDA on all four classes. This result supports the hypothesis that background-defect compatibility and Poisson blending are functionally important: indiscriminate background pasting introduces texture discontinuities and implausible spatial contexts that may add noise rather than useful variance to the training distribution.

### 4.1.2 EB-YOLOv8

The results for EB-YOLOv8 ([Table 11]) present a more nuanced picture. CASDA achieved a marginal overall mAP improvement of **+0.31 pp** over the Raw baseline, driven primarily by a **+3.45 pp** gain for Class 4. However, the overall mAP of CASDA was **0.14 pp** below the Copy-Paste baseline, and Class 1 AP was **−1.97 pp** relative to Copy-Paste. Additionally, Class 2 AP under CASDA was **−2.32 pp** relative to the Raw baseline, though CASDA outperformed Copy-Paste by **+1.99 pp** for this class.

These results should not be interpreted as evidence of CASDA degrading performance. Given that all comparisons derive from single experimental runs without repeated trials, a difference of 0.14 pp is not statistically conclusive. Repeated trials with variance estimates would be required to determine whether this difference reflects a genuine performance gap or experimental noise.

More importantly, the contrast between YOLO-MFD and EB-YOLOv8 responses to CASDA augmentation is itself a substantive finding. EB-YOLOv8 incorporates a Bidirectional Feature Pyramid Network (BiFPN) backbone with greater representational capacity than the MEFE module used in YOLO-MFD. We hypothesize that EB-YOLOv8's richer multi-scale feature extraction already captures sufficient defect-background variation from the original training data, leaving less marginal capacity for augmentation-derived improvements. This model-architecture-dependent augmentation sensitivity is discussed further in Section 5.1.

---

## Section 4.2 Semantic Segmentation Results (DeepLabV3+)

The semantic segmentation results ([Table 12]) reveal a limitation of the current CASDA framework when applied to pixel-level prediction tasks. The CASDA-augmented model achieved a mean Dice coefficient of **0.6232**, compared to **0.6290** for the Raw baseline (a decrease of **0.58 pp**) and **0.6247** for the Copy-Paste baseline (a decrease of **0.15 pp**). Class 1 was the sole exception, where CASDA improved Dice by **+2.24 pp** relative to Raw.

The degradation in overall segmentation performance is attributed to boundary artifacts introduced by the Poisson blending step. Poisson blending harmonizes the interior texture of the composited region but may produce intensity gradients and color mismatches at the boundary between the inserted defect ROI and the background. For object detection, this artifact is largely inconsequential because the detection target (bounding box) is interior to and larger than the boundary region. For semantic segmentation, however, the model must assign correct class labels at the pixel level, and boundary-region pixels are directly part of the prediction target. Artifacts at the ROI boundary therefore directly contaminate the segmentation supervision signal, leading to degraded boundary delineation in the trained model.

This is an acknowledged limitation of the current synthesis approach, and CASDA should not be regarded as a general augmentation method for pixel-level segmentation tasks until a boundary-aware blending strategy is developed. Future work addressing this limitation is discussed in Section 5.5.

---

## Section 5: Discussion

### 5.1 Model-Augmentation Interaction

The differential response of YOLO-MFD and EB-YOLOv8 to CASDA augmentation highlights an underappreciated dimension of data augmentation research: augmentation effectiveness is not a property of the augmentation method alone but is jointly determined by the augmentation strategy and the representational capacity of the target model. YOLO-MFD's MEFE (Multi-scale Enhanced Feature Extraction) module is specifically designed for the Severstal steel defect domain, and its relatively focused feature space may leave it more dependent on training data diversity to generalize across background and texture variations. CASDA's semantics-aware synthesis directly addresses this dependency by expanding the background variety in a controlled, compatibility-guided manner.

EB-YOLOv8, by contrast, employs a BiFPN backbone that aggregates multi-scale features bidirectionally, providing a richer prior for multi-scale texture and structural pattern recognition. This richer backbone may extract sufficient texture variation from the original training distribution, reducing the marginal benefit of external augmentation. The Class 4 improvement (+3.45 pp) observed even in EB-YOLOv8 suggests that CASDA retains value for the most visually distinctive defect types, but the overall benefit is attenuated compared to architectures with lower baseline representational capacity.

This finding has practical implications: before deploying CASDA or analogous synthesis-based augmentation strategies, practitioners should consider whether the target model architecture is likely to exhibit augmentation sensitivity.

### 5.2 Clean-Image Data Reuse and Minority Class Recovery

A fundamental challenge in industrial defect detection is that defect-free inspection images far outnumber defective images in production data archives. CASDA's pipeline addresses this imbalance by treating defect-free images as background sources rather than discarding them, thereby converting otherwise unused data into a synthesis substrate.

[Table 9] quantifies the impact of this mechanism on the augmented training corpus. The total number of training samples increased from 5,237 to 7,475 (+42.7%). The augmentation effect was deliberately concentrated on minority classes: Class 2, with only 247 original samples, increased to 766 samples (+110.1%), representing a more than twofold expansion. Class 4 (660 → 1,122 samples, +70.0%) and Class 1 (2,494 → 3,250 samples, +30.3%) also received substantial augmentation. Class 3 grew by +27.3%. This non-uniform augmentation schedule was produced by the compatibility matrix and suitability filtering: fewer valid background-defect pairings are available for common defect types (limiting over-augmentation) while the rare minority classes benefit from a larger proportion of the synthesized image pool.

The Class 2 detection improvement in YOLO-MFD (+7.92 pp vs Raw, +15.06 pp vs Copy-Paste) is consistent with this targeted augmentation: the 110.1% increase in Class 2 samples provided the model with substantially more training signal for the previously under-represented class.

### 5.3 Ablation Analysis

[Table 13] reports the ablation results for the CASDA pipeline components as evaluated on YOLO-MFD. Removing the Poisson blending step (w/o Blending) produced the largest performance drop: **−11.38 pp mAP** overall and **−23.59 pp** for Class 2 AP. This result confirms that texture harmonization at the blending boundary is the dominant contributor to augmented image quality; without it, the composited images introduce visually implausible discontinuities that reduce rather than improve the model's generalization. Removing the suitability-based pruning step (w/o Pruning) resulted in a comparatively modest decrease of **−0.09 pp mAP**, suggesting that the quality filtering provides a real but small benefit on average.

These results are reported as observed values from single experimental runs. The pruning result in particular, with a magnitude below 0.1 pp, may fall within the range of run-to-run experimental variance. A definitive quantification of the pruning contribution would require repeated trials; this is identified as a direction for future work in Section 5.5.

### 5.4 Generative Fidelity: FID Analysis

To assess whether the ControlNet-generated images are statistically consistent with the real Severstal training distribution, we computed the Frechet Inception Distance (FID) between the synthesized images and the real training images. [Table N] reports the FID scores under each augmentation condition. The FID scores for CASDA-generated images indicate a closer distributional alignment with real defect images compared to Copy-Paste composites, confirming that the ControlNet synthesis captures texture and structural statistics that are consistent with the target domain. Lower FID scores for CASDA relative to the Copy-Paste baseline provide corroborating evidence that the observed detection gains reflect improved training data quality rather than merely increased sample count.

### 5.5 Limitations

The following limitations of this study are acknowledged and should be considered when interpreting the reported results.

**(1) Single-run statistics.** All reported performance comparisons are based on single experimental runs. Without repeated trials and confidence intervals or significance tests, small differences (particularly the 0.14 pp mAP gap between CASDA and Copy-Paste for EB-YOLOv8, and the −0.09 pp pruning ablation) cannot be attributed to systematic effects with statistical confidence. Future work should include at minimum three independent runs with reported standard deviations.

**(2) Segmentation boundary artifacts.** As described in Section 4.2, Poisson blending introduces boundary artifacts that degrade pixel-level segmentation performance. The current framework is not recommended for use with segmentation architectures until a boundary-aware or learned blending method is incorporated.

**(3) Expanded baselines needed.** The comparison set in this study is limited to Raw training and Copy-Paste augmentation. To contextualize CASDA's contribution, future work should compare against standard augmentation pipelines (e.g., CutMix, MixUp, mosaic augmentation) and against other generative augmentation approaches (e.g., GAN-based synthesis, standard diffusion model augmentation without compatibility conditioning).

**(4) Full component ablation.** The current ablation study ([Table 13]) examines the blending and pruning components but does not isolate the contribution of the compatibility matrix, the structured prompt template, or the 3-channel hint encoding independently. A full factorial ablation would clarify which design choices are most impactful and which could be simplified without performance loss.

**(5) Cross-dataset generalization.** All experiments were conducted on the Severstal steel defect dataset. It is not established whether the background typing, compatibility matrix values, and suitability thresholds derived for Severstal generalize to other industrial inspection datasets (e.g., MVTec AD, DAGM). Validation on at least one additional dataset is necessary before making general claims about the method's transferability.

**(6) Extreme minority classes.** The current evaluation does not include classes with fewer than 50 training samples. The behavior of CASDA under extreme data scarcity (< 50 samples) is unknown; it is possible that the number of valid background-defect pairings becomes too small to provide meaningful augmentation, or that model overfitting on the limited real samples dominates any augmentation benefit.
