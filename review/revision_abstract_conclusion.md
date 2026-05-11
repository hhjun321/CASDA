# Abstract and Conclusion — Revised Text for Paper Revision

> **Instructions for the author:** Replace the existing Abstract and Conclusion with the revised versions below. Bracketed references (e.g., [Table N]) must be renumbered to match the final manuscript before submission. The abstract word count target is approximately 200 words; adjust as needed to meet any journal word limit without removing factual claims.

---

## Revised Abstract

Steel surface defect detection in industrial inspection suffers from two compounding challenges: chronic data scarcity and severe class imbalance among defect categories. We propose CASDA (Compatibility-Aware Synthesis for Defect Augmentation), a ControlNet-based data augmentation framework that generates realistic training images by compositing annotated defect regions onto defect-free steel surface patches selected through a defect-background compatibility matrix and a quantitative suitability score. Unlike naive copy-paste augmentation, CASDA harmonizes the composite image via Poisson blending and conditions the ControlNet generator on structured text prompts and a 3-channel spatial hint encoding defect geometry, surface grain orientation, and texture roughness.

Evaluated on the Severstal steel defect dataset across three benchmark models, CASDA yielded clear detection gains for YOLO-MFD: +2.89 pp mAP over the Raw baseline and +5.78 pp over the Copy-Paste baseline, with the most pronounced improvement on the most severely under-represented class (Class 2, +7.92 pp vs Raw; +15.06 pp vs Copy-Paste). For EB-YOLOv8, the overall gain was marginal (+0.31 pp vs Raw), with a notable Class 4 improvement (+3.45 pp), reflecting model-architecture-dependent augmentation sensitivity. For DeepLabV3+ semantic segmentation, CASDA produced a slight decrease in mean Dice (−0.58 pp vs Raw), attributable to Poisson blending boundary artifacts, an acknowledged limitation for pixel-level prediction tasks. FID analysis confirms that CASDA-generated images are distributionally closer to real defect images than Copy-Paste composites. These results establish CASDA as an effective augmentation strategy for detection-oriented architectures, with identified limitations for segmentation tasks.

---

## Revised Conclusion

This paper presented CASDA, a compatibility-aware synthesis framework for steel defect detection that addresses class imbalance by leveraging defect-free images as synthesis substrates within a ControlNet-based generation pipeline. The core contributions are: a background-defect compatibility matrix that guides synthesis toward physically plausible pairings, a suitability-scored ROI selection mechanism, and a structured prompt and 3-channel hint conditioning scheme for ControlNet.

The experimental results on the Severstal dataset demonstrate clear detection improvements for YOLO-MFD (+2.89 pp mAP vs Raw; +5.78 pp vs Copy-Paste), with consistent minority-class AP gains across all evaluated models, most notably for Class 2. A substantive finding beyond raw performance numbers is the evidence of model-architecture-dependent augmentation sensitivity: YOLO-MFD benefited substantially from CASDA while EB-YOLOv8 exhibited only marginal gains, suggesting that backbone representational capacity modulates augmentation effectiveness in ways that warrant further investigation.

Three directions for future work are identified as high priority. First, statistical validation through repeated trials with significance testing is necessary to establish the reliability of small observed differences. Second, a boundary-aware blending strategy (e.g., learned alpha matting or diffusion-based inpainting) is required to extend CASDA's benefits to pixel-level segmentation tasks. Third, evaluation against a broader set of augmentation baselines and on additional industrial inspection datasets is needed to characterize the scope of CASDA's generalizability.
