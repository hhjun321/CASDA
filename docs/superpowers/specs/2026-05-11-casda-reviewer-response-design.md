# CASDA Reviewer Response Design
**Date:** 2026-05-11  
**Approach:** B — Targeted improvements (no new experiments)  
**Output:** Rebuttal letter (3 reports) + Paper revision plan

---

## Context

- Journal submission (not conference)
- 3 reviewer reports: report1 (5 comments), report2 (11 comments), report3 (23 comments)
- Paper: CASDA — ControlNet + Poisson Blending augmentation for steel defect detection
- Models: YOLO-MFD, EB-YOLOv8, DeepLabV3+
- Experimental results fully available in Colab

---

## Common Issues Matrix

| Issue | R1 | R2 | R3 | Priority |
|-------|:--:|:--:|:--:|:--------:|
| No figures | ✓ | | | HIGH |
| Missing implementation details | ✓ | ✓ | ✓ | HIGH |
| EB-YOLOv8 results overstated | ✓ | ✓ | ✓ | HIGH |
| DeepLabV3+ Dice decreases | | ✓ | ✓ | HIGH |
| Threshold justification missing | | ✓ | ✓ | HIGH |
| No statistical significance | | ✓ | ✓ | MEDIUM |
| Ablation incomplete | | ✓ | | MEDIUM |
| Novelty unclear | | ✓ | | MEDIUM |
| Literature / baseline insufficient | ✓ | ✓ | ✓ | MEDIUM |
| Data leakage concern | | ✓ | | MEDIUM |
| Realism metrics beyond heuristic | | ✓ | ✓ | MEDIUM |
| Language / grammar | ✓ | ✓ | ✓ | LOW (deferred) |

---

## Section 1: Strategic Positioning

### Novelty Defense (vs R2 "combination of existing techniques")

Two angles:

**1. Domain-specific integration novelty**  
The core novelty is not any single component but the explicit modeling of defect-background relationship via compatibility matrix + 3-channel hint (R=defect geometry, G=background structure, B=background texture). No prior work encodes this tripartite relationship for conditional defect synthesis.

**2. Clean image data reuse**  
CASDA synthesizes defects onto defect-free images in defect-possible ROI regions — converting previously unusable clean images into augmentation resources. Copy-paste only moves defects between existing defect images. CASDA effectively expands the usable data pool, addressing data sparsity more fundamentally.

### EB-YOLOv8 Strategy

Key numbers:
- CASDA vs Raw: +0.31 pp (positive but marginal)
- CASDA vs Copy-Paste: -0.14 pp (slight underperformance)
- Class 4 vs Raw: **+3.45 pp** (significant minority class gain)
- Class 2 vs Copy-Paste: **+1.99 pp**

Defense angles:
1. -0.14 pp difference is not statistically conclusive (single run, no CI)
2. EB-YOLOv8 enhanced backbone already provides strong feature extraction → lower marginal gain from data augmentation is architecturally expected
3. Per-class analysis shows consistent minority class improvement
4. Reframe in Discussion as "model-augmentation interaction" — contribution to the field as a finding

### DeepLabV3+ Strategy

Key numbers:
- CASDA Dice Mean: 0.6232 vs Raw: 0.6290 → -0.58 pp
- CASDA vs Copy-Paste: -0.15 pp

Defense: **Convert to limitation → future work**  
Poisson blending boundary artifacts are more damaging to pixel-level segmentation than bounding-box detection. This is a valid finding: CASDA is optimized for detection tasks. Segmentation requires boundary-aware blending (e.g., diffusion inpainting). Explicitly framed as a limitation and future direction.

---

## Section 2: Figures to Add

| ID | Figure | Paper Location | Data Source | Addresses |
|----|--------|---------------|-------------|-----------|
| F1 | CASDA pipeline overview (4-stage flowchart) | Method intro | Diagram (new) | R1-1, R2-2 |
| F2 | 3-channel hint image: ROI → R/G/B → generated output | Stage 2 | ControlNet dataset images | R1-1, R2-2 |
| F3 | Visual comparison: Raw / Copy-Paste / CASDA side-by-side | Experiments | augmented_images | R1-1 |
| F4 | Defect type classification examples (4 types) | Stage 1 | roi_patches | R1-1, R2 |
| F5 | Per-class AP bar chart (3 models × 4 methods) | Results | Table 10/11/12 | R2-4, R3-14 |
| F6 | Ablation: w/ vs w/o Blending visual comparison | Ablation | Table 13 + images | R2-7 |

All source images available in Colab. F5, F6 can be generated from existing numerical results (matplotlib). F2, F3, F4 require actual pipeline output images from Drive.

---

## Section 3: Implementation Details to Add

Items missing from paper, to be extracted from code/configs in implementation phase:

| Item | Source | Paper Addition |
|------|--------|---------------|
| Compatibility matrix construction | Stage A scripts | Algorithm description + matrix table |
| Background category definitions | Stage A scripts | Taxonomy table |
| Prompt templates | Stage A/B scripts | Example prompts table |
| ControlNet settings (lr, epochs, batch, guidance scale, sampling steps) | configs/benchmark_experiment.yaml | Table in Method |
| Benchmark model training settings (optimizer, lr, epochs, batch, hardware) | configs/benchmark_experiment.yaml | Table in Experiments |

---

## Section 4: Threshold Justification

| Parameter | Value | Defense Method |
|-----------|-------|---------------|
| Linearity > 0.85 | linear_scratch classification | Show distribution histogram → natural breakpoint |
| Aspect ratio > 5.0 | linear judgment | Same histogram approach |
| Suitability weights (0.5/0.3/0.2) | color/artifact/sharpness | Domain-informed rationale + sensitivity analysis note |
| Quality gate Q ≥ 0.7 | pruning threshold | Table 7 (61.3% high quality) + threshold-vs-datasize tradeoff table |

Rename "heuristic" → "domain-informed empirical threshold" throughout paper.

---

## Section 5: Statistical Significance

**Strategy:** Acknowledge as limitation, defend where magnitude is sufficient.

- Large effects (Blending ablation: -11.38 pp, Class 2 YOLO-MFD: +7.92 pp) → magnitude alone argues against noise
- Small effects (EB-YOLOv8 ±0.14 pp, DeepLabV3+ ±0.15 pp) → explicitly state "not statistically conclusive; single-run experiment"
- Add to Limitations section: "Future work should include repeated runs with multiple random seeds and formal significance testing"

---

## Section 6: Ablation Completeness (R2-7)

**What exists:** Pruning ablation (-0.09 pp) + Blending ablation (-11.38 pp)

**Defense of current ablation:**  
Blending ablation (-11.38 pp, Class 2 -23.59 pp) strongly demonstrates that component's contribution. Pruning ablation shows modest effect — honestly reported.

**What's missing:** Compatibility matrix, 3-channel hint, prompt, quality gate independent tests.

**Response:** Acknowledge as a limitation. Blending effect is the largest contributor and is demonstrated. Full component-wise ablation is listed as future work. If random-background experiment results are available in Colab → include as additional ablation.

---

## Section 7: Literature and Baselines (R1-4, R2-5, R3-2, R3-22)

**Literature additions:**
- Recent defect detection + data augmentation papers (2023–2025)
- Diffusion-based industrial augmentation methods
- GAN-based defect synthesis works
- Data-driven visual inspection frameworks

**Baseline additions:**
- Existing: Raw / Traditional / Copy-Paste
- Ideal addition: GAN-based or diffusion-based augmentation baseline
- If no additional experiments: commit to future comparison study in rebuttal

**FID results:** Already computed (fid_results path in Colab) → add explicitly to paper Results section. Addresses R3-5 "FID beyond mAP".

---

## Section 8: Paper Revision Priority Map

| Priority | Task | Section | New Experiment |
|----------|------|---------|---------------|
| 1 | Add F1–F6 figures | Method + Results | No |
| 2 | Add methodology details (compatibility matrix, prompts, ControlNet settings) | Section 3 | No |
| 3 | Reframe EB-YOLOv8 + DeepLabV3+ discussion | Results + Discussion | No |
| 4 | Add Limitations + Future Work subsection | Discussion | No |
| 5 | Separate detection vs segmentation analysis | Results | No |
| 6 | Add FID results explicitly | Results | No |
| 7 | Add threshold justification + distribution figures | Method | No |
| 8 | Expand literature review | Introduction | No |
| 9 | Rewrite Abstract + Conclusion | Abstract/Conclusion | No |
| 10 | Fix table numbering, remove informal expressions | All | No |

---

## Section 9: Rebuttal Document Structure

One response document per reviewer report.

**Format per comment:**
```
[Comment N]
> [Reviewer's original comment — quoted]

Response: [Acknowledgment] + [Defense or concession] + [Paper change made]
```

**Tone:** Respectful, direct, specific. Cite exact table/section numbers for all paper changes.

### Report 1 (5 comments)
| Comment | Type | Strategy |
|---------|------|---------|
| 1 — No figures | Concede | Add F1–F4, describe each |
| 2 — Implementation details | Concede | List all additions |
| 3 — EB-YOLOv8 overstated | Concede + reframe | model-dependent behavior, per-class data |
| 4 — Limited literature | Concede | List papers to add |
| 5 — Language | Concede | Acknowledge, professional revision planned |

### Report 2 (11 comments)
| Comment | Type | Strategy |
|---------|------|---------|
| 1 — Limited novelty | Defend | Clean-image reuse + context-aware integration |
| 2 — Reproducibility | Concede | All missing details to be added |
| 3 — Heuristic thresholds | Defend + add | Distribution-based justification + sensitivity note |
| 4 — Inconsistent results | Concede + reframe | model-augmentation interaction, per-class view |
| 5 — Insufficient baselines | Partial concede | FID added; additional baselines → future work |
| 6 — No significance tests | Concede | Magnitude defense + limitation acknowledgment |
| 7 — Incomplete ablation | Partial concede | Blending effect is large; full ablation → future work |
| 8 — Data leakage | Clarify | Explain pipeline isolation (ROI from train only) |
| 9 — Realism validation | Partial concede | FID already computed → add to paper |
| 10 — Detection vs segmentation | Concede | Add separate analysis subsection |
| 11 — Writing/presentation | Concede | Fix table numbering, remove informal language |

### Report 3 (23 comments)
| Comment | Type | Strategy |
|---------|------|---------|
| 1 — Language | Concede | Deferred to language revision phase |
| 2 — Literature | Concede | Add recent augmentation papers |
| 3 — ROI sensitivity | Address | Suitability score filters poor ROIs; Q threshold |
| 4 — Weighting justification | Address | Domain rationale + sensitivity note |
| 5 — FID/LPIPS comparison | Address | FID results exist → add to paper |
| 6 — Domain gap | Address | Single-dataset limitation; acknowledged in future work |
| 7 — DeepLabV3+ degradation | Address | Boundary artifact explanation → limitation |
| 8 — Scalability | Address | Framework is generalizable; future multi-dataset study |
| 9 — Computational cost | Address | Add comparison table (inference time) if available |
| 10 — Label noise | Address | Quality gate Q≥0.7 filters unrealistic samples |
| 11 — Morphological index limits | Address | Acknowledge; general class captures edge cases |
| 12 — 42.7% ratio / overfitting | Address | casda_ratio experiment groups address this directly |
| 13 — Reproducibility | Address | Thresholds will be fully documented in revision |
| 14 — EB-YOLOv8 marginal | Address | model-augmentation interaction discussion |
| 15 — Q≥0.7 diversity tradeoff | Address | Threshold-vs-datasize tradeoff table |
| 16 — Alternative blending | Address | Future work: GAN/diffusion inpainting |
| 17 — Extreme imbalance | Address | Class 2 scenario (110.1% increase) demonstrates this |
| 18 — Statistical significance | Address | Magnitude defense + limitation |
| 19 — Overlapping defects | Address | Limitation; future work |
| 20 — 3D extension | Address | Out of scope; future work |
| 21 — Grayscale dataset limits | Address | Limitation; generalization study → future work |
| 22 — Discussion vs published | Concede | Add comparison with published results |
| 23 — Rewrite abstract/conclusion | Concede | Rewrite as part of revision |

---

## Implementation Phase Tasks

1. Extract implementation details from code/configs (Colab scripts + benchmark_experiment.yaml)
2. Generate F5, F6 from existing numerical data (matplotlib code)
3. Write Figure captions and descriptions for F1–F4 (user provides actual images)
4. Draft rebuttal letter: Report 1 → Report 2 → Report 3
5. Draft paper revision sections: Results discussion rewrite, Limitation section, Method additions
6. Update Abstract + Conclusion
