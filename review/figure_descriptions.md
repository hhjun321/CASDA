# Figure Descriptions for CASDA Paper (F1–F4)

Figures F1–F4 require either a diagram tool or actual image patches from the
Severstal dataset. Each entry below specifies:
- insertion point in the paper,
- a ready-to-use caption (academic English, ≤ 3 sentences), and
- the exact content and Colab source paths needed to produce the figure.

---

## F1 — CASDA Four-Stage Pipeline Flowchart

**Paper location:** Before Section 3.1 (introduce as Figure 1 at the start of
the Methodology section).

**Caption:**
> **Figure 1. Overview of the CASDA data-augmentation pipeline.**
> The pipeline consists of four sequential stages executed on complementary
> hardware: Stage A (ROI extraction and 3-channel hint image preparation, CPU)
> feeds into Stage B (ControlNet training and conditional image generation,
> GPU), whose outputs are refined in Stage C (Poisson blending and quality
> pruning via FID-based scoring, CPU), and finally evaluated in Stage D
> (FID measurement and downstream benchmark training, GPU).
> Dashed arrows indicate optional feedback paths used during hyper-parameter
> tuning.

**Content to show:**
Four rectangular boxes arranged left-to-right (or top-to-bottom) connected
by solid arrows, with hardware labels (CPU / GPU) shown as small badges:

| Box | Label | Key operations |
|-----|-------|----------------|
| A | Stage A — ROI Extraction & Hint Prep (CPU) | Crop defect ROIs, classify defect type, build 3-channel hint (R: mask, G: background structure, B: background texture) |
| B | Stage B — ControlNet Training & Generation (GPU) | Fine-tune ControlNet on hint+ROI pairs, generate synthetic defect images |
| C | Stage C — Poisson Blending & Pruning (CPU) | Blend generated ROIs into background via Poisson blending, compute FID per image, discard low-quality samples |
| D | Stage D — FID Evaluation & Benchmark (GPU) | Compute dataset-level FID, train YOLO-MFD / EB-YOLOv8 / DeepLabV3+ on augmented sets, report AP / Dice |

**No Colab source images required** — this figure can be drawn with any
diagram tool (draw.io, PowerPoint, or matplotlib patches). If using
matplotlib, place boxes with `FancyBboxPatch` and connect with `FancyArrowPatch`.

---

## F2 — 3-Channel Hint Image Construction

**Paper location:** Section 3.2 (Hint Image Construction), after the paragraph
describing the three channels.

**Caption:**
> **Figure 2. Construction of the 3-channel hint image used to condition
> ControlNet generation.**
> Starting from a cropped ROI containing a steel-surface defect, the defect
> mask is encoded as the R channel, the low-frequency background structure
> (Gaussian-blurred complement) as the G channel, and the high-frequency
> background texture as the B channel.
> The assembled hint image is passed to ControlNet alongside a text prompt,
> and the generated output is composited back into the source frame via
> Poisson blending to yield the final augmented sample.

**Content to show (left → right, 6 panels):**

| Panel | Label | Description |
|-------|-------|-------------|
| 1 | Source ROI | Original cropped region from the Severstal image; defect visible, with mask outline overlaid in red |
| 2 | R: Defect Mask | Binary mask of the defect region (white = defect, black = background) |
| 3 | G: Background Structure | Gaussian-blurred complement of the defect mask applied to the ROI background — captures low-frequency surface pattern |
| 4 | B: Background Texture | High-pass residual of the background (original minus blurred) — captures fine surface texture |
| 5 | ControlNet Output | ControlNet-generated defect image conditioned on the hint |
| 6 | Poisson Blended Result | Final augmented frame after Poisson-blending the generated ROI into a new background location |

**Colab source paths:**

```
ROI patches and masks:
  /content/drive/MyDrive/data/Severstal/roi_patches/
  (subdirectories per defect class; each patch has a paired *_mask.png)

ControlNet dataset (hint images + paired ROIs):
  /content/drive/MyDrive/data/Severstal/controlnet_dataset/

ControlNet generated outputs:
  /content/drive/MyDrive/data/Severstal/augmented_images/

Final Poisson-blended augmented samples:
  /content/drive/MyDrive/data/Severstal/augmented_dataset/casda_composed/
```

**Assembly note:** Select a single representative defect instance (e.g., one
Class 2 linear scratch patch) and show all six panels at the same scale.
Label each panel with a white-on-black header using `ax.set_title()`. Use
`plt.subplots(1, 6, figsize=(18, 3))`.

---

## F3 — Visual Comparison: Raw vs. Copy-Paste vs. CASDA

**Paper location:** Section 4 (Experiments), immediately before Table 10
(YOLO-MFD per-class AP results).

**Caption:**
> **Figure 3. Qualitative comparison of augmentation strategies on the
> Severstal steel-surface dataset.**
> Each column shows the same background region with a defect introduced by
> different methods: (a) the original annotated training image (Raw), (b) a
> Copy-Paste baseline composited sample, and (c) a CASDA composited sample
> produced by ControlNet generation followed by Poisson blending.
> CASDA composites exhibit more realistic texture continuity at the
> defect–background boundary than Copy-Paste.

**Content to show (3 columns, same background crop):**

| Column | Label | Source |
|--------|-------|--------|
| (a) Raw | Original image with bounding-box or polygon annotation overlaid | `/content/drive/MyDrive/data/Severstal/train_images/` |
| (b) Copy-Paste | Copy-Paste augmented image; same defect type, same region | `/content/drive/MyDrive/data/Severstal/augmented_dataset/copypaste_baseline/` |
| (c) CASDA | CASDA composited image | `/content/drive/MyDrive/data/Severstal/augmented_dataset/casda_composed/` |

**Cropping instruction:** Identify a defect bounding box in the Raw image;
expand by ±50 px on each side; apply the same crop coordinates to all three
panels so the background region is identical.

**Assembly note:** Use `plt.subplots(1, 3, figsize=(12, 4))`. Load images
with `cv2.imread(...)[y0:y1, x0:x1]` (or PIL crop). Add column letters
(a)/(b)/(c) as panel titles at 10 pt. Save at dpi=150.

---

## F4 — Defect Type Classification Examples (4 Types)

**Paper location:** Section 3.1 (ROI Extraction and Defect Classification),
near Table 4 (defect type definitions and key-index thresholds).

**Caption:**
> **Figure 4. Representative ROI patches for each of the four defect type
> categories defined in CASDA.**
> Each patch is annotated with its assigned type label and the key
> morphological index values (elongation ratio, solidity, and relative area)
> used by the rule-based classifier described in Section 3.1.
> The four types — linear scratch, irregular, compact blob, and general —
> capture distinct surface defect morphologies observed in the Severstal
> dataset.

**Content to show (4 panels, one per defect type):**

| Panel | Type label | Morphological characteristics to highlight |
|-------|------------|-------------------------------------------|
| 1 | Linear Scratch | High elongation ratio (≥ threshold); narrow, elongated shape; annotate: `elong=<val>` |
| 2 | Irregular | Low solidity (< threshold); fragmented or branching shape; annotate: `solid=<val>` |
| 3 | Compact Blob | High solidity + low elongation; roughly circular/square shape; annotate: `solid=<val>, elong=<val>` |
| 4 | General | Does not meet any specific threshold; mixed or ambiguous morphology |

**Colab source path:**

```
/content/drive/MyDrive/data/Severstal/roi_patches/
  (select one representative patch per type; prefer patches where the
   morphological feature is visually unambiguous)
```

**Assembly note:** Use `plt.subplots(1, 4, figsize=(14, 3.5))`. For each
panel, overlay a text box in the upper-left corner with the type name and
index values using `ax.text(...)` with a semi-transparent white background
(`bbox=dict(facecolor='white', alpha=0.7)`). Ensure each patch is displayed
at the same pixel scale (resize to 128×128 or the median ROI size). Save
at dpi=150.
