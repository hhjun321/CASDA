"""
figure_stageA_subtypes.py — Defect Subtype Categories
=============================================================================
§3.2.1 Geometric ROI Characterization — defect subtype classification

4 panels (one per defect subtype), each showing:
  Row 1: representative 256×256 ROI patch (original, no overlay)
  Row 2: description box (morphological classification criteria)

Subtypes: linear_scratch / irregular / compact_blob / general

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_stageA_subtypes.py
"""

import ast
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
ROI_META     = DRIVE_DATA / "roi_patches_v5.1/roi_metadata.csv"
ROI_IMGS_DIR = DRIVE_DATA / "roi_patches_v5.1/images"

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "stageA_subtypes.jpg"

IMG_H, IMG_W = 256, 1600

# ─────────────────────────────────────────────────────────────────────────────
# Defect subtype definitions
# ─────────────────────────────────────────────────────────────────────────────
SUBTYPES = ['linear_scratch', 'irregular', 'compact_blob', 'general']

SUBTYPE_META = {
    'linear_scratch': dict(
        label='Linear Scratch', abbr='LIN',
        color='#1565C0', light='#BBDEFB',
        desc='elong ≥ θ_e\nHigh elongation ratio\nNarrow, elongated shape',
    ),
    'irregular': dict(
        label='Irregular', abbr='IRR',
        color='#C62828', light='#FFCDD2',
        desc='solid < θ_s\nLow solidity\nFragmented or branching',
    ),
    'compact_blob': dict(
        label='Compact Blob', abbr='BLB',
        color='#2E7D32', light='#C8E6C9',
        desc='solid ≥ θ_s  &  elong < θ_e\nHigh solidity + low elongation\nCircular / square shape',
    ),
    'general': dict(
        label='General', abbr='GEN',
        color='#6A1B9A', light='#E1BEE7',
        desc='No specific condition met\nMixed / ambiguous morphology\nCatch-all category',
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def safe_bbox(val):
    try:
        return [int(v) for v in ast.literal_eval(str(val))]
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Load metadata & select one representative per subtype
# ─────────────────────────────────────────────────────────────────────────────
print("Loading ROI metadata ...")
df_meta = pd.read_csv(ROI_META, sep=None, engine='python')
df_meta.columns = df_meta.columns.str.strip().str.lower().str.replace(' ', '_')
df_meta['roi_bbox'] = df_meta['roi_bbox'].apply(safe_bbox)
df_meta = df_meta.dropna(subset=['roi_bbox'])
print(f"  {len(df_meta)} ROIs loaded")

reps = {st: None for st in SUBTYPES}

for st in SUBTYPES:
    sub = (df_meta[df_meta['defect_subtype'] == st]
           .sort_values('suitability_score', ascending=False))
    for _, row in sub.iterrows():
        image_id  = str(row['image_id']).strip()
        class_id  = int(row.get('class_id', 0))
        region_id = int(row.get('region_id', 0))

        # prefer pre-extracted ROI patch
        roi_fname = f"{image_id}_class{class_id}_region{region_id}.png"
        roi_fpath = ROI_IMGS_DIR / roi_fname
        if roi_fpath.exists():
            reps[st] = row
            print(f"  [{st:16s}] {image_id}  class={class_id}  "
                  f"score={row['suitability_score']:.3f}")
            break
        # fallback: crop from full image
        if (TRAIN_IMAGES / image_id).exists():
            reps[st] = row
            print(f"  [{st:16s}] {image_id} (crop fallback)  "
                  f"score={row['suitability_score']:.3f}")
            break
    if reps[st] is None:
        print(f"  [{st:16s}] no example found")

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7), facecolor='white')

gs = gridspec.GridSpec(
    2, 4,
    height_ratios=[4.0, 1.2],
    wspace=0.055, hspace=0.06,
    top=0.90, bottom=0.04, left=0.02, right=0.99,
)

for j, st in enumerate(SUBTYPES):
    meta = SUBTYPE_META[st]
    row  = reps[st]

    # ── Load ROI patch ─────────────────────────────────────────────
    patch_sq = np.full((256, 256), 128, dtype=np.uint8)
    src = ''

    if row is not None:
        image_id  = str(row['image_id']).strip()
        class_id  = int(row.get('class_id', 0))
        region_id = int(row.get('region_id', 0))
        src       = image_id[:16]

        roi_fname = f"{image_id}_class{class_id}_region{region_id}.png"
        roi_fpath = ROI_IMGS_DIR / roi_fname
        if roi_fpath.exists():
            patch_sq = cv2.resize(
                np.array(Image.open(roi_fpath).convert('L')),
                (256, 256), interpolation=cv2.INTER_AREA,
            )
        elif (TRAIN_IMAGES / image_id).exists():
            full_img = cv2.imread(str(TRAIN_IMAGES / image_id), cv2.IMREAD_GRAYSCALE)
            x1, y1, x2, y2 = row['roi_bbox']
            crop = full_img[y1:y2, x1:x2]
            if crop.size > 0:
                patch_sq = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)

    # ── Image panel ────────────────────────────────────────────────
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    patch_disp = clahe.apply(patch_sq)

    ax_img = fig.add_subplot(gs[0, j])
    ax_img.imshow(patch_disp, cmap='gray', vmin=0, vmax=255, aspect='equal')
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_linewidth(4.0); sp.set_edgecolor(meta['color'])

    ax_img.set_title(f"({meta['abbr']})  {meta['label']}",
                     fontsize=14, fontweight='bold', color=meta['color'], pad=7)

    ax_img.text(0.5, -0.04, src, ha='center', va='top',
                fontsize=8.5, color='#9E9E9E', style='italic',
                transform=ax_img.transAxes)

    # ── Description panel ──────────────────────────────────────────
    ax_desc = fig.add_subplot(gs[1, j])
    ax_desc.axis('off')
    ax_desc.set_facecolor(meta['light'])
    for sp in ax_desc.spines.values():
        sp.set_linewidth(2.5); sp.set_edgecolor(meta['color'])
    ax_desc.patch.set_alpha(0.88)
    ax_desc.text(0.5, 0.5, meta['desc'],
                 ha='center', va='center', fontsize=10,
                 color='#212121', transform=ax_desc.transAxes,
                 multialignment='center', fontfamily='monospace')

# ── Save ─────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none',
            format='jpeg', pil_kwargs={'quality': 95, 'subsampling': 0})
print(f'\nSaved → {OUT_FILE}')
plt.show()
print('Done.')
