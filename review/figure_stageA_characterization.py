"""
CASDA §3.2.1  Geometric ROI Characterization and Suitability Evaluation
========================================================================
Layout: 4 rows (defect subtypes) × 4 columns (pipeline steps)

  Col 1: Steel image crop (400 px wide) + ROI window + defect bbox
  Col 2: Binary defect mask (RLE decoded)
  Col 3: Geometric overlay on ROI patch — convex hull, major/minor axes,
          bounding box, metric annotations (elongation / solidity / rel_area)
  Col 4: Extracted 256×256 ROI patch + subtype badge + suitability score

Rows (one representative per subtype, highest suitability score):
  linear_scratch  /  irregular  /  compact_blob  /  general

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_stageA_characterization.py
"""

import ast
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image
from skimage import measure

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
TRAIN_CSV    = DRIVE_DATA / "train.csv"
ROI_META     = DRIVE_DATA / "roi_patches_v5.1/roi_metadata.csv"
ROI_IMGS_DIR = DRIVE_DATA / "roi_patches_v5.1/images"

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "stageA_characterization.jpg"

IMG_H, IMG_W = 256, 1600
CROP_HALF    = 200   # ROI 중심 좌우 확장 px (crop 폭 = 400 px)

# ──────────────────────────────────────────────────────────────────────────────
# Subtype metadata
# ──────────────────────────────────────────────────────────────────────────────
SUBTYPES = ['linear_scratch', 'irregular', 'compact_blob', 'general']

SUBTYPE_META = {
    'linear_scratch': {
        'label': 'Linear Scratch',
        'desc':  'High elongation ratio\nNarrow, elongated shape',
        'color': '#1565C0',
    },
    'irregular': {
        'label': 'Irregular',
        'desc':  'Low solidity\nFragmented or branching',
        'color': '#C62828',
    },
    'compact_blob': {
        'label': 'Compact Blob',
        'desc':  'High solidity + low elongation\nCircular / square shape',
        'color': '#2E7D32',
    },
    'general': {
        'label': 'General',
        'desc':  'Mixed / ambiguous\nmorphology',
        'color': '#6A1B9A',
    },
}

COL_TITLES = [
    'Steel Image Crop\n+ ROI & Defect Region',
    'Binary Defect Mask\n(RLE decoded)',
    'Geometric Analysis\n(Convex Hull + Axes)',
    'Extracted ROI Patch\n(256×256 px)',
]
COL_COLORS = ['#37474F', '#1565C0', '#2E7D32', '#E65100']

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def rle_to_mask(rle_str, h=IMG_H, w=IMG_W):
    flat = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle_str, str) and rle_str.strip():
        nums = list(map(int, rle_str.split()))
        for s, l in zip(nums[0::2], nums[1::2]):
            flat[s - 1: s - 1 + l] = 1
    return flat.reshape(w, h).T


def safe_bbox(val):
    try:
        return [int(v) for v in ast.literal_eval(str(val))]
    except Exception:
        return None


def compute_geom(mask_patch):
    """Return (elongation, solidity, rel_area, largest_region) for mask_patch."""
    labels = measure.label(mask_patch > 0)
    props  = measure.regionprops(labels)
    if not props:
        return None, None, None, None
    reg   = max(props, key=lambda p: p.area)
    elong = reg.major_axis_length / (reg.minor_axis_length + 1e-6)
    solid = reg.solidity
    rel   = reg.area / mask_patch.size
    return elong, solid, rel, reg


# ──────────────────────────────────────────────────────────────────────────────
# Load metadata & train.csv
# ──────────────────────────────────────────────────────────────────────────────
df_meta = pd.read_csv(ROI_META, sep=None, engine='python')
df_meta.columns = df_meta.columns.str.strip().str.lower().str.replace(' ', '_')
df_meta['roi_bbox']    = df_meta['roi_bbox'].apply(safe_bbox)
df_meta['defect_bbox'] = df_meta['defect_bbox'].apply(safe_bbox)
df_meta = df_meta.dropna(subset=['roi_bbox'])
print(f'Metadata: {len(df_meta)} ROIs loaded')

train_df = pd.read_csv(TRAIN_CSV, sep=None, engine='python')
train_df.columns = train_df.columns.str.strip()
col_img = [c for c in train_df.columns if 'image' in c.lower()][0]
col_rle = [c for c in train_df.columns if 'encoded' in c.lower() or 'pixel' in c.lower()][0]

# image_id → list of RLE strings (all defect classes)
rle_lookup = {}
for _, row in train_df.iterrows():
    iid = str(row[col_img]).strip()
    rle = row[col_rle]
    if isinstance(rle, str) and rle.strip():
        rle_lookup.setdefault(iid, []).append(rle)

# ──────────────────────────────────────────────────────────────────────────────
# Pick one representative per subtype (highest suitability, image file must exist)
# ──────────────────────────────────────────────────────────────────────────────
reps = {}
for st in SUBTYPES:
    if 'defect_subtype' not in df_meta.columns:
        print(f'  [{st}] defect_subtype column not found')
        reps[st] = None
        continue
    sub = df_meta[df_meta['defect_subtype'] == st].sort_values(
        'suitability_score', ascending=False)
    reps[st] = None
    for _, row in sub.iterrows():
        if (TRAIN_IMAGES / str(row['image_id'])).exists():
            reps[st] = row
            print(f'  [{st:16s}] {row["image_id"]}  '
                  f'class={int(row["class_id"])}  '
                  f'score={row["suitability_score"]:.3f}')
            break
    if reps[st] is None:
        print(f'  [{st:16s}] no matching image file — will show N/A')

# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 18, 13.5
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')

gs_main = gridspec.GridSpec(
    len(SUBTYPES) + 1, len(COL_TITLES) + 1,
    figure=fig,
    height_ratios=[0.32] + [1.0] * len(SUBTYPES),
    width_ratios=[0.52] + [1.0] * len(COL_TITLES),
    hspace=0.06, wspace=0.04,
    top=0.935, bottom=0.05, left=0.04, right=0.985,
)

fig.suptitle(
    'CASDA §3.2.1  Geometric ROI Characterization and Suitability Evaluation\n'
    'One representative ROI per defect subtype  │  '
    'Col 3: geometric analysis (convex hull, axes, metrics)  │  '
    'Col 4: extracted 256×256 ROI patch',
    fontsize=11.5, fontweight='bold', color='#1A237E', y=0.978,
)

# ── Column headers ────────────────────────────────────────────────
fig.add_subplot(gs_main[0, 0]).axis('off')   # top-left corner blank

for j, (title, color) in enumerate(zip(COL_TITLES, COL_COLORS)):
    ax_h = fig.add_subplot(gs_main[0, j + 1])
    ax_h.set_facecolor('#F5F5F5')
    ax_h.set_xticks([]); ax_h.set_yticks([])
    for sp in ax_h.spines.values():
        sp.set_linewidth(1.8); sp.set_edgecolor(color)
    ax_h.text(0.5, 0.5, title,
              ha='center', va='center', fontsize=13.5, fontweight='bold',
              color=color, transform=ax_h.transAxes, multialignment='center')

# ── Data rows ─────────────────────────────────────────────────────
for ri, st in enumerate(SUBTYPES):
    meta  = SUBTYPE_META[st]
    color = meta['color']
    row   = reps[st]

    # Row label (col 0)
    ax_lbl = fig.add_subplot(gs_main[ri + 1, 0])
    ax_lbl.axis('off')
    ax_lbl.set_facecolor('#FAFAFA')
    ax_lbl.axvline(0.94, color=color, linewidth=5, alpha=0.55, clip_on=False)
    ax_lbl.text(0.45, 0.66, meta['label'],
                ha='center', va='center', fontsize=15, fontweight='bold',
                color=color, transform=ax_lbl.transAxes)
    ax_lbl.text(0.45, 0.34, meta['desc'],
                ha='center', va='center', fontsize=11.5,
                color='#455A64', transform=ax_lbl.transAxes,
                multialignment='center')

    # N/A fallback
    if row is None:
        for j in range(len(COL_TITLES)):
            ax = fig.add_subplot(gs_main[ri + 1, j + 1])
            ax.axis('off'); ax.set_facecolor('#F5F5F5')
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    fontsize=11, color='#BDBDBD', transform=ax.transAxes)
        continue

    # ── Load image & mask ────────────────────────────────────────
    image_id    = str(row['image_id']).strip()
    roi_bbox    = row['roi_bbox']           # [x1, y1, x2, y2]
    defect_bbox = row.get('defect_bbox')
    suit_score  = float(row.get('suitability_score', 0))
    class_id    = int(row.get('class_id', 0))
    region_id   = int(row.get('region_id', 0))

    x1r, y1r, x2r, y2r = roi_bbox

    steel_bgr = cv2.imread(str(TRAIN_IMAGES / image_id))
    if steel_bgr is None:
        steel_gray = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    else:
        steel_gray = cv2.cvtColor(steel_bgr, cv2.COLOR_BGR2GRAY)

    full_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for rle in rle_lookup.get(image_id, []):
        full_mask = np.maximum(full_mask, rle_to_mask(rle))

    # Crop window centered on ROI center x
    cx    = (x1r + x2r) // 2
    cx_c  = int(np.clip(cx, CROP_HALF, IMG_W - CROP_HALF))
    xc_l  = cx_c - CROP_HALF
    xc_r  = cx_c + CROP_HALF
    crop_gray = steel_gray[:, xc_l:xc_r]
    crop_mask = full_mask[:, xc_l:xc_r]

    roi_patch_gray = steel_gray[y1r:y2r, x1r:x2r]
    roi_mask_patch = full_mask[y1r:y2r, x1r:x2r]
    h_p, w_p = roi_patch_gray.shape

    elong, solid, rel_area, reg = compute_geom(roi_mask_patch)

    # ── Col 1: Steel crop + ROI window + defect bbox ──────────────
    ax1 = fig.add_subplot(gs_main[ri + 1, 1])
    ax1.imshow(crop_gray, cmap='gray', vmin=0, vmax=255, aspect='auto',
               extent=[xc_l, xc_r, IMG_H, 0])
    ax1.add_patch(Rectangle((x1r, y1r), x2r - x1r, y2r - y1r,
                              linewidth=2.0, edgecolor=color, facecolor='none',
                              linestyle='--', zorder=3))
    if defect_bbox:
        dx1, dy1, dx2, dy2 = defect_bbox
        ax1.add_patch(Rectangle((dx1, dy1), dx2 - dx1, dy2 - dy1,
                                  linewidth=1.5, edgecolor='#FFD600',
                                  facecolor='none', zorder=3))
    ax1.set_xlim(xc_l, xc_r); ax1.set_ylim(IMG_H, 0)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.text(0.02, 0.97, f'Class {class_id}',
             ha='left', va='top', fontsize=7, fontweight='bold', color='white',
             transform=ax1.transAxes,
             bbox=dict(boxstyle='round,pad=0.18', facecolor=color,
                       alpha=0.85, edgecolor='none'))
    for sp in ax1.spines.values():
        sp.set_linewidth(1.5); sp.set_edgecolor(color)

    # ── Col 2: Binary mask ────────────────────────────────────────
    ax2 = fig.add_subplot(gs_main[ri + 1, 2])
    ax2.imshow(crop_mask, cmap='gray', vmin=0, vmax=1, aspect='auto',
               extent=[xc_l, xc_r, IMG_H, 0])
    ax2.add_patch(Rectangle((x1r, y1r), x2r - x1r, y2r - y1r,
                              linewidth=1.8, edgecolor=color, facecolor='none',
                              linestyle='--', zorder=3))
    ax2.set_xlim(xc_l, xc_r); ax2.set_ylim(IMG_H, 0)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.text(0.02, 0.97, 'RLE mask',
             ha='left', va='top', fontsize=7, color='white',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.18', facecolor='#37474F',
                       alpha=0.80, edgecolor='none'))
    for sp in ax2.spines.values():
        sp.set_linewidth(1.5); sp.set_edgecolor('#1565C0')

    # ── Col 3: Geometric overlay (zoomed to defect bbox, shape preserved) ───
    ax3 = fig.add_subplot(gs_main[ri + 1, 3])
    ax3.imshow(roi_patch_gray, cmap='gray', vmin=0, vmax=255, aspect='equal')

    if reg is not None:
        # Mask overlay (semi-transparent red)
        rgba_mask = np.zeros((*roi_patch_gray.shape, 4), dtype=np.float32)
        rgba_mask[roi_mask_patch > 0] = [0.9, 0.2, 0.2, 0.38]
        ax3.imshow(rgba_mask, aspect='equal')

        # Bounding box of largest region
        rr0, cc0, rr1, cc1 = reg.bbox
        ax3.add_patch(Rectangle((cc0, rr0), cc1 - cc0, rr1 - rr0,
                                  linewidth=1.5, edgecolor='#FFD600',
                                  facecolor='none', zorder=4))

        # Major / minor axes
        y_c, x_c = reg.centroid
        ori = reg.orientation
        ma  = reg.major_axis_length / 2
        mi  = reg.minor_axis_length / 2
        ax3.plot([x_c - np.cos(ori) * ma, x_c + np.cos(ori) * ma],
                 [y_c + np.sin(ori) * ma, y_c - np.sin(ori) * ma],
                 color='#FFD600', lw=2.0, zorder=5)
        ax3.plot([x_c + np.sin(ori) * mi, x_c - np.sin(ori) * mi],
                 [y_c + np.cos(ori) * mi, y_c - np.cos(ori) * mi],
                 color='#80DEEA', lw=1.4, linestyle='--', zorder=5)

        # Convex hull outline
        hull_img  = reg.convex_image.astype(np.uint8)
        rr0b, cc0b = reg.bbox[:2]
        hull_full = np.zeros_like(roi_mask_patch)
        hull_full[rr0b:rr0b + hull_img.shape[0],
                  cc0b:cc0b + hull_img.shape[1]] = hull_img
        contours, _ = cv2.findContours(hull_full, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull_pts = cv2.convexHull(contours[0])[:, 0, :]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax3.plot(hull_pts[:, 0], hull_pts[:, 1],
                     color='#69F0AE', lw=1.6, zorder=5)

        # Metric annotations
        ax3.text(0.97, 0.97,
                 f'Elong: {elong:.2f}\nSolid: {solid:.2f}\nRelA:  {rel_area:.4f}',
                 ha='right', va='top', fontsize=6.5, color='white',
                 transform=ax3.transAxes, fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#212121',
                           alpha=0.82, edgecolor='none'))

        # Zoom to defect bbox with padding — preserves bbox shape via aspect='equal'
        pad  = max(20, int(0.25 * max(rr1 - rr0, cc1 - cc0)))
        ax3.set_xlim(max(0, cc0 - pad), min(w_p, cc1 + pad))
        ax3.set_ylim(min(h_p, rr1 + pad), max(0, rr0 - pad))
    else:
        ax3.set_xlim(0, w_p)
        ax3.set_ylim(h_p, 0)

    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values():
        sp.set_linewidth(1.5); sp.set_edgecolor('#2E7D32')

    # ── Col 4: Extracted ROI patch + badges ──────────────────────
    ax4 = fig.add_subplot(gs_main[ri + 1, 4])

    roi_fname = f"{image_id}_class{class_id}_region{region_id}.png"
    roi_fpath = ROI_IMGS_DIR / roi_fname
    if roi_fpath.exists():
        roi_img = np.array(Image.open(roi_fpath).convert('L'))
    else:
        roi_img = roi_patch_gray

    ax4.imshow(roi_img, cmap='gray', vmin=0, vmax=255, aspect='equal')
    ax4.set_xticks([]); ax4.set_yticks([])

    # Subtype badge (bottom center)
    ax4.text(0.5, 0.05, meta['label'],
             ha='center', va='bottom', fontsize=7.5, fontweight='bold',
             color='white', transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.28', facecolor=color,
                       alpha=0.90, edgecolor='none'))

    # Suitability score (top left)
    score_color = '#E65100' if suit_score >= 0.75 else ('#2E7D32' if suit_score >= 0.5 else '#C62828')
    ax4.text(0.04, 0.96, f'S={suit_score:.3f}',
             ha='left', va='top', fontsize=7.5, fontweight='bold',
             color='white', transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.22', facecolor=score_color,
                       alpha=0.90, edgecolor='none'))

    for sp in ax4.spines.values():
        sp.set_linewidth(2.5); sp.set_edgecolor(color)

# ── Legend ────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=SUBTYPE_META[st]['color'], alpha=0.85,
                   label=f"{SUBTYPE_META[st]['label']}: "
                         f"{SUBTYPE_META[st]['desc'].replace(chr(10), '  ')}")
    for st in SUBTYPES
]
legend_handles += [
    mpatches.Patch(facecolor='#FFD600', alpha=0.85,
                   label='Bounding box (yellow) / Major axis'),
    mpatches.Patch(facecolor='#69F0AE', alpha=0.85, label='Convex hull'),
    mpatches.Patch(facecolor='#80DEEA', alpha=0.75, label='Minor axis'),
    mpatches.Patch(facecolor='#E65100', alpha=0.75,
                   label='S score: ≥0.75 orange / ≥0.5 green / <0.5 red'),
]
fig.legend(handles=legend_handles, loc='lower center',
           ncol=4, fontsize=7.5, framealpha=0.90,
           bbox_to_anchor=(0.5, 0.002),
           columnspacing=1.0, handlelength=1.2)

# ── Save ──────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none',
            format='jpeg', pil_kwargs={'quality': 95, 'subsampling': 0})
print(f'\nSaved → {OUT_FILE}')
plt.show()
print('Done.')
