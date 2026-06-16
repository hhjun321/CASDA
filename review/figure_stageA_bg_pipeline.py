"""
figure_stageA_bg_pipeline.py — Background Texture Classification Pipeline
==========================================================================
§3.2.2 ControlNet Configuration — background texture classification procedure

Layout: 5 rows (background types) × 4 columns (extraction pipeline)

  Col 1: Steel image crop (400 px wide) + 64px grid + classified cells + ROI bbox
  Col 2: Canny edge overlay on ROI patch — visualises edge_density feature
  Col 3: Sobel direction map (R=Sobel_x, B=Sobel_y) — visualises direction features
  Col 4: Classified 256×256 ROI patch + background type badge + metrics

Rows (one representative per background type, highest suitability score):
  smooth / vertical_stripe / horizontal_stripe / textured / complex

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_stageA_bg_pipeline.py
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

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
ROI_META     = DRIVE_DATA / "roi_patches_v5.1/roi_metadata.csv"
ROI_IMGS_DIR = DRIVE_DATA / "roi_patches_v5.1/images"

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "stageA_bg_pipeline.png"

IMG_H, IMG_W = 256, 1600
CROP_HALF    = 200    # crop half-width → 400 px wide crop
GRID_SIZE    = 64

# ─────────────────────────────────────────────────────────────────────────────
# Background type definitions
# ─────────────────────────────────────────────────────────────────────────────
BG_TYPES = ['smooth', 'vertical_stripe', 'horizontal_stripe', 'textured', 'complex']

BG_META = {
    'smooth': dict(
        color='#1565C0', light='#BBDEFB', abbr='SMO',
        label='Smooth',
        desc='σ² < 200\nEdge density < 2%\nUniform surface finish',
    ),
    'vertical_stripe': dict(
        color='#2E7D32', light='#C8E6C9', abbr='VER',
        label='Vertical Stripe',
        desc='Sobel_x > 65%\nStrong vertical edges\nRolling / grinding marks',
    ),
    'horizontal_stripe': dict(
        color='#E65100', light='#FFE0B2', abbr='HOR',
        label='Horizontal Stripe',
        desc='Sobel_y > 65%\nStrong horizontal edges\nLayered surface bands',
    ),
    'textured': dict(
        color='#6A1B9A', light='#E1BEE7', abbr='TEX',
        label='Textured',
        desc='σ² > 500\nEdge density 5–15%\nRich micro-texture',
    ),
    'complex': dict(
        color='#C62828', light='#FFCDD2', abbr='COM',
        label='Complex',
        desc='Edge density > 15%\nMulti-directional\nOverlapping structures',
    ),
}

COL_TITLES = [
    'Steel Image Crop\n(400×256 px) + 64px Grid',
    'Edge Density Map\n(Canny  threshold 50/150)',
    'Sobel Direction Map\n(R = Sobel_x  │  B = Sobel_y)',
    'Classified ROI Patch\n(256×256 px)',
]
COL_COLORS = ['#37474F', '#0D47A1', '#4A148C', '#E65100']

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def safe_bbox(val):
    try:
        return [int(v) for v in ast.literal_eval(str(val))]
    except Exception:
        return None


def classify_bg_with_metrics(patch_gray):
    """Run the classification algorithm and return all intermediate metrics."""
    patch = cv2.resize(patch_gray, (128, 128), interpolation=cv2.INTER_AREA)
    var   = float(np.var(patch.astype(np.float32)))
    edges = cv2.Canny(patch, 50, 150)
    ed    = float(np.count_nonzero(edges) / edges.size)

    sx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    ex = float(np.mean(np.abs(sx)))
    ey = float(np.mean(np.abs(sy)))
    tot = ex + ey + 1e-7

    if var < 200 and ed < 0.02:
        bg = 'smooth'
    elif ex / tot > 0.65 and ed > 0.02:
        bg = 'vertical_stripe'
    elif ey / tot > 0.65 and ed > 0.02:
        bg = 'horizontal_stripe'
    elif ed > 0.15:
        bg = 'complex'
    elif var > 500 or ed > 0.05:
        bg = 'textured'
    else:
        bg = 'smooth'

    return dict(bg_type=bg, variance=var, edge_density=ed,
                sobel_x_ratio=ex / tot, sobel_y_ratio=ey / tot)


def canny_edge_overlay(patch_gray_256, color_hex):
    """Canny edges (computed on 128px) overlaid in color on grayscale patch."""
    p128  = cv2.resize(patch_gray_256, (128, 128), interpolation=cv2.INTER_AREA)
    e128  = cv2.Canny(p128, 50, 150)
    e256  = cv2.resize(e128, (256, 256), interpolation=cv2.INTER_NEAREST)
    rgb   = np.stack([patch_gray_256] * 3, axis=-1).copy()
    r = int(color_hex[1:3], 16)
    g = int(color_hex[3:5], 16)
    b = int(color_hex[5:7], 16)
    mask = e256 > 0
    rgb[mask, 0] = r
    rgb[mask, 1] = g
    rgb[mask, 2] = b
    return rgb


def sobel_direction_map(patch_gray_256):
    """False-color Sobel map: R = |Sobel_x|, B = |Sobel_y|."""
    sx = cv2.Sobel(patch_gray_256.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(patch_gray_256.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mx = max(float(np.abs(sx).max()), float(np.abs(sy).max()), 1.0)
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(np.abs(sx) / mx * 255, 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip(np.abs(sy) / mx * 255, 0, 255).astype(np.uint8)
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Load metadata & select one representative per background type
# ─────────────────────────────────────────────────────────────────────────────
print("Loading ROI metadata ...")
df_meta = pd.read_csv(ROI_META, sep=None, engine='python')
df_meta.columns = df_meta.columns.str.strip().str.lower().str.replace(' ', '_')
df_meta['roi_bbox'] = df_meta['roi_bbox'].apply(safe_bbox)
df_meta = df_meta.dropna(subset=['roi_bbox'])

if 'background_type' in df_meta.columns:
    df_meta['background_type'] = (df_meta['background_type']
                                  .str.replace('complex_pattern', 'complex', regex=False))

print(f"  {len(df_meta)} ROIs loaded")

reps = {bg: None for bg in BG_TYPES}

for bg in BG_TYPES:
    if 'background_type' in df_meta.columns:
        sub = df_meta[df_meta['background_type'] == bg]
    else:
        sub = df_meta
    sub = sub.sort_values('suitability_score', ascending=False)

    for _, row in sub.iterrows():
        img_path = TRAIN_IMAGES / str(row['image_id'])
        if not img_path.exists():
            continue
        reps[bg] = row
        print(f"  [{bg:20s}] {row['image_id']}  score={row['suitability_score']:.3f}")
        break

    if reps[bg] is None:
        print(f"  [{bg:20s}] no example found")

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 18, 16.5
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')

gs_main = gridspec.GridSpec(
    len(BG_TYPES) + 1, len(COL_TITLES) + 1,
    figure=fig,
    height_ratios=[0.32] + [1.0] * len(BG_TYPES),
    width_ratios=[0.52] + [1.0] * len(COL_TITLES),
    hspace=0.06, wspace=0.04,
    top=0.935, bottom=0.05, left=0.04, right=0.985,
)

fig.suptitle(
    'CASDA §3.2.2  Background Texture Classification Pipeline\n'
    'One representative ROI per background type  │  '
    'Col 2: Canny edge density  │  '
    'Col 3: Sobel direction (R = Sobel_x  B = Sobel_y)  │  '
    'Col 4: classified 256×256 ROI patch',
    fontsize=11.5, fontweight='bold', color='#1A237E', y=0.978,
)

# ── Column headers ─────────────────────────────────────────────────────────
fig.add_subplot(gs_main[0, 0]).axis('off')

for j, (title, hdr_color) in enumerate(zip(COL_TITLES, COL_COLORS)):
    ax_h = fig.add_subplot(gs_main[0, j + 1])
    ax_h.set_facecolor('#F5F5F5')
    ax_h.set_xticks([]); ax_h.set_yticks([])
    for sp in ax_h.spines.values():
        sp.set_linewidth(1.8); sp.set_edgecolor(hdr_color)
    ax_h.text(0.5, 0.5, title,
              ha='center', va='center', fontsize=13.5, fontweight='bold',
              color=hdr_color, transform=ax_h.transAxes, multialignment='center')

# ── Data rows ──────────────────────────────────────────────────────────────
for ri, bg in enumerate(BG_TYPES):
    meta  = BG_META[bg]
    color = meta['color']
    row   = reps[bg]

    # ── Row label ──────────────────────────────────────────────────
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

    # ── Load image data ────────────────────────────────────────────
    image_id  = str(row['image_id']).strip()
    x1r, y1r, x2r, y2r = row['roi_bbox']
    suit_score = float(row.get('suitability_score', 0))
    class_id   = int(row.get('class_id', 0))
    region_id  = int(row.get('region_id', 0))

    full_img  = cv2.imread(str(TRAIN_IMAGES / image_id), cv2.IMREAD_GRAYSCALE)
    roi_patch = full_img[y1r:y2r, x1r:x2r]
    roi_256   = cv2.resize(roi_patch, (256, 256), interpolation=cv2.INTER_AREA)
    metrics   = classify_bg_with_metrics(roi_patch)

    # ── Col 1: Steel crop + 64px grid + ROI bbox ───────────────────
    ax1 = fig.add_subplot(gs_main[ri + 1, 1])

    cx      = (x1r + x2r) // 2
    crop_x1 = max(0, cx - CROP_HALF)
    crop_x2 = min(IMG_W, cx + CROP_HALF)
    crop    = full_img[0:IMG_H, crop_x1:crop_x2]

    ax1.imshow(crop, cmap='gray', vmin=0, vmax=255, aspect='auto',
               extent=[crop_x1, crop_x2, IMG_H, 0])

    # 64px grid
    for gx in range((crop_x1 // GRID_SIZE) * GRID_SIZE, crop_x2 + 1, GRID_SIZE):
        ax1.axvline(gx, color='cyan', linewidth=0.5, alpha=0.55)
    for gy in range(0, IMG_H + 1, GRID_SIZE):
        ax1.axhline(gy, color='cyan', linewidth=0.5, alpha=0.55)

    # Cells overlapping with ROI — filled with background type color
    for gx in range((x1r // GRID_SIZE) * GRID_SIZE, x2r, GRID_SIZE):
        for gy in range((y1r // GRID_SIZE) * GRID_SIZE, y2r, GRID_SIZE):
            if crop_x1 <= gx < crop_x2:
                ax1.add_patch(Rectangle(
                    (gx, gy), GRID_SIZE, GRID_SIZE,
                    facecolor=color, alpha=0.25, edgecolor='none',
                ))

    # ROI bbox
    ax1.add_patch(Rectangle(
        (x1r, y1r), x2r - x1r, y2r - y1r,
        linewidth=2.0, edgecolor=color, facecolor='none', linestyle='--',
    ))

    ax1.set_xlim(crop_x1, crop_x2)
    ax1.set_ylim(IMG_H, 0)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_linewidth(1.5); sp.set_edgecolor(color)

    ax1.text(0.03, 0.04, image_id[:14],
             ha='left', va='bottom', fontsize=6.5, color='white',
             transform=ax1.transAxes,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#212121',
                       alpha=0.72, edgecolor='none'))

    # ── Col 2: Canny edge overlay ──────────────────────────────────
    ax2 = fig.add_subplot(gs_main[ri + 1, 2])
    ax2.imshow(canny_edge_overlay(roi_256, color), aspect='equal')
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_linewidth(1.5); sp.set_edgecolor(color)

    ax2.text(0.97, 0.97,
             f'var:  {metrics["variance"]:7.1f}\n'
             f'ed:   {metrics["edge_density"]:.4f}',
             ha='right', va='top', fontsize=7, color='white',
             transform=ax2.transAxes, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#212121',
                       alpha=0.82, edgecolor='none'))

    # ── Col 3: Sobel direction map ─────────────────────────────────
    ax3 = fig.add_subplot(gs_main[ri + 1, 3])
    ax3.imshow(sobel_direction_map(roi_256), aspect='equal')
    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values():
        sp.set_linewidth(1.5); sp.set_edgecolor(color)

    sx_r = metrics['sobel_x_ratio']
    sy_r = metrics['sobel_y_ratio']
    dominant = ('Sobel_x ▶' if sx_r > sy_r else 'Sobel_y ▶')
    ax3.text(0.97, 0.97,
             f'Sx: {sx_r:.3f}\nSy: {sy_r:.3f}\n{dominant}',
             ha='right', va='top', fontsize=7, color='white',
             transform=ax3.transAxes, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#212121',
                       alpha=0.82, edgecolor='none'))

    # ── Col 4: Classified ROI patch ────────────────────────────────
    ax4 = fig.add_subplot(gs_main[ri + 1, 4])

    roi_fname = f"{image_id}_class{class_id}_region{region_id}.png"
    roi_fpath = ROI_IMGS_DIR / roi_fname
    roi_display = (np.array(Image.open(roi_fpath).convert('L'))
                   if roi_fpath.exists() else roi_256)

    ax4.imshow(roi_display, cmap='gray', vmin=0, vmax=255, aspect='equal')
    ax4.set_xticks([]); ax4.set_yticks([])

    ax4.text(0.5, 0.05, f"{meta['abbr']}  {meta['label']}",
             ha='center', va='bottom', fontsize=8, fontweight='bold',
             color='white', transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.28', facecolor=color,
                       alpha=0.90, edgecolor='none'))

    sc_color = '#E65100' if suit_score >= 0.75 else ('#2E7D32' if suit_score >= 0.5 else '#C62828')
    ax4.text(0.04, 0.96, f'S={suit_score:.3f}',
             ha='left', va='top', fontsize=8, fontweight='bold',
             color='white', transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.22', facecolor=sc_color,
                       alpha=0.90, edgecolor='none'))

    for sp in ax4.spines.values():
        sp.set_linewidth(2.5); sp.set_edgecolor(color)

# ── Legend ─────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=BG_META[bg]['color'], alpha=0.85,
                   label=f"{BG_META[bg]['abbr']}: {BG_META[bg]['label']}")
    for bg in BG_TYPES
] + [
    mpatches.Patch(facecolor='cyan',    alpha=0.7,  label='64 px sliding grid'),
    mpatches.Patch(facecolor='#FF0000', alpha=0.7,  label='Col 3 R: Sobel_x  (vertical edge strength)'),
    mpatches.Patch(facecolor='#0000FF', alpha=0.7,  label='Col 3 B: Sobel_y  (horizontal edge strength)'),
    mpatches.Patch(facecolor='#E65100', alpha=0.75, label='S score: ≥0.75 orange / ≥0.5 green / <0.5 red'),
]
fig.legend(handles=legend_handles, loc='lower center',
           ncol=4, fontsize=7.5, framealpha=0.90,
           bbox_to_anchor=(0.5, 0.002),
           columnspacing=1.0, handlelength=1.2)

# ── Save ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none',
            format='png')
print(f'\nSaved → {OUT_FILE}')
plt.show()
print('Done.')
