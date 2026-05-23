"""
figure_stageA_grid.py — Background Texture Categories
=============================================================================
§3.2.2 ControlNet Configuration — background texture classification

5 panels (one per background type), each showing:
  Row 1: representative 256×256 ROI patch from Severstal data
  Row 2: description box (classification criteria)

Background types: smooth / vertical_stripe / horizontal_stripe / textured / complex

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_stageA_grid.py
"""

import ast
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
ROI_META     = DRIVE_DATA / "roi_patches_v5.1/roi_metadata.csv"

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "stageA_grid.jpg"

IMG_H, IMG_W = 256, 1600

# ─────────────────────────────────────────────────────────────────────────────
# Background texture definitions
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

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def safe_bbox(val):
    try:
        return [int(v) for v in ast.literal_eval(str(val))]
    except Exception:
        return None


def classify_bg(patch_gray):
    patch = cv2.resize(patch_gray, (128, 128), interpolation=cv2.INTER_AREA)
    var   = np.var(patch.astype(np.float32))
    edges = cv2.Canny(patch, 50, 150)
    ed    = np.count_nonzero(edges) / edges.size
    if var < 200 and ed < 0.02:
        return 'smooth'
    sx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    ex, ey = np.mean(np.abs(sx)), np.mean(np.abs(sy))
    tot = ex + ey + 1e-7
    if ex / tot > 0.65 and ed > 0.02:
        return 'vertical_stripe'
    if ey / tot > 0.65 and ed > 0.02:
        return 'horizontal_stripe'
    if ed > 0.15:
        return 'complex'
    if var > 500 or ed > 0.05:
        return 'textured'
    return 'smooth'


def synthetic_bg(bg_type, size=256):
    """Fallback synthetic texture when no real example is found."""
    np.random.seed({'smooth': 0, 'vertical_stripe': 1, 'horizontal_stripe': 2,
                    'textured': 3, 'complex': 4}.get(bg_type, 0))
    img = np.ones((size, size), dtype=np.uint8) * 140
    if bg_type == 'smooth':
        img[:] = 155
        img = np.clip(img.astype(float) + np.random.normal(0, 4, (size, size)),
                      0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 2)
    elif bg_type == 'vertical_stripe':
        for x in range(0, size, 16):
            img[:, x:x + 8]    = 95
            img[:, x + 8:x+16] = 185
        img = cv2.GaussianBlur(img, (1, 7), 2)
    elif bg_type == 'horizontal_stripe':
        for y in range(0, size, 16):
            img[y:y + 8, :]    = 95
            img[y + 8:y + 16, :] = 185
        img = cv2.GaussianBlur(img, (7, 1), 2)
    elif bg_type == 'textured':
        noise = np.random.normal(130, 45, (size, size))
        img = np.clip(noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif bg_type == 'complex':
        noise = np.random.normal(130, 50, (size, size))
        img = np.clip(noise, 0, 255).astype(np.uint8)
        for i in range(0, size, 20):
            cv2.line(img, (0, i), (size, (i + 40) % size), 55, 2)
            cv2.line(img, (i, 0), ((i + 40) % size, size), 75, 2)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Load metadata & find representative patch per background type
# ─────────────────────────────────────────────────────────────────────────────
print("Loading ROI metadata ...")
df_meta = pd.read_csv(ROI_META, sep=None, engine='python')
df_meta.columns = df_meta.columns.str.strip().str.lower().str.replace(' ', '_')

# Normalize: 'complex_pattern' → 'complex'
if 'background_type' in df_meta.columns:
    df_meta['background_type'] = (df_meta['background_type']
                                  .str.replace('complex_pattern', 'complex', regex=False))

sort_col = 'suitability_score' if 'suitability_score' in df_meta.columns else None

bg_reps = {}   # bg_type -> {'patch': ndarray, 'image_id': str, 'synthetic': bool}

for bg in BG_TYPES:
    if 'background_type' in df_meta.columns:
        sub = df_meta[df_meta['background_type'] == bg].copy()
    else:
        sub = df_meta.copy()

    if sort_col:
        sub = sub.sort_values(sort_col, ascending=False)

    found = False
    for _, row in sub.iterrows():
        img_path = TRAIN_IMAGES / str(row['image_id'])
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        bbox = safe_bbox(row.get('roi_bbox'))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if 'background_type' not in df_meta.columns and classify_bg(crop) != bg:
            continue
        bg_reps[bg] = {'patch': crop, 'image_id': str(row['image_id']),
                       'score': float(row.get('suitability_score', 0)),
                       'synthetic': False}
        print(f"  [{bg:20s}] {row['image_id']}  score={row.get('suitability_score',0):.3f}")
        found = True
        break

    if not found:
        print(f"  [{bg:20s}] no real example found — using synthetic placeholder")
        bg_reps[bg] = {'patch': synthetic_bg(bg), 'image_id': '(synthetic)',
                       'score': 0.0, 'synthetic': True}

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout  (Section A only)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 7), facecolor='white')

gs_a = gridspec.GridSpec(
    2, 5,
    height_ratios=[4.0, 1.2],
    wspace=0.055, hspace=0.06,
    top=0.90, bottom=0.04, left=0.02, right=0.99,
)

for j, bg in enumerate(BG_TYPES):
    meta  = BG_META[bg]
    rep   = bg_reps[bg]
    patch = rep['patch']

    patch_sq   = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA)
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    patch_disp = clahe.apply(patch_sq)

    # ── Image panel ────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs_a[0, j])
    ax_img.imshow(patch_disp, cmap='gray', vmin=0, vmax=255, aspect='equal')
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_linewidth(4.0); sp.set_edgecolor(meta['color'])

    ax_img.set_title(f"({meta['abbr']})  {meta['label']}",
                     fontsize=14, fontweight='bold', color=meta['color'], pad=7)

    src = '(synthetic)' if rep['synthetic'] else rep['image_id'][:16]
    ax_img.text(0.5, -0.04, src, ha='center', va='top',
                fontsize=8.5, color='#9E9E9E', style='italic',
                transform=ax_img.transAxes)

    # ── Description panel ──────────────────────────────────────────
    ax_desc = fig.add_subplot(gs_a[1, j])
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
