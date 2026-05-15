"""
CASDA Stage A: ROI Extraction & Multi-Channel Hint Generation
Loads actual Severstal steel images, train.csv (RLE masks),
and roi_metadata.csv to produce a publication-quality figure.
Run in Google Colab.

Outputs:
  casda_roi_extraction.png   — combined figure
  casda_panel1_overview.png  — Panel 1: steel image + ROI windows
  casda_panel2_scoregrid.png — Panel 2a: suitability score grid (image bg)
  casda_panel2_hints.png     — Panel 2b: multi-channel hint channels
  casda_panel3_summary.png   — Panel 3: pipeline summary

Adjust DEMO_IMAGE_ID to any image with multiple ROIs (e.g. 0002cc93b.jpg).
"""

import ast, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from PIL import Image

# ════════════════════════════════════════════════════════════════
# PATHS  — update to match your Colab environment
# ════════════════════════════════════════════════════════════════
DRIVE          = '/content/drive/MyDrive/data/Severstal'
TRAIN_IMAGES   = f'{DRIVE}/train_images'
TRAIN_CSV      = f'{DRIVE}/train.csv'
ROI_METADATA   = f'{DRIVE}/roi_patches_v5.1/roi_metadata.csv'
ROI_IMAGES_DIR = f'{DRIVE}/roi_patches_v5.1/images'
ROI_MASKS_DIR  = f'{DRIVE}/roi_patches_v5.1/masks'

DEMO_IMAGE_ID = '0002cc93b.jpg'
THRESH        = 0.50
CELL          = 64                # grid cell size (px)

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def rle_to_mask(rle_str, h=256, w=1600):
    flat = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle_str, str) and rle_str.strip():
        nums = list(map(int, rle_str.split()))
        for start, length in zip(nums[0::2], nums[1::2]):
            flat[start - 1:start - 1 + length] = 1
    return flat.reshape(w, h).T

def safe_literal(val):
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return None

def save_panel(fig, axes_list, filename, pad=0.15, dpi=180):
    """Extract bounding box union of axes_list from fig and save."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = []
    for ax in axes_list:
        tb = ax.get_tightbbox(renderer)
        if tb is not None:
            bboxes.append(tb)
    if not bboxes:
        print(f'Warning: no valid bboxes for {filename}')
        return
    union_bb = Bbox.union(bboxes)
    bb_in = union_bb.transformed(fig.dpi_scale_trans.inverted())
    padded = Bbox([[bb_in.x0 - pad, bb_in.y0 - pad],
                   [bb_in.x1 + pad, bb_in.y1 + pad]])
    fig.savefig(filename, dpi=dpi, bbox_inches=padded,
                facecolor='white', edgecolor='none')
    print(f'Saved: {filename}')

# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════
df_all = pd.read_csv(ROI_METADATA, sep=None, engine='python')
print(f'[metadata] columns: {list(df_all.columns[:6])} ...')
print(f'[metadata] shape:   {df_all.shape}')

df_all.columns = df_all.columns.str.strip().str.lower().str.replace(' ', '_')
df_all['roi_bbox']    = df_all['roi_bbox'].apply(safe_literal)
df_all['defect_bbox'] = df_all['defect_bbox'].apply(safe_literal)
df_all = df_all.dropna(subset=['roi_bbox'])

df_img = df_all[df_all['image_id'] == DEMO_IMAGE_ID].reset_index(drop=True)
if df_img.empty:
    available = df_all['image_id'].unique()[:10]
    raise ValueError(f"DEMO_IMAGE_ID '{DEMO_IMAGE_ID}' not found.\n"
                     f"Available (first 10): {list(available)}")
print(f'{DEMO_IMAGE_ID}: {len(df_img)} ROIs in metadata')

steel_np = np.array(Image.open(os.path.join(TRAIN_IMAGES, DEMO_IMAGE_ID)))
if steel_np.ndim == 3:
    steel_gray = steel_np.mean(axis=2).astype(np.float32) / 255.0
else:
    steel_gray = steel_np.astype(np.float32) / 255.0
IMG_H, IMG_W = steel_gray.shape   # 256, 1600

train_df = pd.read_csv(TRAIN_CSV, sep=None, engine='python')
train_df.columns = train_df.columns.str.strip()
col_img = [c for c in train_df.columns if 'image' in c.lower()][0]
col_rle = [c for c in train_df.columns if 'encoded' in c.lower() or 'pixel' in c.lower()][0]
full_mask = np.zeros((IMG_H, IMG_W), dtype=np.float32)
img_rows  = train_df[train_df[col_img] == DEMO_IMAGE_ID]
for _, row in img_rows.iterrows():
    full_mask = np.maximum(full_mask,
                           rle_to_mask(row[col_rle], IMG_H, IMG_W))

# ════════════════════════════════════════════════════════════════
# DEMO ROI — highest suitability score
# ════════════════════════════════════════════════════════════════
demo_row  = df_img.loc[df_img['suitability_score'].idxmax()]
demo_bbox = demo_row['roi_bbox']

demo_fname = (f"{demo_row['image_id']}"
              f"_class{int(demo_row['class_id'])}"
              f"_region{int(demo_row['region_id'])}.png")
roi_gray = (np.array(Image.open(os.path.join(ROI_IMAGES_DIR, demo_fname))
                     .convert('L'), dtype=np.float32) / 255.0)
roi_msk  = (np.array(Image.open(os.path.join(ROI_MASKS_DIR,  demo_fname))
                     .convert('L'), dtype=np.float32) / 255.0)
roi_msk  = (roi_msk > 0.5).astype(np.float32)

# 3-channel hint from actual ROI
ch_R = np.clip(gaussian_filter(roi_msk, sigma=2.5) * 2.2, 0, 1)

sx = sobel(roi_gray, axis=1);  sy = sobel(roi_gray, axis=0)
edges = np.hypot(sx, sy);      edges /= edges.max() + 1e-8
ch_G  = np.clip(gaussian_filter(edges, sigma=0.5) * 2.2, 0, 1)

mu    = uniform_filter(roi_gray, size=8)
rough = np.sqrt(np.clip(uniform_filter((roi_gray - mu) ** 2, size=8), 0, None))
rough /= rough.max() + 1e-8
ch_B  = np.clip(rough * 1.6, 0, 1)

# ════════════════════════════════════════════════════════════════
# SCORE GRID — CELL×CELL cells from actual defect mask
# ════════════════════════════════════════════════════════════════
NC = IMG_W // CELL   # 25
NR = IMG_H // CELL   #  4

grid_s = np.zeros((NR, NC), dtype=np.float32)
for r in range(NR):
    for c in range(NC):
        cm_ = full_mask[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL]
        cs_ = steel_gray[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL]
        grid_s[r, c] = np.clip(cm_.mean() * 18.0 + cs_.std() * 1.2, 0.0, 1.0)

# ════════════════════════════════════════════════════════════════
# FIGURE LAYOUT
# ════════════════════════════════════════════════════════════════
SCORE_COL = {'suitable': '#FFD600', 'acceptable': '#2E7D32'}
cmap_sg   = plt.get_cmap('viridis')
norm_sg   = Normalize(vmin=0, vmax=1)

fig = plt.figure(figsize=(18, 13))
gs0 = GridSpec(3, 1, figure=fig,
               height_ratios=[2.2, 5.0, 0.9],
               hspace=0.42, top=0.94, bottom=0.04,
               left=0.04, right=0.97)

# ── PANEL 1: Actual steel image + ROI windows ────────────────────
ax_i = fig.add_subplot(gs0[0])
ax_i.imshow(steel_gray, cmap='gray', aspect='auto', vmin=0, vmax=1,
            extent=[0, IMG_W, IMG_H, 0])

for _, row in df_img.iterrows():
    x1, y1, x2, y2 = row['roi_bbox']
    sc       = row['suitability_score']
    col      = SCORE_COL.get(row['recommendation'], '#1565C0')
    is_demo  = (row['region_id'] == demo_row['region_id'] and
                row['class_id']  == demo_row['class_id'])
    lw = 2.8 if is_demo else 1.6
    ax_i.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                              linewidth=0, facecolor=col,
                              alpha=0.18 if is_demo else 0.08))
    ax_i.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                              linewidth=lw, edgecolor=col, facecolor='none'))
    ax_i.text((x1+x2)/2, (y1+y2)/2, f'{sc:.3f}',
              ha='center', va='center', fontsize=6.5,
              color='white', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.12',
                        facecolor=col, alpha=0.78, edgecolor='none'))
    if is_demo:
        ax_i.text((x1+x2)/2, y1 - 14, '★ demo ROI',
                  ha='center', va='bottom', fontsize=7.5,
                  color='#E65100', fontweight='bold')
    else:
        ax_i.text((x1+x2)/2, y1 + 12,
                  row['defect_subtype'].replace('_', '\n'),
                  ha='center', va='top', fontsize=5.5,
                  color='white', alpha=0.9)

ax_i.set_title(
    f'Input: {IMG_W}×{IMG_H} px  │  Image: {DEMO_IMAGE_ID}  │  '
    f'ROI size: {x2-x1}×{y2-y1} px  │  '
    f'{len(df_img)} overlapping ROI windows extracted',
    fontsize=9.5, fontweight='bold', color='#1A237E', pad=7)
ax_i.set_xlim(0, IMG_W)
ax_i.set_ylim(IMG_H + 18, -30)
ax_i.set_xlabel('Pixel position (strip width)', fontsize=8.5)
ax_i.set_yticks([])
ax_i.tick_params(axis='x', labelsize=7.5)

n_suit = (df_img['recommendation'] == 'suitable').sum()
n_acc  = (df_img['recommendation'] == 'acceptable').sum()
leg = [
    mpatches.Patch(facecolor='#FFD600', alpha=0.85,
                   label=f'Suitable  (score ≥ 0.75)  n={n_suit}'),
    mpatches.Patch(facecolor='#2E7D32', alpha=0.80,
                   label=f'Acceptable  (0.5–0.75)  n={n_acc}'),
    mpatches.Patch(facecolor='#FFD600', alpha=0.90,
                   label=f'Demo ROI  (score={demo_row["suitability_score"]:.3f})'),
]
ax_i.legend(handles=leg, loc='upper right', fontsize=7.5, framealpha=0.88)

# ── PANEL 2 layout: header row + content row ─────────────────────
# 2×2 grid: [header-grid | header-hints] / [score-grid | hint-channels]
gs1 = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[1],
                               height_ratios=[0.09, 1.0],
                               width_ratios=[1.15, 1.85],
                               wspace=0.26, hspace=0.08)

# Header: score grid
ax_g_hdr = fig.add_subplot(gs1[0, 0])
ax_g_hdr.axis('off')
ax_g_hdr.text(0.5, 0.35,
              f'Suitability Score Grid  ({NC}×{NR} cells, {CELL} px each)\n'
              'Actual image + semi-transparent overlay  │  Gold = demo ROI',
              ha='center', va='center', fontsize=8.5, fontweight='bold',
              color='#1A237E', transform=ax_g_hdr.transAxes)

# Header: hint channels
ax_h_hdr = fig.add_subplot(gs1[0, 1])
ax_h_hdr.axis('off')
ax_h_hdr.text(0.5, 0.35,
              f'Multi-Channel Hint Image Generation  '
              f'(score={demo_row["suitability_score"]:.3f}, '
              f'{demo_row["recommendation"].upper()})',
              ha='center', va='center', fontsize=8.8, fontweight='bold',
              color='#1A237E', transform=ax_h_hdr.transAxes)

# ─── Score grid: actual image background + transparent score overlay ──
ax_g = fig.add_subplot(gs1[1, 0])

# Background: full steel strip at native pixel coordinates
ax_g.imshow(steel_gray, cmap='gray', aspect='auto', vmin=0, vmax=1,
            extent=[0, IMG_W, IMG_H, 0])

# Semi-transparent viridis cells on top
for r in range(NR):
    for c in range(NC):
        s    = float(grid_s[r, c])
        x0   = c * CELL;  y0 = r * CELL
        rgba = list(cmap_sg(norm_sg(s)));  rgba[3] = 0.48   # alpha 48%
        ax_g.add_patch(Rectangle((x0, y0), CELL, CELL,
                                  facecolor=rgba,
                                  edgecolor=(1.0, 1.0, 1.0, 0.65),
                                  linewidth=0.6))
        ax_g.text(x0 + CELL / 2, y0 + CELL / 2, f'{s:.2f}',
                  ha='center', va='center', fontsize=4.5,
                  color='white', fontweight='bold')

# ROI footprints in pixel coordinates (same space as image)
for _, row in df_img.iterrows():
    x1, y1, x2, y2 = row['roi_bbox']
    is_demo = (row['region_id'] == demo_row['region_id'] and
               row['class_id']  == demo_row['class_id'])
    ax_g.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2.5 if is_demo else 1.1,
                               edgecolor='#FFD600' if is_demo else '#42A5F5',
                               facecolor='none',
                               alpha=1.0 if is_demo else 0.55,
                               zorder=5))

ax_g.set_xlim(0, IMG_W)
ax_g.set_ylim(IMG_H, 0)   # top-down (image convention)
ax_g.set_xticks(np.arange(0, IMG_W + 1, CELL * 5))
ax_g.set_xticklabels([str(int(i)) for i in np.arange(0, IMG_W + 1, CELL * 5)],
                      fontsize=6)
ax_g.set_yticks(np.arange(0, IMG_H + 1, CELL))
ax_g.set_yticklabels([str(int(i)) for i in np.arange(0, IMG_H + 1, CELL)],
                      fontsize=6.5)
ax_g.set_xlabel('X position (px)', fontsize=7.5)
ax_g.set_ylabel('Y position (px)', fontsize=7.5)

sm = plt.cm.ScalarMappable(cmap=cmap_sg, norm=norm_sg)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax_g, fraction=0.035, pad=0.04)
cb.set_label('Suitability score', fontsize=7)
cb.ax.tick_params(labelsize=6)
cb.ax.axhline(THRESH, color='white', linewidth=2.0, linestyle='--')
cb.ax.text(0.5, THRESH + 0.05, f'≥ {THRESH}',
           ha='center', va='bottom', fontsize=6.5, color='white',
           fontweight='bold', transform=cb.ax.transAxes)

# ─── Multi-channel hint channels ────────────────────────────────
gs2 = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs1[1, 1], wspace=0.20)

ch_data   = [roi_gray,  ch_R,                       ch_G,                    ch_B]
ch_cmaps  = ['gray',    'Reds',                      'Greens',                'Blues']
ch_titles = ['ROI Patch\n(original)',
             'Ch. R\nDefect Mask',
             'Ch. G\nCanny Edges',
             'Ch. B\nRoughness']
ch_colors = ['#455A64', '#C62828',                   '#2E7D32',               '#1565C0']
ch_descs  = [
    f'Class {int(demo_row["class_id"])}\n'
    f'{demo_row["defect_subtype"].replace("_"," ")}\n'
    f'BG: {demo_row["background_type"].replace("_"," ")}',
    'Dilated RLE mask\n→ defect shape\n& boundary',
    'Sobel gradient\n→ surface pattern\n& structure',
    'Local std. map\n→ texture\nroughness',
]

axes_ch = [fig.add_subplot(gs2[j]) for j in range(4)]
for j, ax in enumerate(axes_ch):
    ax.imshow(ch_data[j], cmap=ch_cmaps[j], aspect='auto', vmin=0, vmax=1)
    ax.set_title(ch_titles[j], fontsize=7.5, fontweight='bold',
                 color=ch_colors[j], pad=4)
    ax.set_xticks([]);  ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(2.2);  sp.set_edgecolor(ch_colors[j])
    ax.text(0.5, -0.20, ch_descs[j],
            ha='center', va='top', fontsize=6.5, color='#37474F',
            transform=ax.transAxes, multialignment='center')

axes_ch[0].annotate(
    '', xy=(1.12, 0.5), xytext=(1.02, 0.5),
    xycoords='axes fraction', textcoords='axes fraction',
    arrowprops=dict(arrowstyle='->', color='#455A64', lw=2.2),
    annotation_clip=False)

# ── PANEL 3: Summary ─────────────────────────────────────────────
ax_s = fig.add_subplot(gs0[2])
ax_s.axis('off')

prompt_display = demo_row['prompt']
if len(prompt_display) > 90:
    prompt_display = prompt_display[:88] + '...'

ax_s.text(0.5, 0.68,
          f'Pipeline output: 3,247 ROI patches  │  '
          f'Suitable (≥0.75): {n_suit}  │  Acceptable (0.5–0.75): {n_acc}  │  '
          f'Rejected (<0.50): filtered  │  '
          f'Prompt: "{prompt_display}"',
          ha='center', va='center', fontsize=8.0, color='white',
          transform=ax_s.transAxes,
          bbox=dict(boxstyle='round,pad=0.50',
                    facecolor='#1565C0', edgecolor='#0D47A1', linewidth=1.5))

ax_s.text(0.5, 0.15,
          'Hint image composition:  '
          'R = defect mask geometry (dilated RLE)  │  '
          'G = background structure (Canny/Sobel edges)  │  '
          'B = surface roughness (local std. texture map)  →  train.jsonl',
          ha='center', va='center', fontsize=8.0, color='#263238',
          transform=ax_s.transAxes,
          bbox=dict(boxstyle='round,pad=0.40',
                    facecolor='#ECEFF1', edgecolor='#90A4AE', linewidth=1.2))

plt.suptitle('CASDA Stage A: ROI Extraction & Multi-Channel Hint Generation',
             fontsize=13, fontweight='bold', color='#1A237E', y=0.975)

# ════════════════════════════════════════════════════════════════
# SAVE — combined then individual panels
# ════════════════════════════════════════════════════════════════
fig.savefig('casda_roi_extraction.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'Saved: casda_roi_extraction.png  ({len(df_img)} ROIs for {DEMO_IMAGE_ID})')

save_panel(fig, [ax_i],
           'casda_panel1_overview.png')
save_panel(fig, [ax_g_hdr, ax_g, cb.ax],
           'casda_panel2_scoregrid.png')
save_panel(fig, [ax_h_hdr] + axes_ch,
           'casda_panel2_hints.png')
save_panel(fig, [ax_s],
           'casda_panel3_summary.png')

plt.show()
print('Done.')
