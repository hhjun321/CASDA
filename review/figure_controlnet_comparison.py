"""
CASDA ControlNet: Class-by-Class Generation Comparison Figure
Loads comparison images from test_controlnet.py → comparisons/
Each file is a 4-panel grid: [Hint | Generated #1 | Generated #2 | Original]

Run in Google Colab → casda_controlnet_comparison.png
                       casda_comparison_class{N}.png  (per-class panels)

Adjust N_SAMPLES and COMPARISONS_DIR as needed.
"""

import os
import re
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
from PIL import Image

# ════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════
DRIVE           = '/content/drive/MyDrive/data/Severstal'
AUG_IMAGES_DIR  = f'{DRIVE}/augmented_images'
COMPARISONS_DIR = f'{AUG_IMAGES_DIR}/comparisons'
SUMMARY_JSON    = f'{AUG_IMAGES_DIR}/generation_summary.json'

N_SAMPLES     = 3          # samples to display per class
CLASSES       = [1, 2, 3, 4]

# Defect class descriptions (Severstal dataset)
CLASS_META = {
    1: {'name': 'Class 1', 'desc': 'Scratches\n(elongated linear)', 'color': '#C62828'},
    2: {'name': 'Class 2', 'desc': 'Patches\n(compact blob)',        'color': '#6A1B9A'},
    3: {'name': 'Class 3', 'desc': 'Crazing\n(irregular network)',   'color': '#1565C0'},
    4: {'name': 'Class 4', 'desc': 'Inclusions\n(scattered spots)',  'color': '#2E7D32'},
}

# Column layout: order in comparison image (Hint | Gen1 | Gen2 | Original)
COL_LABELS  = ['Hint\n(ControlNet Input)', 'Generated  #1', 'Generated  #2', 'Original ROI']
COL_COLORS  = ['#455A64',                  '#1565C0',        '#1565C0',       '#E65100']
COL_BG      = ['#ECEFF1',                  '#E3F2FD',        '#E3F2FD',       '#FFF3E0']

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def parse_class_from_name(fname):
    """Extract class number from comparison filename."""
    m = re.search(r'class[_\-]?(\d)', os.path.basename(fname), re.IGNORECASE)
    return int(m.group(1)) if m else None

def load_and_split(fpath, n_panels=4):
    """Load comparison grid image and split into n_panels equal horizontal slices."""
    img = np.array(Image.open(fpath).convert('RGB'))
    h, w = img.shape[:2]
    pw = w // n_panels
    return [img[:, i * pw:(i + 1) * pw, :] for i in range(n_panels)]

def save_panel(fig, axes_list, filename, pad=0.12, dpi=180):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [ax.get_tightbbox(renderer) for ax in axes_list
              if ax.get_tightbbox(renderer) is not None]
    if not bboxes:
        return
    union_bb = Bbox.union(bboxes)
    bb_in = union_bb.transformed(fig.dpi_scale_trans.inverted())
    padded = Bbox([[bb_in.x0 - pad, bb_in.y0 - pad],
                   [bb_in.x1 + pad, bb_in.y1 + pad]])
    fig.savefig(filename, dpi=dpi, bbox_inches=padded,
                facecolor='white', edgecolor='none')
    print(f'Saved: {filename}')

# ════════════════════════════════════════════════════════════════
# LOAD & GROUP COMPARISON FILES
# ════════════════════════════════════════════════════════════════
all_files = sorted(glob.glob(os.path.join(COMPARISONS_DIR, '*.png')))
print(f'Found {len(all_files)} comparison images in:\n  {COMPARISONS_DIR}')

by_class = {c: [] for c in CLASSES}
unmatched = []
for f in all_files:
    cls = parse_class_from_name(f)
    if cls in by_class:
        by_class[cls].append(f)
    else:
        unmatched.append(f)

for cls in CLASSES:
    print(f'  Class {cls}: {len(by_class[cls])} images')
if unmatched:
    print(f'  Unmatched (no class in filename): {len(unmatched)}')

# Optional: re-sort by quality score from generation_summary.json
if os.path.exists(SUMMARY_JSON):
    with open(SUMMARY_JSON) as fh:
        summary = json.load(fh)
    print(f'Loaded {SUMMARY_JSON}: {len(summary)} entries')
    # If summary is a dict keyed by filename with quality scores, sort here.
    # Adjust key/field names to match actual JSON structure as needed.
    try:
        score_map = {os.path.basename(e['comparison_path']): e.get('quality_score', 0)
                     for e in summary if 'comparison_path' in e}
        for cls in CLASSES:
            by_class[cls].sort(
                key=lambda f: score_map.get(os.path.basename(f), 0),
                reverse=True)
        print('  Re-sorted by quality_score (desc)')
    except Exception as ex:
        print(f'  Could not sort by quality score: {ex}')

# Select N_SAMPLES per class (evenly spaced when more than N_SAMPLES available)
selected = {}
for cls in CLASSES:
    files = by_class[cls]
    n = len(files)
    if n == 0:
        selected[cls] = []
    elif n <= N_SAMPLES:
        selected[cls] = files
    else:
        # Top-N already sorted by quality; take evenly spaced for visual diversity
        indices = np.linspace(0, min(n, N_SAMPLES * 4) - 1, N_SAMPLES, dtype=int)
        selected[cls] = [files[int(i)] for i in indices]

# ════════════════════════════════════════════════════════════════
# FIGURE LAYOUT
# Rows: 1 header + (4 classes × N_SAMPLES image rows) + class separators
# Cols: label (spans N_SAMPLES) + 4 panels
# ════════════════════════════════════════════════════════════════
N_CLS     = len(CLASSES)
N_IMGROWS = N_CLS * N_SAMPLES

# Height ratios: header=0.4, each image row=1.0, inter-class gap=0.08
row_heights = [0.4]
for ci in range(N_CLS):
    row_heights.extend([1.0] * N_SAMPLES)
    if ci < N_CLS - 1:
        row_heights.append(0.08)   # thin gap between class groups

N_ROWS  = len(row_heights)
FIG_W   = 18
FIG_H   = sum(row_heights) * 1.9   # ~1.9 inches per unit

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs  = GridSpec(N_ROWS, 5,
               figure=fig,
               height_ratios=row_heights,
               width_ratios=[0.60, 1.0, 1.0, 1.0, 1.0],
               hspace=0.035, wspace=0.025,
               top=0.945, bottom=0.025, left=0.025, right=0.985)

# ── Header row (row 0) ───────────────────────────────────────────
ax_hblank = fig.add_subplot(gs[0, 0]);  ax_hblank.axis('off')

for j, (lbl, col, bg) in enumerate(zip(COL_LABELS, COL_COLORS, COL_BG)):
    ax_h = fig.add_subplot(gs[0, j + 1])
    ax_h.set_facecolor(bg)
    for sp in ax_h.spines.values():
        sp.set_linewidth(1.6);  sp.set_edgecolor(col)
    ax_h.set_xticks([]);  ax_h.set_yticks([])
    ax_h.text(0.5, 0.48, lbl,
              ha='center', va='center', fontsize=9.5, fontweight='bold',
              color=col, transform=ax_h.transAxes, multialignment='center')

# Track axes per class for per-class saving
class_axes_map = {cls: [] for cls in CLASSES}

# ── Image rows ───────────────────────────────────────────────────
def row_index_of(ci, si):
    """Convert (class_idx, sample_idx) to GridSpec row index."""
    # row 0 = header
    # Each class group: N_SAMPLES image rows + 1 gap row (except last)
    idx = 1
    for k in range(ci):
        idx += N_SAMPLES + 1   # N_SAMPLES images + 1 separator
    idx += si
    return idx

for ci, cls in enumerate(CLASSES):
    meta  = CLASS_META[cls]
    color = meta['color']
    files = selected[cls]

    # Class label spanning N_SAMPLES rows
    r0  = row_index_of(ci, 0)
    r1  = row_index_of(ci, N_SAMPLES - 1) + 1
    ax_cls = fig.add_subplot(gs[r0:r1, 0])
    ax_cls.axis('off')
    ax_cls.set_facecolor('#FAFAFA')
    # Colored left border
    ax_cls.axvline(0.92, color=color, linewidth=5, alpha=0.6,
                   clip_on=False)
    ax_cls.text(0.42, 0.62,
                meta['name'],
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=color, transform=ax_cls.transAxes)
    ax_cls.text(0.42, 0.38,
                meta['desc'],
                ha='center', va='center', fontsize=7.5,
                color='#455A64', transform=ax_cls.transAxes,
                multialignment='center')
    # Sample count annotation
    n_avail = len(by_class[cls])
    ax_cls.text(0.42, 0.12,
                f'n={n_avail}',
                ha='center', va='center', fontsize=7,
                color='#78909C', transform=ax_cls.transAxes)
    class_axes_map[cls].append(ax_cls)

    for si in range(N_SAMPLES):
        row_idx = row_index_of(ci, si)

        if si < len(files):
            try:
                panels = load_and_split(files[si])
            except Exception as e:
                panels = [np.ones((128, 128, 3), dtype=np.uint8) * 200] * 4
                print(f'Warning: could not load {files[si]}: {e}')
        else:
            panels = None

        for j in range(4):
            ax = fig.add_subplot(gs[row_idx, j + 1])
            class_axes_map[cls].append(ax)

            if panels is not None:
                ax.imshow(panels[j], aspect='auto', interpolation='bilinear')
                ax.set_xticks([]);  ax.set_yticks([])

                # Spine style: generated panels get class color, original gets orange
                if j in (1, 2):   # Generated #1, #2
                    ec, lw = color, 2.0
                elif j == 3:      # Original
                    ec, lw = '#E65100', 1.5
                else:             # Hint
                    ec, lw = '#607D8B', 1.2
                for sp in ax.spines.values():
                    sp.set_linewidth(lw);  sp.set_edgecolor(ec)

                # Sample index tag on hint panel (j=0)
                if j == 0:
                    ax.text(0.03, 0.97, f'#{si + 1}',
                            ha='left', va='top', fontsize=7.5,
                            color='white', fontweight='bold',
                            transform=ax.transAxes,
                            bbox=dict(boxstyle='round,pad=0.18',
                                      facecolor='#37474F', alpha=0.80,
                                      edgecolor='none'))

                # Arrow overlay between hint and gen#1 (first sample only)
                if j == 0 and si == 0:
                    ax.annotate('',
                                xy=(1.035, 0.5), xytext=(1.005, 0.5),
                                xycoords='axes fraction',
                                textcoords='axes fraction',
                                arrowprops=dict(arrowstyle='-|>',
                                                color=color, lw=1.6,
                                                mutation_scale=10),
                                annotation_clip=False)

            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=9, color='#BDBDBD',
                        transform=ax.transAxes)
                ax.set_facecolor('#F5F5F5')

    # Separator axes between class groups
    if ci < N_CLS - 1:
        sep_row = row_index_of(ci, N_SAMPLES - 1) + 1
        ax_sep = fig.add_subplot(gs[sep_row, :])
        ax_sep.axis('off')
        ax_sep.axhline(0.5, color='#B0BEC5', linewidth=0.8, linestyle='--')

# ── Title & legend ───────────────────────────────────────────────
plt.suptitle(
    'CASDA ControlNet — Class-by-Class Generation Quality Comparison\n'
    'Columns left→right:  Hint (3-ch input)  │  Generated #1  │  Generated #2  │  Original ROI',
    fontsize=12, fontweight='bold', color='#1A237E', y=0.975)

legend_handles = [
    mpatches.Patch(facecolor=CLASS_META[c]['color'], alpha=0.85,
                   label=f"{CLASS_META[c]['name']}: {CLASS_META[c]['desc'].replace(chr(10), ' ')}")
    for c in CLASSES
]
legend_handles += [
    mpatches.Patch(facecolor='#607D8B', alpha=0.7, label='Hint (ControlNet conditioning)'),
    mpatches.Patch(facecolor='#E65100', alpha=0.7, label='Original ROI (reference)'),
]
fig.legend(handles=legend_handles, loc='lower center',
           ncol=3, fontsize=7.5, framealpha=0.90,
           bbox_to_anchor=(0.5, 0.002),
           columnspacing=1.2, handlelength=1.2)

# ════════════════════════════════════════════════════════════════
# SAVE — combined + per-class panels
# ════════════════════════════════════════════════════════════════
combined_out = 'casda_controlnet_comparison.png'
plt.savefig(combined_out, dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'Saved: {combined_out}')

for cls in CLASSES:
    save_panel(fig, class_axes_map[cls],
               f'casda_comparison_class{cls}.png')

plt.show()
print('Done.')
