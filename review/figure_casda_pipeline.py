"""
CASDA Framework Pipeline Flowchart
Style reference: multi-level hierarchical diagram (Image #2) + two-panel layout (Image #3)
Run in Google Colab: outputs casda_pipeline_flowchart.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis('off')

# ── Color Palette ──────────────────────────────────────────────
C_STAGE_A  = '#1565C0'   # dark blue  – Stage A label
C_STAGE_B  = '#6A1B9A'   # purple     – Stage B label
C_STAGE_C  = '#2E7D32'   # dark green – Stage C label
C_STAGE_D  = '#C62828'   # dark red   – Stage D label
C_BOX_HDR  = '#E3F2FD'   # light blue box header bg
C_BOX_BODY = '#FAFAFA'   # box body bg
C_ARROW    = '#455A64'   # arrow color
C_BORDER   = '#B0BEC5'   # box border
C_ABLATION = '#FFF8E1'   # ablation branch bg
C_EVAL     = '#FCE4EC'   # evaluation box bg

def stage_label(ax, y, label, color):
    """Left-side stage label (parallelogram style)."""
    x0, w, h = 0.15, 1.7, 0.5
    pts = [[x0, y], [x0+w, y], [x0+w-0.2, y+h], [x0-0.2, y+h]]
    poly = plt.Polygon(pts, closed=True, facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(poly)
    ax.text(x0 + w/2 - 0.1, y + h/2, label,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', zorder=4)

def box(ax, x, y, w, h, title, bullets, bg=C_BOX_BODY, hdr=C_BOX_HDR,
        border=C_BORDER, title_color='#0D47A1', fs=8.2):
    """Rounded box with header title + bullet body."""
    # Body
    body = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.04',
                          facecolor=bg, edgecolor=border, linewidth=1.2, zorder=2)
    ax.add_patch(body)
    # Header band
    hdr_h = 0.38
    hdr_patch = FancyBboxPatch((x, y + h - hdr_h), w, hdr_h,
                               boxstyle='round,pad=0.04',
                               facecolor=hdr, edgecolor=border, linewidth=0, zorder=3)
    ax.add_patch(hdr_patch)
    ax.text(x + w/2, y + h - hdr_h/2, title,
            ha='center', va='center', fontsize=fs, fontweight='bold',
            color=title_color, zorder=4)
    # Bullets
    line_h = (h - hdr_h - 0.12) / max(len(bullets), 1)
    for i, b in enumerate(bullets):
        ay = y + h - hdr_h - 0.12 - (i + 0.6) * line_h
        ax.text(x + 0.15, ay, f'• {b}', ha='left', va='center',
                fontsize=fs - 0.8, color='#212121', zorder=4)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=1.8, connectionstyle='arc3,rad=0.0'))

def dashed_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#78909C',
                                lw=1.2, linestyle='dashed',
                                connectionstyle='arc3,rad=0.0'))


# ════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════
ax.text(7, 17.6, 'CASDA Framework Pipeline',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#1A237E')
ax.text(7, 17.2, 'Context-Aware Steel Defect Augmentation',
        ha='center', va='center', fontsize=10, color='#546E7A')

# ════════════════════════════════════════════════════════════════
# INPUT
# ════════════════════════════════════════════════════════════════
box(ax, 3.5, 15.9, 7.0, 1.0,
    'INPUT: Severstal Steel Defect Dataset',
    ['Kaggle Severstal: 1600×256 images, train.csv (RLE mask labels)',
     'Class 1–4 defects  |  Severe class imbalance (Class 2: 247 samples)'],
    bg='#E8EAF6', hdr='#3949AB', title_color='white')

arrow(ax, 7, 15.9, 7, 15.3)

# ════════════════════════════════════════════════════════════════
# STAGE A
# ════════════════════════════════════════════════════════════════
stage_label(ax, 13.4, 'Stage A\n(CPU)', C_STAGE_A)

box(ax, 2.0, 13.55, 4.5, 1.6,
    'Step 1 · ROI Extraction',
    ['Sliding window (grid=64px) on defect regions',
     'ROI size: 256×256 px',
     'Suitability score ≥ 0.5 filter',
     'Output: 3,247 ROI patches'],
    hdr='#BBDEFB', title_color=C_STAGE_A)

box(ax, 7.5, 13.55, 4.5, 1.6,
    'Step 2 · ControlNet Data Prep',
    ['Multi-channel hint image (3-ch):',
     '  R = Defect mask geometry',
     '  G = Background structure (Canny)',
     '  B = Surface texture (roughness)',
     'Hybrid text prompt generation → train.jsonl'],
    hdr='#BBDEFB', title_color=C_STAGE_A)

arrow(ax, 7, 15.9,  4.25, 15.15)
arrow(ax, 7, 15.9,  9.75, 15.15)
arrow(ax, 4.25, 13.55, 4.25, 13.1)
arrow(ax, 9.75, 13.55, 9.75, 13.1)

# merge arrow
ax.annotate('', xy=(7, 12.7), xytext=(4.25, 13.1),
            arrowprops=dict(arrowstyle='-', color=C_ARROW, lw=1.8))
ax.annotate('', xy=(7, 12.7), xytext=(9.75, 13.1),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.8))

# ════════════════════════════════════════════════════════════════
# STAGE B
# ════════════════════════════════════════════════════════════════
stage_label(ax, 10.5, 'Stage B\n(GPU)', C_STAGE_B)

box(ax, 2.0, 11.55, 10.0, 1.0,
    'Step 3 · ControlNet Fine-Tuning',
    ['Base: Stable Diffusion v1.5 + sd-controlnet-canny',
     'lr=1e-5, cosine scheduler, fp16, epochs≤20, early-stopping patience=20, SNR γ=5.0'],
    hdr='#E1BEE7', title_color=C_STAGE_B)

arrow(ax, 7, 12.7, 7, 12.55)
arrow(ax, 7, 11.55, 7, 11.1)

box(ax, 2.0, 10.1, 4.8, 0.95,
    'Step 4 · Validation (optional)',
    ['Phase 1–4: generation quality,',
     'hint fidelity, class diversity,',
     'full-pipeline integration check'],
    hdr='#E1BEE7', title_color=C_STAGE_B, fs=7.8)

box(ax, 7.2, 10.1, 4.8, 0.95,
    'Step 5 · Synthetic Image Generation',
    ['Inference: 30 steps, CFG=7.5, scale=0.7',
     'Per-class qty: {1:2, 2:10, 3:1, 4:2}',
     'Output: 512×512 synthetic ROI patches'],
    hdr='#E1BEE7', title_color=C_STAGE_B, fs=7.8)

arrow(ax, 7, 11.55, 4.4, 11.05)
arrow(ax, 7, 11.55, 9.6, 11.05)
arrow(ax, 9.6, 10.1, 9.6,  9.65)
arrow(ax, 4.4, 10.1, 4.4,  9.65)
ax.annotate('', xy=(7, 9.3), xytext=(4.4, 9.65),
            arrowprops=dict(arrowstyle='-', color=C_ARROW, lw=1.8))
ax.annotate('', xy=(7, 9.3), xytext=(9.6, 9.65),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.8))

# ════════════════════════════════════════════════════════════════
# STAGE C
# ════════════════════════════════════════════════════════════════
stage_label(ax, 7.2, 'Stage C\n(CPU)', C_STAGE_C)

box(ax, 2.0, 8.3, 10.0, 0.95,
    'Step 6 · Poisson Blending Composition',
    ['Composite 512×512 synthetic ROI → 1600×256 clean background',
     'Context-aware bg selection (compatibility matrix)  |  5 compositions per ROI'],
    hdr='#C8E6C9', title_color=C_STAGE_C)

arrow(ax, 7, 9.3, 7, 9.25)
arrow(ax, 7, 8.3, 7, 7.95)

# Quality scoring
box(ax, 2.0, 6.95, 4.8, 0.95,
    'Step 7–8 · Ablation Variants',
    ['Step 7: CopyPaste baseline',
     '  (raw ROI, no ControlNet, no blend)',
     'Step 8: w/o Blending ablation',
     '  (ControlNet ROI, direct paste)'],
    bg=C_ABLATION, hdr='#FFE082', title_color='#E65100', fs=7.8)

box(ax, 7.2, 6.95, 4.8, 0.95,
    'Step 9–10 · Quality Filtering',
    ['Quality score Q = f(color consistency,',
     '  artifact detection, sharpness)',
     'Pruning: Q ≥ 0.7  (stratified top-k)',
     'Accepted: ~92.8% of composed samples'],
    hdr='#C8E6C9', title_color=C_STAGE_C, fs=7.8)

arrow(ax, 7, 7.95, 4.4, 7.9)
arrow(ax, 7, 7.95, 9.6, 7.9)
arrow(ax, 4.4, 6.95, 4.4, 6.55)
arrow(ax, 9.6, 6.95, 9.6, 6.55)
ax.annotate('', xy=(7, 6.2), xytext=(4.4, 6.55),
            arrowprops=dict(arrowstyle='-', color=C_ARROW, lw=1.8))
ax.annotate('', xy=(7, 6.2), xytext=(9.6, 6.55),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.8))

# ════════════════════════════════════════════════════════════════
# AUGMENTED DATASET
# ════════════════════════════════════════════════════════════════
box(ax, 3.5, 5.25, 7.0, 0.9,
    'Augmented Dataset Construction',
    ['Original: 5,237 samples  →  CASDA added: 2,238  →  Total: 7,475 (+42.7%)',
     'Class 2: +110.1%  |  Class 4: +70.0%  |  Class 1: +30.3%  |  Class 3: +27.3%'],
    bg='#E3F2FD', hdr='#1565C0', title_color='white')

arrow(ax, 7, 6.2, 7, 6.15)
arrow(ax, 7, 5.25, 7, 4.85)

# ════════════════════════════════════════════════════════════════
# STAGE D
# ════════════════════════════════════════════════════════════════
stage_label(ax, 2.9, 'Stage D\n(GPU)', C_STAGE_D)

# Three evaluation boxes
box(ax, 1.2, 3.3, 3.5, 1.5,
    'FID Evaluation',
    ['ROI-level FID',
     'Full-image FID',
     'Per-class & subtype',
     'CASDA vs. CopyPaste vs. Raw'],
    hdr='#FFCDD2', title_color=C_STAGE_D, fs=7.8)

box(ax, 5.25, 3.3, 3.5, 1.5,
    'Benchmark: 3 Models × 7 Groups',
    ['YOLO-MFD  (mAP@0.5)',
     'EB-YOLOv8 (mAP@0.5)',
     'DeepLabV3+(Dice score)',
     '7 dataset groups (ablation)'],
    hdr='#FFCDD2', title_color=C_STAGE_D, fs=7.8)

box(ax, 9.3, 3.3, 3.5, 1.5,
    'Statistical Testing',
    ['Wilcoxon signed-rank test',
     'Bootstrap CI (95%)',
     'Effect size (Cohen\'s d)',
     'Multi-model significance'],
    hdr='#FFCDD2', title_color=C_STAGE_D, fs=7.8)

arrow(ax, 7, 4.85, 2.95, 4.8)
arrow(ax, 7, 4.85, 7.0,  4.8)
arrow(ax, 7, 4.85, 11.05, 4.8)
arrow(ax, 2.95, 4.8, 2.95, 4.8)
arrow(ax, 2.95, 3.3,  2.95, 3.0)
arrow(ax, 7.0,  3.3,  7.0,  3.0)
arrow(ax, 11.05,3.3,  11.05,3.0)

# ════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════
ax.annotate('', xy=(7, 2.65), xytext=(2.95, 3.0),
            arrowprops=dict(arrowstyle='-', color=C_ARROW, lw=1.8))
ax.annotate('', xy=(7, 2.65), xytext=(11.05, 3.0),
            arrowprops=dict(arrowstyle='-', color=C_ARROW, lw=1.8))
ax.annotate('', xy=(7, 2.65), xytext=(7.0, 3.0),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.8))

box(ax, 3.0, 1.6, 8.0, 1.0,
    'OUTPUT: CASDA Augmentation Effect Verification',
    ['YOLO-MFD:  mAP +2.89 pp vs. raw  |  +5.78 pp vs. CopyPaste  (Class 2: +7.92 pp)',
     'EB-YOLOv8: marginal gain  |  DeepLabV3+: competitive  |  Statistically validated'],
    bg='#E8F5E9', hdr='#2E7D32', title_color='white')

# ════════════════════════════════════════════════════════════════
# CPU / GPU badges
# ════════════════════════════════════════════════════════════════
for y_pos, label, color in [(14.4,'CPU','#1565C0'), (11.2,'GPU','#6A1B9A'),
                             (8.0,'CPU','#2E7D32'),  (4.6,'GPU','#C62828')]:
    ax.text(13.5, y_pos, label, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='none'))

plt.tight_layout()
plt.savefig('casda_pipeline_flowchart.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("Saved: casda_pipeline_flowchart.png")
