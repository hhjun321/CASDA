"""
CASDA Framework Pipeline — 5-Stage Overview  (§3.2)
Revision: Stage A–D → Stage 1–5 per revised paper structure

  3.2.1  Stage 1: Geometric ROI Characterization and Suitability Evaluation
  3.2.2  Stage 2: Create ControlNet Configuration
  3.2.3  Stage 3: Context-Based Defect Generation and Synthesis
  3.2.4  Stage 4: Quality Verification Gate
  3.2.5  Stage 5: Dataset Integration

Run in Google Colab: outputs casda_pipeline_flowchart.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis('off')

# ── Color palette ────────────────────────────────────────────────
C = {1: '#1565C0', 2: '#283593', 3: '#6A1B9A', 4: '#00695C', 5: '#2E7D32'}
H = {1: '#BBDEFB', 2: '#C5CAE9', 3: '#E1BEE7', 4: '#B2DFDB', 5: '#C8E6C9'}
C_ARROW  = '#455A64'
C_BORDER = '#B0BEC5'


def stage_label(ax, y, label, color):
    x0, w, h = 0.15, 1.7, 0.5
    pts = [[x0, y], [x0+w, y], [x0+w-0.2, y+h], [x0-0.2, y+h]]
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor=color,
                             edgecolor='white', linewidth=1.5, zorder=3))
    ax.text(x0+w/2-0.1, y+h/2, label, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='white', zorder=4)


def box(ax, x, y, w, h, title, bullets, bg='#FAFAFA', hdr='#E3F2FD',
        border=C_BORDER, title_color='#0D47A1', fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.04',
                                facecolor=bg, edgecolor=border,
                                linewidth=1.2, zorder=2))
    hdr_h = 0.38
    ax.add_patch(FancyBboxPatch((x, y+h-hdr_h), w, hdr_h,
                                boxstyle='round,pad=0.04',
                                facecolor=hdr, edgecolor=border,
                                linewidth=0, zorder=3))
    ax.text(x+w/2, y+h-hdr_h/2, title, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=title_color, zorder=4)
    line_h = (h - hdr_h - 0.12) / max(len(bullets), 1)
    for i, b in enumerate(bullets):
        ay = y + h - hdr_h - 0.12 - (i + 0.6) * line_h
        ax.text(x+0.15, ay, f'• {b}', ha='left', va='center',
                fontsize=fs-0.8, color='#212121', zorder=4)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=1.8, connectionstyle='arc3,rad=0.0'))


# ════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════
ax.text(7, 17.6, 'CASDA Framework Pipeline',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#1A237E')
ax.text(7, 17.2, 'Context-Aware Steel Defect Augmentation  (§3.2)',
        ha='center', va='center', fontsize=10, color='#546E7A')

# ════════════════════════════════════════════════════════════════
# INPUT
# ════════════════════════════════════════════════════════════════
box(ax, 2.5, 15.9, 9.0, 1.0,
    'INPUT: Severstal Steel Defect Dataset',
    ['Kaggle Severstal: 1,600×256 px images, train.csv (RLE mask labels)',
     'Class 1–4 defects  |  Severe class imbalance (Class 2: 247 samples)'],
    bg='#E8EAF6', hdr='#3949AB', title_color='white')

arrow(ax, 7, 15.9, 7, 14.8)

# ════════════════════════════════════════════════════════════════
# STAGE 1
# ════════════════════════════════════════════════════════════════
stage_label(ax, 13.65, 'Stage 1\n§3.2.1', C[1])

box(ax, 2.0, 13.0, 10.0, 1.8,
    '3.2.1  Geometric ROI Characterization and Suitability Evaluation',
    ['Sliding window scan (64 px grid) across defect mask regions in 1,600×256 px images',
     'ROI window: 256×256 px  |  Rule-based defect type classification  (linear / irregular / blob / general)',
     'Suitability score S ≥ 0.5 filter  →  Output: 3,247 ROI patches'],
    hdr=H[1], title_color=C[1])

arrow(ax, 7, 13.0, 7, 12.0)

# ════════════════════════════════════════════════════════════════
# STAGE 2
# ════════════════════════════════════════════════════════════════
stage_label(ax, 10.85, 'Stage 2\n§3.2.2', C[2])

box(ax, 2.0, 10.2, 10.0, 1.8,
    '3.2.2  Create ControlNet Configuration',
    ['Multi-channel hint image (3-ch):  R = defect mask geometry  |  G = background structure (Canny edges)',
     'B = surface texture (roughness map)  →  encodes geometric, structural, and textural context',
     'Hybrid text prompt generation  →  train.jsonl  (subtype + surface + quality vocabulary)'],
    hdr=H[2], title_color=C[2])

arrow(ax, 7, 10.2, 7, 9.2)

# ════════════════════════════════════════════════════════════════
# STAGE 3
# ════════════════════════════════════════════════════════════════
stage_label(ax, 7.85, 'Stage 3\n§3.2.3', C[3])

box(ax, 2.0, 7.0, 10.0, 2.2,
    '3.2.3  Context-Based Defect Generation and Synthesis',
    ['ControlNet fine-tuning on (hint, ROI) pairs  (SD v1.5, lr=1e-5, cosine scheduler, fp16)',
     'Inference: 30 steps, CFG=7.5, scale=0.7  →  512×512 synthetic ROI patches',
     'Per-class generation ratio:  {Class 1: ×2,  Class 2: ×10,  Class 3: ×1,  Class 4: ×2}',
     'Poisson Blending Composition:  synthetic ROI → 1,600×256 clean background'],
    hdr=H[3], title_color=C[3])

arrow(ax, 7, 7.0, 7, 6.0)

# ════════════════════════════════════════════════════════════════
# STAGE 4
# ════════════════════════════════════════════════════════════════
stage_label(ax, 4.85, 'Stage 4\n§3.2.4', C[4])

box(ax, 2.0, 4.2, 10.0, 1.8,
    '3.2.4  Quality Verification Gate',
    ['Quality score  Q = f(color consistency, artifact detection, sharpness)',
     'Pruning threshold: Q ≥ 0.7  (stratified top-k per defect class)',
     'Accepted: 2,238 samples  (acceptance rate: 92.8%)'],
    hdr=H[4], title_color=C[4])

arrow(ax, 7, 4.2, 7, 3.2)

# ════════════════════════════════════════════════════════════════
# STAGE 5
# ════════════════════════════════════════════════════════════════
stage_label(ax, 2.05, 'Stage 5\n§3.2.5', C[5])

box(ax, 2.0, 1.4, 10.0, 1.8,
    '3.2.5  Dataset Integration',
    ['Original dataset:  5,237 samples',
     'CASDA augmented:  +2,238 samples  (Class 2: +110.1%  |  Class 4: +70.0%  |  Class 1: +30.3%  |  Class 3: +27.3%)',
     'Total:  7,475 samples  (+42.7% overall)'],
    bg='#E8F5E9', hdr='#2E7D32', title_color='white')

# ════════════════════════════════════════════════════════════════
# CPU / GPU badges (right side)
# ════════════════════════════════════════════════════════════════
for y_pos, label, color in [
    (13.9, 'CPU', C[1]),
    (11.1, 'CPU', C[2]),
    (8.1,  'GPU', C[3]),
    (5.1,  'CPU', C[4]),
    (2.3,  'CPU', C[5]),
]:
    ax.text(13.2, y_pos, label, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='none'))

plt.tight_layout()
plt.savefig('casda_pipeline_flowchart.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("Saved: casda_pipeline_flowchart.png")
