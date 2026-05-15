"""
CASDA: Hybrid Text Prompt Construction Flowchart
Style: clean academic decision-tree (Image #5 reference)
  - black/white, rounded rectangles, diamond decisions
  - 4 semantic fields merged into fixed template
Run in Google Colab → casda_hybrid_prompt.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 20)
ax.axis('off')

# ── Primitives (Image #5 style: thin black border, white fill) ──

def rbox(ax, x, y, w, h, text, fs=8.4, bold=False,
         fc='white', ec='black', lw=1.1, tc='black'):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.1',
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal',
            color=tc, zorder=4, multialignment='center')

def diamond(ax, cx, cy, hw, hh, text, fs=8.2):
    pts = [[cx, cy+hh], [cx+hw, cy], [cx, cy-hh], [cx-hw, cy]]
    poly = plt.Polygon(pts, closed=True,
                       facecolor='white', edgecolor='black',
                       linewidth=1.1, zorder=3)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='black', zorder=4)

def arr(ax, x1, y1, x2, y2, lc='black'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=lc, lw=1.1,
                                connectionstyle='arc3,rad=0.0'))

def hline(ax, x1, x2, y, lc='black'):
    ax.plot([x1, x2], [y, y], color=lc, lw=1.1, zorder=2)

def vline(ax, x, y1, y2, lc='black'):
    ax.plot([x, x], [y1, y2], color=lc, lw=1.1, zorder=2)

def blabel(ax, x, y, text):
    """Branch label (italic, small)."""
    ax.text(x, y, text, ha='center', va='center',
            fontsize=7.2, fontstyle='italic', color='#333333', zorder=5)


# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
ax.text(7, 19.7, 'Hybrid Text Prompt Construction',
        ha='center', va='center', fontsize=13, fontweight='bold')
ax.text(7, 19.35, 'Four semantic fields concatenated in a fixed template for ControlNet conditioning',
        ha='center', va='center', fontsize=8.8, color='#444444')

# ══════════════════════════════════════════════════════════════
# TOP INPUT
# ══════════════════════════════════════════════════════════════
rbox(ax, 3.5, 18.5, 7.0, 0.65,
     'ROI Patch Analysis Outputs\n'
     'morphological metrics  ·  background type  ·  stability score  ·  class label')

arr(ax, 7.0, 18.5, 7.0, 18.25)

# ══════════════════════════════════════════════════════════════
# FIELD 1 — Defect Form
# ══════════════════════════════════════════════════════════════
# Section label
ax.text(0.25, 17.3, 'Field 1', ha='center', va='center', fontsize=7.5,
        fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#37474F', edgecolor='none'))
ax.text(0.25, 17.05, 'Defect\nForm', ha='center', va='center',
        fontsize=7, color='#37474F')

diamond(ax, 7.0, 17.75, 1.5, 0.5, 'Defect form?', fs=8.2)

# Five branches from diamond
xs = [1.3, 3.1, 5.5, 8.9, 12.0]
labels_d = ['linear\n scratch', 'compact\nblob', 'irregular', 'elongated', 'general']

for xi in xs:
    vline(ax, xi, 17.25, 17.75)

hline(ax, xs[0], xs[-1], 17.25)

branch_texts_d = [
    'λ>0.7',
    'σ>0.8',
    'default',
    'AR>3.0',
    'else'
]
label_x_offsets = [-0.35, -0.35, 0.0, 0.3, 0.3]
for i, (xi, bt) in enumerate(zip(xs, branch_texts_d)):
    blabel(ax, xi + label_x_offsets[i], 17.08, bt)

# Field 1 output boxes
field1_texts = [
    '"a high-linearity\nelongated scratch"',
    '"a solid compact\ndefect spot"',
    '"an irregular defect\nwith complex\nboundaries"',
    '"a moderately\nelongated\ndefect region"',
    '"a general\nsurface defect"'
]
box_ws = [2.2, 2.0, 2.0, 2.0, 1.8]
box_xs = [0.2, 2.1, 4.5, 7.95, 11.2]

for xi, bx, bw, bt in zip(xs, box_xs, box_ws, field1_texts):
    arr(ax, xi, 17.25, xi, 16.9)
    rbox(ax, bx, 15.8, bw, 1.05, bt, fs=7.5, fc='#FAFAFA')

# Merge Field 1
merge_y1 = 15.8
for bx, bw in zip(box_xs, box_ws):
    cx = bx + bw/2
    vline(ax, cx, merge_y1, 15.45)

hline(ax, box_xs[0] + box_ws[0]/2, box_xs[-1] + box_ws[-1]/2, 15.45)
arr(ax, 7.0, 15.45, 7.0, 15.2)

rbox(ax, 3.5, 14.6, 7.0, 0.55,
     'Field 1: Defect form description  →  [F1]',
     fs=8.2, bold=True, fc='#ECEFF1', ec='#455A64', tc='#263238')

arr(ax, 7.0, 14.6, 7.0, 14.2)

# ══════════════════════════════════════════════════════════════
# FIELD 2 — Background Connection
# ══════════════════════════════════════════════════════════════
ax.text(0.25, 13.7, 'Field 2', ha='center', va='center', fontsize=7.5,
        fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#37474F', edgecolor='none'))
ax.text(0.25, 13.45, 'Background\nConnection', ha='center', va='center',
        fontsize=7, color='#37474F')

diamond(ax, 7.0, 13.9, 1.7, 0.5, 'Background type?', fs=8.2)

bg_xs = [1.4, 3.35, 6.3, 9.25, 11.8]
bg_labels = ['smooth', 'vertical\nstripe', 'horizontal\nstripe', 'textured', 'complex\npattern']
bg_blabels = ['smooth', 'v. stripe', 'h. stripe', 'textured', 'complex']

for xi in bg_xs:
    vline(ax, xi, 13.4, 13.9)
hline(ax, bg_xs[0], bg_xs[-1], 13.4)

bg_label_x = [-0.3, -0.3, 0.0, 0.3, 0.35]
for i, (xi, bt) in enumerate(zip(bg_xs, bg_blabels)):
    blabel(ax, xi + bg_label_x[i], 13.23, bt)

bg_box_texts = [
    '"smooth metal\nsurface,\nuniform texture"',
    '"vertical striped\nmetal surface,\ndirectional texture"',
    '"horizontal striped\nmetal surface,\ndirectional texture"',
    '"textured metal\nsurface,\ngrainy texture"',
    '"complex patterned\nsurface,\nmulti-dir texture"'
]
bg_box_ws = [1.9, 2.0, 2.1, 1.9, 2.0]
bg_box_xs = [0.25, 2.3, 5.25, 8.5, 11.05]

for xi, bx, bw, bt in zip(bg_xs, bg_box_xs, bg_box_ws, bg_box_texts):
    arr(ax, xi, 13.4, xi, 13.05)
    rbox(ax, bx, 12.05, bw, 0.95, bt, fs=7.3, fc='#FAFAFA')

for bx, bw in zip(bg_box_xs, bg_box_ws):
    cx = bx + bw/2
    vline(ax, cx, 12.05, 11.75)

hline(ax, bg_box_xs[0]+bg_box_ws[0]/2, bg_box_xs[-1]+bg_box_ws[-1]/2, 11.75)
arr(ax, 7.0, 11.75, 7.0, 11.5)

rbox(ax, 3.5, 10.9, 7.0, 0.55,
     'Field 2: Background + texture description  →  [F2]',
     fs=8.2, bold=True, fc='#ECEFF1', ec='#455A64', tc='#263238')

arr(ax, 7.0, 10.9, 7.0, 10.5)

# ══════════════════════════════════════════════════════════════
# FIELD 3 — Surface Condition
# ══════════════════════════════════════════════════════════════
ax.text(0.25, 10.2, 'Field 3', ha='center', va='center', fontsize=7.5,
        fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#37474F', edgecolor='none'))
ax.text(0.25, 9.95, 'Part\nCondition', ha='center', va='center',
        fontsize=7, color='#37474F')

diamond(ax, 7.0, 10.2, 1.9, 0.5, 'Stability score?', fs=8.2)

sc_xs = [3.0, 7.0, 11.0]
sc_blabels = ['score ≥ 0.8', 'score ≥ 0.5', 'score < 0.5']
sc_label_offsets = [-0.8, 0.15, 0.9]

for xi in sc_xs:
    vline(ax, xi, 9.7, 10.2)
hline(ax, sc_xs[0], sc_xs[-1], 9.7)

for xi, bt, off in zip(sc_xs, sc_blabels, sc_label_offsets):
    blabel(ax, xi + off, 9.53, bt)

sc_box_texts = [
    'High quality\n"pristine /\nwell-maintained /\nclean"',
    'Medium quality\n"standard /\ntypical / normal"',
    'Low quality\n"worn /\nweathered / aged"'
]
sc_box_xs = [1.8, 5.5, 9.5]
sc_box_w = 2.4

for xi, bx, bt in zip(sc_xs, sc_box_xs, sc_box_texts):
    arr(ax, xi, 9.7, xi, 9.35)
    rbox(ax, bx, 8.4, sc_box_w, 0.9, bt, fs=7.5, fc='#FAFAFA')

for bx in sc_box_xs:
    cx = bx + sc_box_w/2
    vline(ax, cx, 8.4, 8.1)

hline(ax, sc_box_xs[0]+sc_box_w/2, sc_box_xs[-1]+sc_box_w/2, 8.1)
arr(ax, 7.0, 8.1, 7.0, 7.85)

rbox(ax, 3.5, 7.25, 7.0, 0.55,
     'Field 3: Surface condition modifier  →  [F3]',
     fs=8.2, bold=True, fc='#ECEFF1', ec='#455A64', tc='#263238')

arr(ax, 7.0, 7.25, 7.0, 6.9)

# ══════════════════════════════════════════════════════════════
# FIELD 4 — Class Identifier
# ══════════════════════════════════════════════════════════════
ax.text(0.25, 6.65, 'Field 4', ha='center', va='center', fontsize=7.5,
        fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#37474F', edgecolor='none'))
ax.text(0.25, 6.4, 'Surface\nState', ha='center', va='center',
        fontsize=7, color='#37474F')

diamond(ax, 7.0, 6.6, 1.6, 0.45, 'Class label?', fs=8.2)

cl_xs = [2.5, 5.2, 8.2, 11.5]
cl_blabels = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
cl_offsets = [-0.55, -0.3, 0.25, 0.6]

for xi in cl_xs:
    vline(ax, xi, 6.15, 6.6)
hline(ax, cl_xs[0], cl_xs[-1], 6.15)

for xi, bt, off in zip(cl_xs, cl_blabels, cl_offsets):
    blabel(ax, xi + off, 5.98, bt)

cl_texts = [
    '"Class 1\nsteel defect"',
    '"Class 2\nsteel defect"',
    '"Class 3\nsteel defect"',
    '"Class 4\nsteel defect"'
]
cl_box_w = 2.0
cl_box_xs = [1.4, 4.15, 7.15, 10.45]

for xi, bx, bt in zip(cl_xs, cl_box_xs, cl_texts):
    arr(ax, xi, 6.15, xi, 5.8)
    rbox(ax, bx, 5.2, cl_box_w, 0.55, bt, fs=7.5, fc='#FAFAFA')

for bx in cl_box_xs:
    cx = bx + cl_box_w/2
    vline(ax, cx, 5.2, 4.95)

hline(ax, cl_box_xs[0]+cl_box_w/2, cl_box_xs[-1]+cl_box_w/2, 4.95)
arr(ax, 7.0, 4.95, 7.0, 4.7)

rbox(ax, 3.5, 4.1, 7.0, 0.55,
     'Field 4: Class identifier  →  [F4]',
     fs=8.2, bold=True, fc='#ECEFF1', ec='#455A64', tc='#263238')

arr(ax, 7.0, 4.1, 7.0, 3.8)

# ══════════════════════════════════════════════════════════════
# TEMPLATE CONCATENATION
# ══════════════════════════════════════════════════════════════
rbox(ax, 2.0, 3.05, 10.0, 0.7,
     'Fixed Template:  [F1]  +  "on"  +  [F2]  +  "with texture"  +  ([F3] condition)  +  [F4]',
     fs=8.8, bold=True, fc='#E8EAF6', ec='#3949AB', tc='#1A237E', lw=1.5)

arr(ax, 7.0, 3.05, 7.0, 2.75)

# ══════════════════════════════════════════════════════════════
# EXAMPLE OUTPUT
# ══════════════════════════════════════════════════════════════
rbox(ax, 1.5, 1.85, 11.0, 0.85,
     'Example output:\n'
     '"A high-linearity elongated scratch on a vertical striped metal surface\n'
     ' with directional texture (pristine condition), Class 1 steel defect"',
     fs=8.3, fc='white', ec='black', lw=1.3)

arr(ax, 7.0, 1.85, 7.0, 1.55)

rbox(ax, 3.5, 0.75, 7.0, 0.75,
     'Hybrid Text Prompt  →  train.jsonl\n'
     'Paired with hint image for ControlNet conditioning',
     fs=8.5, bold=True, fc='#212121', ec='#212121', tc='white', lw=1.5)

plt.tight_layout()
plt.savefig('casda_hybrid_prompt.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("Saved: casda_hybrid_prompt.png")
