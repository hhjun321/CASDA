"""
figure_suitability_heatmap.py  —  Stage A §5: ROI Suitability Matching Matrix
==============================================================================
논문 §5 "결함-배경 적합도 매칭 행렬" 시각화.

Layout:
  Left  : Matching matrix heatmap (defect_subtype × background_type)
  Right : Intuition diagram — why each pairing makes visual sense

Run in Google Colab:
    !python /content/CASDA/review/figure_suitability_heatmap.py
"""

import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "suitability_heatmap.png"

# ──────────────────────────────────────────────────────────────────────────────
# 매칭 행렬 데이터  (src/analysis/roi_suitability.py:MATCHING_RULES 에서 추출)
# ──────────────────────────────────────────────────────────────────────────────
# rows: defect subtypes,  cols: background types
DEFECT_SUBTYPES = [
    "linear_scratch",
    "elongated",
    "compact_blob",
    "irregular",
    "general",
]
BG_TYPES = [
    "smooth",
    "vertical_stripe",
    "horizontal_stripe",
    "textured",
    "complex_pattern",
]

# shape (5, 5) — rows=subtypes, cols=bg_types
MATCH_MATRIX = np.array([
    # smo   ver   hor   tex   com
    [0.7,  1.0,  1.0,  0.5,  0.3],   # linear_scratch
    [0.8,  0.9,  0.9,  0.6,  0.4],   # elongated
    [1.0,  0.5,  0.5,  0.7,  0.6],   # compact_blob
    [0.6,  0.5,  0.5,  0.8,  1.0],   # irregular
    [0.6,  0.6,  0.6,  0.6,  0.6],   # general (neutral)
])

# Display labels (shorter)
DEFECT_LABELS = [
    "linear_scratch\n(Linearity > 0.7)",
    "elongated\n(Aspect > 3.0)",
    "compact_blob\n(Solidity > 0.8)",
    "irregular",
    "general\n(default)",
]
BG_LABELS = [
    "smooth",
    "vertical\nstripe",
    "horizontal\nstripe",
    "textured",
    "complex\npattern",
]

# ──────────────────────────────────────────────────────────────────────────────
# 컬러맵: white(0) → yellow(0.5) → green(1.0)
# ──────────────────────────────────────────────────────────────────────────────
CMAP = LinearSegmentedColormap.from_list(
    "suit",
    [(0.0,  "#FFEBEE"),   # light red-pink for low match
     (0.3,  "#FFF9C4"),   # pale yellow for medium
     (0.6,  "#C8E6C9"),   # light green
     (1.0,  "#1B5E20")],  # dark green for best match
)

# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 5.8), facecolor="white")
fig.suptitle(
    "CASDA Stage A §5 — Defect–Background Suitability Matching Matrix",
    fontsize=11, fontweight="bold", color="#0D47A1", y=0.99,
)

gs = gridspec.GridSpec(
    1, 2,
    figure=fig,
    width_ratios=[1.0, 0.82],
    wspace=0.38,
    top=0.90, bottom=0.10, left=0.04, right=0.98,
)

# ══════════════════════════════════════════════════════════════════════════════
# Left: Heatmap
# ══════════════════════════════════════════════════════════════════════════════
ax_heat = fig.add_subplot(gs[0, 0])

im = ax_heat.imshow(MATCH_MATRIX, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")

n_r, n_c = MATCH_MATRIX.shape
for r in range(n_r):
    for c in range(n_c):
        val = MATCH_MATRIX[r, c]
        txt_col = "white" if val > 0.75 else "black"
        weight  = "bold"  if val >= 0.9  else "normal"
        # best-match star
        star = " ★" if val == MATCH_MATRIX[r].max() and val >= 0.9 else ""
        ax_heat.text(
            c, r, f"{val:.1f}{star}",
            ha="center", va="center",
            fontsize=9.5, color=txt_col, fontweight=weight,
        )

ax_heat.set_xticks(np.arange(n_c))
ax_heat.set_yticks(np.arange(n_r))
ax_heat.set_xticklabels(BG_LABELS, fontsize=8.5)
ax_heat.set_yticklabels(DEFECT_LABELS, fontsize=8.5)
ax_heat.set_xlabel("Background Type", fontsize=9, labelpad=6)
ax_heat.set_ylabel("Defect Subtype", fontsize=9, labelpad=6)
ax_heat.tick_params(length=0)

# Grid lines
ax_heat.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
ax_heat.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
ax_heat.grid(which="minor", color="white", linewidth=1.5)
ax_heat.tick_params(which="minor", length=0)

# Colorbar
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.038, pad=0.03, aspect=18)
cbar.set_label("Suitability Score", fontsize=8)
cbar.ax.tick_params(labelsize=7.5)
cbar.set_ticks([0.3, 0.5, 0.7, 0.9, 1.0])

ax_heat.set_title(
    "(A)  Match Score Matrix  (defect_subtype × background_type)",
    fontsize=9, fontweight="bold", color="#1A237E", pad=6,
)

# ══════════════════════════════════════════════════════════════════════════════
# Right: Intuition diagram
# ══════════════════════════════════════════════════════════════════════════════
ax_int = fig.add_subplot(gs[0, 1])
ax_int.axis("off")

ax_int.set_title(
    "(B)  Design Rationale",
    fontsize=9, fontweight="bold", color="#1A237E", pad=6,
)

intuitions = [
    ("linear_scratch\n+ vertical/horizontal stripe",
     "1.0 ★",
     "Scratch aligns with stripe direction\n"
     "→ defect blends into background pattern\n"
     "→ ControlNet learns contrast, not background",
     "#43A047"),

    ("compact_blob\n+ smooth",
     "1.0 ★",
     "Blob stands out clearly on uniform surface\n"
     "→ minimal texture distraction\n"
     "→ optimal for defect boundary learning",
     "#43A047"),

    ("irregular\n+ complex_pattern",
     "1.0 ★",
     "Irregular defect edges match complex background\n"
     "→ realistic visual overlap\n"
     "→ prevents over-segmentation in training",
     "#43A047"),

    ("linear_scratch\n+ complex_pattern",
     "0.3 ✗",
     "Linear defect overwhelmed by noise\n"
     "→ defect signal hard to isolate\n"
     "→ poor conditioning quality",
     "#E53935"),
]

y = 0.97
for (pair, score, reason, col) in intuitions:
    ax_int.text(0.02, y, f"  {pair}  →  {score}",
                transform=ax_int.transAxes,
                fontsize=8, fontweight="bold", color=col, va="top")
    y -= 0.09
    for line in reason.split("\n"):
        ax_int.text(0.06, y, line,
                    transform=ax_int.transAxes,
                    fontsize=7, color="#424242", va="top", style="italic")
        y -= 0.065
    y -= 0.035

# Legend
ax_int.text(
    0.02, y - 0.01,
    "★  Best match (score ≥ 0.9)      ✗  Avoid (score ≤ 0.3)",
    transform=ax_int.transAxes,
    fontsize=7.5, color="#555", va="top",
    bbox=dict(fc="#F5F5F5", alpha=0.85, pad=4, boxstyle="round,pad=0.5"),
)

# ──────────────────────────────────────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FILE, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved → {OUT_FILE}")
plt.show()
