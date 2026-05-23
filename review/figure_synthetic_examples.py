"""
figure_synthetic_examples.py
=============================
§3.2.3 Stage C: ControlNet-Generated Synthetic Defect Image Examples

Layout: 4 rows (Class 1–4) × 5 cols (5 examples per class)
Each cell: ControlNet-generated ROI patch, CLAHE enhanced

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_synthetic_examples.py
"""

import json
import re
from pathlib import Path
import sys

try:
    PROJ_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJ_ROOT = Path("/content/CASDA")

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_DATA    = Path("/content/drive/MyDrive/data/Severstal")
GENERATED_DIR = DRIVE_DATA / "augmented_images/generated"
META_COMPOSED = DRIVE_DATA / "augmented_dataset/casda_composed/metadata.json"

OUT_DIR  = PROJ_ROOT / "review/figures"
OUT_FILE = OUT_DIR / "synthetic_examples.jpg"

N_EXAMPLES = 5

CLASS_COLOR = {1: "#2196F3", 2: "#F44336", 3: "#4CAF50", 4: "#FF9800"}

# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────
print("Loading metadata ...")
with open(META_COMPOSED) as f:
    metadata = json.load(f)
print(f"  {len(metadata)} entries")

by_class: dict = {}
for e in metadata:
    cls = int(e.get("class_id", 0))
    by_class.setdefault(cls, []).append(e)
for cls in by_class:
    by_class[cls].sort(key=lambda x: x.get("suitability_score", 0), reverse=True)

selected: dict = {}
for cls_id in [1, 2, 3, 4]:
    rows = []
    for e in by_class.get(cls_id, []):
        if (GENERATED_DIR / e["source_generated"]).exists():
            rows.append(e)
            if len(rows) >= N_EXAMPLES:
                break
    if len(rows) < N_EXAMPLES:
        print(f"  [WARN] Class {cls_id}: {len(rows)}/{N_EXAMPLES} examples found")
    selected[cls_id] = rows
    print(f"  Class {cls_id}: {len(rows)} examples  "
          f"score={rows[0].get('suitability_score', 0):.3f}" if rows else
          f"  Class {cls_id}: 0 examples")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 생성
# ──────────────────────────────────────────────────────────────────────────────
N_ROWS = 4
N_COLS = N_EXAMPLES
FIG_W  = 2.6 * N_COLS + 0.8
FIG_H  = 2.6 * N_ROWS + 0.6

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

gs = gridspec.GridSpec(
    N_ROWS, N_COLS,
    figure=fig,
    hspace=0.06, wspace=0.05,
    top=0.93, bottom=0.03, left=0.09, right=0.98,
)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for row_idx, cls_id in enumerate([1, 2, 3, 4]):
    cls_color = CLASS_COLOR[cls_id]
    rows = selected.get(cls_id, [])

    for col_idx in range(N_COLS):
        ax = fig.add_subplot(gs[row_idx, col_idx])

        if col_idx < len(rows):
            gen_path = GENERATED_DIR / rows[col_idx]["source_generated"]
            img = cv2.imread(str(gen_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_disp = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                img_disp = clahe.apply(img_disp)
            else:
                img_disp = np.full((256, 256), 96, dtype=np.uint8)
        else:
            img_disp = np.full((256, 256), 64, dtype=np.uint8)

        ax.imshow(img_disp, cmap="gray", aspect="equal", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(2.5)
            sp.set_edgecolor(cls_color)

        if row_idx == 0:
            ax.set_title(f"Example {col_idx + 1}", fontsize=12, fontweight="bold",
                         color="#546E7A", pad=5)

        if col_idx == 0:
            ax.set_ylabel(f"Class {cls_id}", fontsize=13, fontweight="bold",
                          color=cls_color, rotation=90, labelpad=5)

# ──────────────────────────────────────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none",
            format="jpeg", pil_kwargs={"quality": 95, "subsampling": 0})
print(f"Saved → {OUT_FILE}")
plt.show()
print("Done.")
