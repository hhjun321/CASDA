"""
figure_stageB_hint.py
=============================
3.2.2 Stage B: ControlNet 3-Channel Hint Image Construction

Layout: 4행 (Class 1~4) × 5열
  Col 1: ROI Patch (256×256) + defect mask overlay + bg_type badge
  Col 2: R Channel — defect geometry (skeletonize / Canny+fill / fill)
  Col 3: G Channel — surface orientation via Sobel (direction by bg_type)
  Col 4: B Channel — surface roughness via local variance (7×7 kernel)
  Col 5: Hint Image — 0.5R + 0.3G + 0.2B weighted grayscale

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_stageB_hint.py
"""

import ast
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

# ──────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
TRAIN_CSV    = DRIVE_DATA / "train.csv"
ROI_META     = DRIVE_DATA / "roi_patches_v5.1/roi_metadata.csv"

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "stageB_hint.jpg"

IMG_H, IMG_W = 256, 1600

# ──────────────────────────────────────────────────────────────────────────────
# 클래스 색상
# ──────────────────────────────────────────────────────────────────────────────
CLASS_COLOR = {1: "#2196F3", 2: "#F44336", 3: "#4CAF50", 4: "#FF9800"}

BG_TYPE_COLOR = {
    "smooth":            "#78909C",
    "vertical_stripe":   "#42A5F5",
    "horizontal_stripe": "#EF5350",
    "textured":          "#AB47BC",
    "complex_pattern":   "#FF7043",
}

# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
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
        return ast.literal_eval(str(val))
    except Exception:
        return None


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Hint channel 생성 (hint_generator.py 동일 로직)
# ──────────────────────────────────────────────────────────────────────────────

def generate_red_channel(mask, linearity, solidity):
    """
    R channel: defect geometry encoding.
      linearity > 0.7 → skeletonize + dilate
      solidity  > 0.8 → fill (200)
      else            → Canny edges + fill (100)
    """
    out = np.zeros_like(mask, dtype=np.uint8)
    if mask.max() == 0:
        return out

    if linearity > 0.7:
        skel = skeletonize(mask > 0).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        out = cv2.dilate(skel, kernel, iterations=1)
    elif solidity > 0.8:
        out[mask > 0] = 200
    else:
        edges = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges_d = cv2.dilate(edges, kernel, iterations=1)
        out[mask > 0] = 100
        out[edges_d > 0] = 255
    return out


def generate_green_channel(gray_patch, background_type, stability_score):
    """
    G channel: surface orientation via directional Sobel.
    Direction selected by background_type.
    Intensity modulated by stability_score.
    """
    sobel_x = cv2.Sobel(gray_patch.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_patch.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)

    if background_type == "vertical_stripe":
        edge_map = np.abs(sobel_x)
    elif background_type == "horizontal_stripe":
        edge_map = np.abs(sobel_y)
    elif background_type == "complex_pattern":
        edge_map = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    else:  # smooth, textured
        edge_map = np.sqrt(sobel_x ** 2 + sobel_y ** 2) * 0.3

    edge_norm = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX)
    intensity_factor = 0.5 + 0.5 * float(stability_score)
    return np.clip(edge_norm * intensity_factor, 0, 255).astype(np.uint8)


def generate_blue_channel(gray_patch, background_type):
    """
    B channel: surface roughness via local variance (7×7 kernel).
    smooth → constant 20.
    """
    if background_type == "smooth":
        return np.full_like(gray_patch, 20, dtype=np.uint8)

    kernel = np.ones((7, 7), np.float32) / 49
    f32 = gray_patch.astype(np.float32)
    local_mean    = cv2.filter2D(f32, -1, kernel)
    local_sq_mean = cv2.filter2D(f32 ** 2, -1, kernel)
    local_var     = np.maximum(local_sq_mean - local_mean ** 2, 0)
    out = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if background_type in ("textured", "complex_pattern"):
        out = np.clip(out.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    return out


def generate_hint(r_ch, g_ch, b_ch):
    """Final hint: weighted grayscale stack (0.5R + 0.3G + 0.2B)."""
    gray = (0.5 * r_ch.astype(np.float32)
            + 0.3 * g_ch.astype(np.float32)
            + 0.2 * b_ch.astype(np.float32))
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────
print("Loading ROI metadata ...")
df_all = pd.read_csv(ROI_META, sep=None, engine="python")
df_all.columns = df_all.columns.str.strip().str.lower().str.replace(" ", "_")
df_all["roi_bbox"] = df_all["roi_bbox"].apply(safe_bbox)
df_all = df_all.dropna(subset=["roi_bbox"])

print("Loading train.csv ...")
train_df = pd.read_csv(TRAIN_CSV, sep=None, engine="python")
train_df.columns = train_df.columns.str.strip()
col_img = [c for c in train_df.columns if "image"   in c.lower()][0]
col_cls = [c for c in train_df.columns if "class"   in c.lower()][0]
col_rle = [c for c in train_df.columns if "encoded" in c.lower()
                                        or "pixel"  in c.lower()][0]

sort_col = "suitability_score" if "suitability_score" in df_all.columns \
           else df_all.columns[0]

# 클래스별 대표 ROI
selected = {}
for cls_id in [1, 2, 3, 4]:
    df_cls = df_all[df_all["class_id"] == cls_id].sort_values(
        sort_col, ascending=False
    )
    if df_cls.empty:
        print(f"  [WARN] Class {cls_id} ROI 없음")
        continue
    selected[cls_id] = df_cls.iloc[0]
    r = selected[cls_id]
    bg = r.get("background_type", "unknown")
    print(f"  Class {cls_id}: {r['image_id']}  bg={bg}  "
          f"score={r.get(sort_col, 0):.3f}")

# 이미지·마스크 캐시
steel_cache, mask_cache = {}, {}
for cls_id, row in selected.items():
    img_id = row["image_id"]
    if img_id not in steel_cache:
        steel_cache[img_id] = load_gray(TRAIN_IMAGES / img_id)
        fm = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for _, tr in train_df[train_df[col_img] == img_id].iterrows():
            fm = np.maximum(fm, rle_to_mask(tr[col_rle]))
        mask_cache[img_id] = fm

# ──────────────────────────────────────────────────────────────────────────────
# Figure 생성
# ──────────────────────────────────────────────────────────────────────────────
N_ROWS    = len(selected)
N_COLS    = 5
FIG_W     = 13.0
FIG_H     = 2.5 * N_ROWS + 0.6

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

col_ratios = [1.0, 1.0, 1.0, 1.0, 1.0]

gs_outer = gridspec.GridSpec(
    N_ROWS, 1,
    figure=fig,
    hspace=0.08,
    top=0.96, bottom=0.06, left=0.04, right=0.97,
)

STEP_LABELS = [
    "Step 1 — ROI Patch\n(256×256)",
    "Step 2a — R Channel\nDefect Geometry",
    "Step 2b — G Channel\nSurface Orientation",
    "Step 2c — B Channel\nSurface Roughness",
    "Step 3 — Hint Image\n(0.5R + 0.3G + 0.2B)",
]

for row_idx, cls_id in enumerate([1, 2, 3, 4]):
    if cls_id not in selected:
        continue

    row       = selected[cls_id]
    img_id    = row["image_id"]
    steel_g   = steel_cache[img_id]
    cls_color = CLASS_COLOR[cls_id]

    x1r, y1r, x2r, y2r = [int(v) for v in row["roi_bbox"]]

    # cls mask
    cls_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for _, tr in train_df[
        (train_df[col_img] == img_id) & (train_df[col_cls] == cls_id)
    ].iterrows():
        cls_mask = np.maximum(cls_mask, rle_to_mask(tr[col_rle]))

    roi_patch  = steel_g[y1r:y2r, x1r:x2r]
    mask_patch = cls_mask[y1r:y2r, x1r:x2r]

    # 메타데이터에서 hint 생성에 필요한 파라미터
    linearity    = float(row.get("linearity",    0.0))
    solidity     = float(row.get("solidity",     0.5))
    bg_type      = str(row.get("background_type", "smooth")).strip()
    stability    = float(row.get("stability_score", 0.5))

    # hint channels
    r_ch = generate_red_channel(mask_patch, linearity, solidity)
    g_ch = generate_green_channel(roi_patch, bg_type, stability)
    b_ch = generate_blue_channel(roi_patch, bg_type)
    hint = generate_hint(r_ch, g_ch, b_ch)

    # ── 행 GridSpec ─────────────────────────────────────────────────
    gs_row = gridspec.GridSpecFromSubplotSpec(
        1, N_COLS,
        subplot_spec=gs_outer[row_idx],
        width_ratios=col_ratios,
        wspace=0.07,
    )

    # ── 공통 spine 스타일 ────────────────────────────────────────────
    def style_ax(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.8); sp.set_edgecolor(cls_color)

    # ════════════════════════════════════════════════════════════════
    # COL 1: ROI Patch + defect mask overlay + bg_type badge
    # ════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs_row[0, 0])
    ax1.imshow(roi_patch, cmap="gray", aspect="auto", vmin=0, vmax=255)

    if mask_patch.any():
        rgba_m = np.zeros((*roi_patch.shape, 4), dtype=np.float32)
        rgba_m[mask_patch > 0] = [1.0, 0.2, 0.2, 0.45]
        ax1.imshow(rgba_m, aspect="auto")

    ax1.set_ylabel(f"Class {cls_id}", fontsize=13, fontweight="bold",
                   color=cls_color, rotation=90, labelpad=5)
    style_ax(ax1)

    # ════════════════════════════════════════════════════════════════
    # COL 2: R channel
    # ════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs_row[0, 1])
    ax2.imshow(r_ch, cmap="inferno", aspect="auto", vmin=0, vmax=255)

    style_ax(ax2)

    # ════════════════════════════════════════════════════════════════
    # COL 3: G channel
    # ════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs_row[0, 2])
    ax3.imshow(g_ch, cmap="viridis", aspect="auto", vmin=0, vmax=255)

    style_ax(ax3)

    # ════════════════════════════════════════════════════════════════
    # COL 4: B channel
    # ════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs_row[0, 3])
    ax4.imshow(b_ch, cmap="magma", aspect="auto", vmin=0, vmax=255)

    style_ax(ax4)

    # ════════════════════════════════════════════════════════════════
    # COL 5: Final hint image
    # ════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs_row[0, 4])
    ax5.imshow(hint, aspect="auto")
    style_ax(ax5)
    for sp in ax5.spines.values():
        sp.set_linewidth(2.4); sp.set_edgecolor(cls_color)

    # ── 첫 행에만 열 제목 ────────────────────────────────────────────
    if row_idx == 0:
        for ax, lbl in zip([ax1, ax2, ax3, ax4, ax5], STEP_LABELS):
            ax.set_title(lbl, fontsize=12, fontweight="bold",
                         color="#546E7A", pad=6)

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
