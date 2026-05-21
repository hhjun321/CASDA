"""
figure_hint_channels.py  —  Stage A §6: Multi-Channel Hint Image Generation
=============================================================================
논문 §6 "멀티채널 힌트 이미지" 시각화 figure.

Layout  (2 rows × 5 columns):
  Each row = one representative ROI (linear_scratch vs compact_blob)
  Col 0 : Original ROI patch (grayscale steel)
  Col 1 : R channel — Defect mask (false-colored RED)
  Col 2 : G channel — Background structure (false-colored GREEN)
  Col 3 : B channel — Texture density  (false-colored BLUE)
  Col 4 : Final Hint — 0.5R + 0.3G + 0.2B  (grayscale)

  Bottom annotation row: formula + color-shortcut prevention rationale

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_hint_channels.py
"""

import ast
import os
import sys
from pathlib import Path

try:
    PROJ_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJ_ROOT = Path("/content/CASDA")   # Colab 셀 붙여넣기 시 폴백

sys.path.insert(0, str(PROJ_ROOT))

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.hint_generator import HintImageGenerator

# ──────────────────────────────────────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
TRAIN_CSV    = DRIVE_DATA / "train.csv"
ROI_META     = DRIVE_DATA / "roi_patches/roi_metadata.csv"

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "hint_channels.png"

IMG_H, IMG_W = 256, 1600
PATCH_SIZE   = 256

# 대표 ROI 유형: [linear_scratch, compact_blob]
# (부족하면 class_id 기준으로 대체)
TARGET_SUBTYPES = ["linear_scratch", "compact_blob"]

# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def rle_to_mask(rle_str: str, h=IMG_H, w=IMG_W) -> np.ndarray:
    flat = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle_str, str) and rle_str.strip():
        nums = list(map(int, rle_str.split()))
        for s, l in zip(nums[0::2], nums[1::2]):
            flat[s - 1: s - 1 + l] = 1
    return flat.reshape(w, h).T


def safe_bbox(val):
    try:
        return [int(v) for v in ast.literal_eval(str(val))]
    except Exception:
        return None


def load_image_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")
    return img


def colorize(channel: np.ndarray, rgb: tuple) -> np.ndarray:
    """단일 채널 → false-color RGB 이미지. rgb=(r,g,b) in 0-1."""
    out = np.zeros((*channel.shape, 3), dtype=np.float32)
    out[:, :, 0] = channel / 255.0 * rgb[0]
    out[:, :, 1] = channel / 255.0 * rgb[1]
    out[:, :, 2] = channel / 255.0 * rgb[2]
    return out


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
col_img = [c for c in train_df.columns if "image" in c.lower()][0]
col_cls = [c for c in train_df.columns if "class" in c.lower()][0]
col_rle = [c for c in train_df.columns if "encoded" in c.lower() or "pixel" in c.lower()][0]

# ──────────────────────────────────────────────────────────────────────────────
# 대표 ROI 2개 선택
# ──────────────────────────────────────────────────────────────────────────────
hint_gen = HintImageGenerator()
selected = []

for target_sub in TARGET_SUBTYPES:
    sub_col = "defect_subtype" if "defect_subtype" in df_all.columns else None

    if sub_col:
        pool = df_all[df_all[sub_col] == target_sub]
    else:
        # fallback: class 1 → linear-like, class 3 → compact-like
        cls_map = {"linear_scratch": 1, "compact_blob": 3}
        pool = df_all[df_all.get("class_id", df_all.iloc[:, 0]) == cls_map.get(target_sub, 1)]

    # suitability 높은 순, valid bbox 필터
    sort_col = "suitability_score" if "suitability_score" in pool.columns else pool.columns[0]
    pool = pool.sort_values(sort_col, ascending=False)

    found = False
    for _, row in pool.iterrows():
        x1, y1, x2, y2 = row["roi_bbox"]
        img_id = row.get("image_id", None)
        if img_id is None:
            continue
        img_path = TRAIN_IMAGES / str(img_id)
        if not img_path.exists():
            continue
        # 마스크 합산
        rle_rows = train_df[train_df[col_img] == str(img_id)]
        if rle_rows.empty:
            continue
        full_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for _, rr in rle_rows.iterrows():
            full_mask = np.maximum(full_mask, rle_to_mask(rr[col_rle]))
        roi_mask = full_mask[y1:y2, x1:x2]
        if roi_mask.sum() < 50:    # 결함이 거의 없으면 스킵
            continue

        gray_full = load_image_gray(img_path)
        roi_gray  = gray_full[y1:y2, x1:x2]

        # 결함 지표
        linearity    = float(row.get("linearity",    0.5))
        solidity     = float(row.get("solidity",     0.5))
        extent       = float(row.get("extent",       0.5))
        aspect_ratio = float(row.get("aspect_ratio", 1.0))
        defect_metrics = dict(linearity=linearity, solidity=solidity,
                              extent=extent, aspect_ratio=aspect_ratio)
        bg_type   = str(row.get("background_type", "smooth"))
        stability = float(row.get("stability_score", 0.7))
        class_id  = int(row.get("class_id", 0))
        subtype   = str(row.get("defect_subtype", target_sub))

        # 채널 생성
        ch_r = hint_gen.generate_red_channel(roi_mask, defect_metrics)
        ch_g = hint_gen.generate_green_channel(roi_gray, bg_type, stability)
        ch_b = hint_gen.generate_blue_channel(roi_gray, bg_type)
        gray_hint = np.clip(
            0.5 * ch_r.astype(np.float32)
            + 0.3 * ch_g.astype(np.float32)
            + 0.2 * ch_b.astype(np.float32),
            0, 255,
        ).astype(np.uint8)

        selected.append(dict(
            img_id=img_id, roi_gray=roi_gray, roi_mask=roi_mask,
            ch_r=ch_r, ch_g=ch_g, ch_b=ch_b, gray_hint=gray_hint,
            defect_metrics=defect_metrics, bg_type=bg_type,
            stability=stability, class_id=class_id, subtype=subtype,
            linearity=linearity, solidity=solidity,
        ))
        found = True
        print(f"  [{target_sub}] selected {img_id}  "
              f"subtype={subtype}  bg={bg_type}  "
              f"lin={linearity:.2f}  sol={solidity:.2f}")
        break

    if not found:
        print(f"  WARNING: no valid ROI found for subtype={target_sub}")

if len(selected) < 2:
    raise RuntimeError("대표 ROI 2개를 찾지 못했습니다. ROI metadata를 확인하세요.")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 생성
# ──────────────────────────────────────────────────────────────────────────────
N_ROWS = len(selected)   # 2
N_COLS = 5               # original | R | G | B | Hint

PANEL_SIZE = 2.3         # inch per patch panel
LABEL_W    = 1.6         # inch for left row label
FIG_W  = LABEL_W + N_COLS * PANEL_SIZE + 0.4
FIG_H  = N_ROWS * PANEL_SIZE + 2.2  # + annotation row

COL_TITLES = [
    "Original\n(steel ROI)",
    "R Channel\n(Defect Mask)",
    "G Channel\n(Background\nStructure)",
    "B Channel\n(Texture\nDensity)",
    "Final Hint\n(0.5R + 0.3G + 0.2B)",
]
COL_COLORS = ["#ECEFF1", "#FFCDD2", "#C8E6C9", "#BBDEFB", "#F5F5F5"]

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
fig.suptitle(
    "CASDA Stage A §6 — Multi-Channel Hint Image Generation\n"
    "(Grayscale conversion prevents ControlNet color-shortcut learning)",
    fontsize=10, fontweight="bold", color="#0D47A1", y=0.99,
)

gs = gridspec.GridSpec(
    N_ROWS + 1, N_COLS + 1,          # +1 행: 하단 수식 주석, +1 열: 행 레이블
    figure=fig,
    width_ratios=[0.55] + [1.0] * N_COLS,
    height_ratios=[1.0] * N_ROWS + [0.55],
    hspace=0.28, wspace=0.12,
    top=0.91, bottom=0.04, left=0.03, right=0.99,
)

# ── 컬럼 헤더 ──
for c, (title, bg_col) in enumerate(zip(COL_TITLES, COL_COLORS)):
    ax_h = fig.add_subplot(gs[0, c + 1])   # row 0 used as header in first sample
# (headers are drawn as ax_patch titles instead — simpler)

SUBTYPE_LABELS = {
    "linear_scratch": "Linear Scratch\n(Linearity > 0.7)",
    "compact_blob":   "Compact Blob\n(Solidity > 0.8)",
    "elongated":      "Elongated\n(Aspect > 3.0)",
    "irregular":      "Irregular",
    "general":        "General",
}

for row_i, entry in enumerate(selected):
    subtype_label = SUBTYPE_LABELS.get(entry["subtype"], entry["subtype"])

    # ── 행 레이블 (col 0) ──
    ax_lbl = fig.add_subplot(gs[row_i, 0])
    ax_lbl.axis("off")
    ax_lbl.text(
        0.5, 0.5, subtype_label,
        transform=ax_lbl.transAxes, ha="center", va="center",
        fontsize=8, fontweight="bold", color="#1A237E",
        rotation=90,
    )

    panels = [
        ("original", entry["roi_gray"],  "gray", "gray",    None),
        ("red",      entry["ch_r"],      None,   "red_ch",  (1.0, 0.15, 0.15)),
        ("green",    entry["ch_g"],      None,   "grn_ch",  (0.15, 0.82, 0.15)),
        ("blue",     entry["ch_b"],      None,   "blu_ch",  (0.15, 0.45, 1.0)),
        ("hint",     entry["gray_hint"], "gray", "hint",    None),
    ]

    for col_i, (panel_type, data, cmap, _, rgb_tint) in enumerate(panels):
        ax = fig.add_subplot(gs[row_i, col_i + 1])

        if cmap == "gray":
            ax.imshow(data, cmap="gray", vmin=0, vmax=255, aspect="equal")
        elif rgb_tint is not None:
            ax.imshow(colorize(data, rgb_tint), aspect="equal")
        else:
            ax.imshow(data, cmap="gray", vmin=0, vmax=255, aspect="equal")

        # 결함 마스크 오버레이 (original 패널에만)
        if panel_type == "original" and entry["roi_mask"].any():
            mask_rgba = np.zeros((*entry["roi_mask"].shape, 4), dtype=np.float32)
            mask_rgba[entry["roi_mask"] > 0] = [1.0, 0.2, 0.2, 0.55]
            ax.imshow(mask_rgba, aspect="equal")

        # 테두리 색상
        border_colors = {
            "original": "#607D8B",
            "red_ch":   "#E53935",
            "grn_ch":   "#43A047",
            "blu_ch":   "#1E88E5",
            "hint":     "#1A237E",
        }
        for sp in ax.spines.values():
            sp.set_linewidth(2.5)
            sp.set_edgecolor(border_colors.get(_, "#aaa"))

        ax.set_xticks([]); ax.set_yticks([])

        # 첫 번째 행: 열 타이틀
        if row_i == 0:
            ax.set_title(
                COL_TITLES[col_i],
                fontsize=7.5, fontweight="bold", pad=4,
                color=border_colors.get(_, "#333"),
                bbox=dict(fc=COL_COLORS[col_i], alpha=0.75, pad=2,
                          boxstyle="round,pad=0.4"),
            )

        # 메타 주석 (패치 내부 하단)
        if panel_type == "original":
            info = (f"Class {entry['class_id']}  "
                    f"lin={entry['linearity']:.2f}  "
                    f"sol={entry['solidity']:.2f}")
        elif panel_type == "red_ch":
            if entry["linearity"] > 0.7:
                proc = "skeleton + dilation"
            elif entry["solidity"] > 0.8:
                proc = "filled mask"
            else:
                proc = "Canny edges + fill"
            info = proc
        elif panel_type == "grn_ch":
            bg_map = {
                "vertical_stripe":   "Sobel-X  ×{:.2f}",
                "horizontal_stripe": "Sobel-Y  ×{:.2f}",
                "complex_pattern":   "Sobel-XY ×{:.2f}",
            }
            tmpl = bg_map.get(entry["bg_type"], "Sobel-XY ×{:.2f}×0.3")
            info = tmpl.format(0.5 + 0.5 * entry["stability"])
        elif panel_type == "blu_ch":
            info = "7×7 local variance" if entry["bg_type"] != "smooth" else "constant (20)"
        else:
            info = "0.5R + 0.3G + 0.2B"

        ax.text(
            0.5, 0.03, info,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=5.8, color="white",
            bbox=dict(fc="black", alpha=0.62, pad=1.8, boxstyle="round,pad=0.25"),
        )

# ──────────────────────────────────────────────────────────────────────────────
# 하단 수식 + 설계 의도 주석
# ──────────────────────────────────────────────────────────────────────────────
ax_ann = fig.add_subplot(gs[N_ROWS, :])
ax_ann.axis("off")

formula_text = (
    "Hint Grayscale Formula:   "
    r"$\mathbf{gray}$"
    " = 0.5 × R  +  0.3 × G  +  0.2 × B"
    "       →       "
    r"$\mathbf{hint}$"
    " = [gray, gray, gray]  (3-channel replication)"
)
ax_ann.text(
    0.5, 0.70, formula_text,
    transform=ax_ann.transAxes, ha="center", va="top",
    fontsize=9, fontweight="bold", color="#0D47A1",
    bbox=dict(fc="#E3F2FD", alpha=0.9, pad=5, boxstyle="round,pad=0.5"),
)

rationale_lines = [
    "Why grayscale?  RGB hint → ControlNet learns color shortcut",
    "(red mask region copied as red pixels into generated image)",
    "Grayscale removes color information → model learns structure, not color",
]
ax_ann.text(
    0.5, 0.25,
    "   ·   ".join(rationale_lines),
    transform=ax_ann.transAxes, ha="center", va="top",
    fontsize=7.5, color="#B71C1C",
    style="italic",
)

# ──────────────────────────────────────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FILE, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved → {OUT_FILE}")
plt.show()
