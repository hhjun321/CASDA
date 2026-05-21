"""
figure_stageA_grid.py  —  Stage A ROI Extraction: Grid & Background Classification
====================================================================================
[Image #1] b18d448a7.jpg_debug.png 의 3패널 레이아웃을 논문 품질로 재현.

Layout:
  Row A: Original steel image (1600×256) + 64 px sliding grid overlay
  Row B: Background type classification grid (per-cell color + label + score)
  Row C: 3 extracted ROI patches with metadata (각 배경 유형 대표 1개씩)

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_stageA_grid.py
"""

import ast
import os
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

# ──────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
TRAIN_CSV    = DRIVE_DATA / "train.csv"
ROI_META     = DRIVE_DATA / "roi_patches/roi_metadata.csv"   # roi_patches_v5.1 로 수정 가능

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "stageA_grid.png"

DEMO_IMAGE_ID = "b18d448a7.jpg"
GRID_SIZE     = 64        # px — 슬라이딩 그리드 간격
IMG_H, IMG_W  = 256, 1600

# ──────────────────────────────────────────────────────────────────────────────
# 배경 유형 색상 & 약어
# ──────────────────────────────────────────────────────────────────────────────
BG_META = {
    "smooth":            dict(abbr="SMO", color="#2196F3", text_col="white"),
    "vertical_stripe":   dict(abbr="VER", color="#4CAF50", text_col="white"),
    "horizontal_stripe": dict(abbr="HOR", color="#FF9800", text_col="black"),
    "textured":          dict(abbr="TEX", color="#9C27B0", text_col="white"),
    "complex_pattern":   dict(abbr="COM", color="#F44336", text_col="white"),
}
BG_DEFAULT = dict(abbr="?",   color="#607D8B", text_col="white")

ROI_BOX_COLORS = {
    "smooth":            "#00E676",   # 민트 그린
    "vertical_stripe":   "#40C4FF",   # 하늘 파랑
    "horizontal_stripe": "#FFEB3B",   # 노랑
    "textured":          "#CE93D8",   # 라벤더
    "complex_pattern":   "#FFAB40",   # 오렌지
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
        raise FileNotFoundError(f"이미지 로드 실패: {path}")
    return img


def classify_bg_from_patch(patch_gray: np.ndarray) -> str:
    """
    compose_casda_images.py:classify_background_simple() 와 동일 로직.
    단, roi_metadata의 background_type 컬럼이 있으면 그것을 우선 사용.
    """
    h, w = patch_gray.shape
    patch = cv2.resize(patch_gray, (128, 128), interpolation=cv2.INTER_AREA)
    variance = np.var(patch.astype(np.float32))
    edges = cv2.Canny(patch, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    if variance < 200 and edge_density < 0.02:
        return "smooth"

    sx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    ex = np.mean(np.abs(sx))
    ey = np.mean(np.abs(sy))
    tot = ex + ey + 1e-7
    if ex / tot > 0.65 and edge_density > 0.02:
        return "vertical_stripe"
    if ey / tot > 0.65 and edge_density > 0.02:
        return "horizontal_stripe"
    if edge_density > 0.15:
        return "complex_pattern"
    if variance > 500 or edge_density > 0.05:
        return "textured"
    return "smooth"


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading {DEMO_IMAGE_ID} ...")
steel_gray = load_gray(TRAIN_IMAGES / DEMO_IMAGE_ID)

print("Loading ROI metadata ...")
df_all = pd.read_csv(ROI_META, sep=None, engine="python")
df_all.columns = df_all.columns.str.strip().str.lower().str.replace(" ", "_")
df_all["roi_bbox"] = df_all["roi_bbox"].apply(safe_bbox)
df_all = df_all.dropna(subset=["roi_bbox"])
df_img = df_all[df_all["image_id"] == DEMO_IMAGE_ID].reset_index(drop=True)
print(f"  {len(df_img)} ROIs for {DEMO_IMAGE_ID}")

# RLE 마스크 (결함 표시용)
print("Loading train.csv ...")
train_df = pd.read_csv(TRAIN_CSV, sep=None, engine="python")
train_df.columns = train_df.columns.str.strip()
col_img = [c for c in train_df.columns if "image" in c.lower()][0]
col_cls = [c for c in train_df.columns if "class" in c.lower()][0]
col_rle = [c for c in train_df.columns if "encoded" in c.lower() or "pixel" in c.lower()][0]

full_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
for _, row in train_df[train_df[col_img] == DEMO_IMAGE_ID].iterrows():
    full_mask = np.maximum(full_mask, rle_to_mask(row[col_rle]))

# ──────────────────────────────────────────────────────────────────────────────
# 그리드 배경 유형 계산
# ──────────────────────────────────────────────────────────────────────────────
NC = IMG_W // GRID_SIZE   # 25 columns
NR = IMG_H // GRID_SIZE   #  4 rows

# roi_metadata에 background_type 컬럼이 있으면 활용
has_bg_col = "background_type" in df_all.columns

# 셀별 배경 유형 (재계산 + metadata 교차 참조)
cell_bg   = np.full((NR, NC), "", dtype=object)
cell_score = np.zeros((NR, NC), dtype=np.float32)  # suitability proxy

for r in range(NR):
    for c in range(NC):
        patch = steel_gray[r * GRID_SIZE:(r + 1) * GRID_SIZE,
                           c * GRID_SIZE:(c + 1) * GRID_SIZE]
        cell_bg[r, c] = classify_bg_from_patch(patch)
        # defect density as proxy score
        mask_cell = full_mask[r * GRID_SIZE:(r + 1) * GRID_SIZE,
                              c * GRID_SIZE:(c + 1) * GRID_SIZE]
        cell_score[r, c] = float(np.clip(mask_cell.mean() * 20.0 + patch.std() / 255 * 1.2, 0, 1))

# ──────────────────────────────────────────────────────────────────────────────
# ROI 대표 샘플 선택 (3개: 다른 배경 유형)
# ──────────────────────────────────────────────────────────────────────────────
# background_type 기준으로 diverse 선택; suitability_score 내림차순
selected_rois = []
seen_bg_types = set()
target_bgs = ["smooth", "vertical_stripe", "complex_pattern",
              "horizontal_stripe", "textured"]

sort_col = "suitability_score" if "suitability_score" in df_img.columns else df_img.columns[0]
for _, row in df_img.sort_values(sort_col, ascending=False).iterrows():
    bg = row.get("background_type", cell_bg[
        min(int(row["roi_bbox"][1] // GRID_SIZE), NR - 1),
        min(int(row["roi_bbox"][0] // GRID_SIZE), NC - 1),
    ])
    if bg not in seen_bg_types:
        seen_bg_types.add(bg)
        selected_rois.append(row)
    if len(selected_rois) >= 3:
        break

# 부족하면 상위 3개로 채우기
if len(selected_rois) < 3:
    for _, row in df_img.sort_values(sort_col, ascending=False).iterrows():
        if row.name not in [r.name for r in selected_rois]:
            selected_rois.append(row)
        if len(selected_rois) >= 3:
            break

print(f"Selected {len(selected_rois)} representative ROIs:")
for i, r in enumerate(selected_rois):
    print(f"  [{i}] bbox={r['roi_bbox']}  bg={r.get('background_type','-')}  "
          f"score={r.get('suitability_score', 0):.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 생성
# ──────────────────────────────────────────────────────────────────────────────
FIG_W = 16.0
FIG_H = 8.5
ASPECT = IMG_W / IMG_H   # 1600/256 ≈ 6.25

# 행 비율: [원본 이미지] : [배경 유형 그리드] : [ROI 패치] = 2.2 : 2.4 : 3.5
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
gs_main = gridspec.GridSpec(
    3, 1,
    figure=fig,
    height_ratios=[2.2, 2.4, 3.5],
    hspace=0.38,
    top=0.91, bottom=0.04, left=0.04, right=0.98,
)

suptitle_kw = dict(fontsize=11, fontweight="bold", color="#1A237E")

# ══════════════════════════════════════════════════════════════════════════════
# ROW A: 원본 이미지 + 그리드 라인 + ROI 박스
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs_main[0])
ax_a.imshow(steel_gray, cmap="gray", aspect="auto",
            extent=[0, IMG_W, IMG_H, 0], vmin=0, vmax=255)

# 그리드 라인 (cyan)
for c in range(0, IMG_W + 1, GRID_SIZE):
    ax_a.axvline(c, color="cyan", linewidth=0.5, alpha=0.65)
for r in range(0, IMG_H + 1, GRID_SIZE):
    ax_a.axhline(r, color="cyan", linewidth=0.5, alpha=0.65)

# 전체 ROI 박스 (흐리게)
if not df_img.empty:
    for _, row in df_img.iterrows():
        x1, y1, x2, y2 = [int(v) for v in row["roi_bbox"]]
        sc = row.get("suitability_score", 0.5)
        alpha = 0.25 + 0.3 * sc
        ax_a.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0.8, edgecolor="white", facecolor="none", alpha=alpha,
        ))

# 선택된 3개 ROI 하이라이트
for i, row in enumerate(selected_rois):
    x1, y1, x2, y2 = [int(v) for v in row["roi_bbox"]]
    bg = row.get("background_type",
                 cell_bg[min(int(y1 // GRID_SIZE), NR-1),
                         min(int(x1 // GRID_SIZE), NC-1)])
    col = ROI_BOX_COLORS.get(bg, "#FFFFFF")
    label = f"ROI #{i+1}"
    ax_a.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2.2, edgecolor=col, facecolor=col, alpha=0.15,
    ))
    ax_a.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2.2, edgecolor=col, facecolor="none",
    ))
    ax_a.text(
        (x1 + x2) / 2, y1 - 6, label,
        ha="center", va="bottom", fontsize=7, fontweight="bold", color=col,
        bbox=dict(fc="black", alpha=0.55, pad=1.5, boxstyle="round,pad=0.25"),
    )

ax_a.set_xlim(0, IMG_W)
ax_a.set_ylim(IMG_H + 12, -20)
ax_a.set_title(
    f"(A)  Original Image: {DEMO_IMAGE_ID}  [{IMG_W}×{IMG_H} px]  │  "
    f"Sliding Grid: {GRID_SIZE}px → {NC}×{NR} cells  │  "
    f"{len(df_img)} ROI windows",
    fontsize=8.5, fontweight="bold", color="#1A237E", pad=5,
)
ax_a.set_xlabel("pixel x", fontsize=7.5)
ax_a.set_yticks([])
ax_a.tick_params(axis="x", labelsize=7)

# 범례
leg_a = [
    mpatches.Patch(ec="cyan", fc="none", lw=0.8, label="64 px grid"),
    mpatches.Patch(ec="white", fc="none", lw=0.8, alpha=0.6, label="candidate ROI window"),
] + [
    mpatches.Patch(ec=ROI_BOX_COLORS.get(r.get("background_type","smooth"), "#fff"),
                   fc="none", lw=2.0,
                   label=f"ROI #{i+1}: {r.get('background_type','-').replace('_',' ')}")
    for i, r in enumerate(selected_rois)
]
ax_a.legend(handles=leg_a, loc="upper right", fontsize=6.5,
            framealpha=0.82, ncol=3)

# ══════════════════════════════════════════════════════════════════════════════
# ROW B: 배경 유형 분류 그리드
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs_main[1])

# 배경 이미지 (dim)
ax_b.imshow(steel_gray, cmap="gray", aspect="auto",
            extent=[0, IMG_W, IMG_H, 0], vmin=0, vmax=255, alpha=0.45)

# 배경 유형 컬러 셀
for r in range(NR):
    for c in range(NC):
        bg_type = cell_bg[r, c]
        meta = BG_META.get(bg_type, BG_DEFAULT)
        score = cell_score[r, c]
        x0, y0 = c * GRID_SIZE, r * GRID_SIZE

        # 반투명 컬러 채우기
        color_hex = meta["color"]
        ax_b.add_patch(Rectangle(
            (x0, y0), GRID_SIZE, GRID_SIZE,
            facecolor=color_hex, edgecolor="white",
            linewidth=0.55, alpha=0.52,
        ))
        # 약어 라벨
        ax_b.text(
            x0 + GRID_SIZE / 2, y0 + GRID_SIZE * 0.36,
            meta["abbr"],
            ha="center", va="center", fontsize=5.0,
            fontweight="bold", color=meta["text_col"],
        )
        # 점수
        ax_b.text(
            x0 + GRID_SIZE / 2, y0 + GRID_SIZE * 0.72,
            f"{score:.2f}",
            ha="center", va="center", fontsize=4.5,
            color=meta["text_col"], alpha=0.9,
        )

# 선택 ROI 강조
for i, row in enumerate(selected_rois):
    x1, y1, x2, y2 = [int(v) for v in row["roi_bbox"]]
    bg = row.get("background_type",
                 cell_bg[min(int(y1 // GRID_SIZE), NR-1),
                         min(int(x1 // GRID_SIZE), NC-1)])
    col = ROI_BOX_COLORS.get(bg, "#ffffff")
    gc = int(x1 // GRID_SIZE)
    gr = int(y1 // GRID_SIZE)
    ax_b.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2.0, edgecolor=col, facecolor="none", zorder=5,
    ))
    ax_b.text(
        x2 + 4, (y1 + y2) / 2,
        f"ROI #{i+1}\n{bg.replace('_',' ')}\nGrid[{gr},{gc}]",
        va="center", fontsize=6, fontweight="bold", color=col,
        bbox=dict(fc="black", alpha=0.6, pad=2, boxstyle="round,pad=0.3"),
        zorder=6,
    )

ax_b.set_xlim(0, IMG_W)
ax_b.set_ylim(IMG_H + 12, -20)
ax_b.set_title(
    f"(B)  Background Type Classification  "
    f"({NC}×{NR} = {NC*NR} cells)  │  "
    f"Sobel direction + edge density analysis  │  Score = defect density proxy",
    fontsize=8.5, fontweight="bold", color="#1A237E", pad=5,
)
ax_b.set_xlabel("pixel x", fontsize=7.5)
ax_b.set_yticks([])
ax_b.tick_params(axis="x", labelsize=7)

# 범례
leg_b = [
    mpatches.Patch(fc=v["color"], alpha=0.7,
                   label=f"{v['abbr']}: {k.replace('_',' ')}")
    for k, v in BG_META.items()
]
ax_b.legend(handles=leg_b, loc="upper right", fontsize=6.5,
            framealpha=0.88, ncol=5)

# ══════════════════════════════════════════════════════════════════════════════
# ROW C: 추출된 ROI 패치 (3개, 각 256×256)
# ══════════════════════════════════════════════════════════════════════════════
n_rois = len(selected_rois)
# 각 패치 옆에 메타 텍스트 공간 → 2 서브컬럼 per ROI
gs_c = gridspec.GridSpecFromSubplotSpec(
    1, n_rois * 2,
    subplot_spec=gs_main[2],
    width_ratios=[3, 1] * n_rois,
    wspace=0.05,
)

for i, roi_row in enumerate(selected_rois):
    x1, y1, x2, y2 = [int(v) for v in roi_row["roi_bbox"]]
    bg_type = roi_row.get(
        "background_type",
        cell_bg[min(int(y1 // GRID_SIZE), NR-1),
                min(int(x1 // GRID_SIZE), NC-1)],
    )
    col = ROI_BOX_COLORS.get(bg_type, "#ffffff")
    meta = BG_META.get(bg_type, BG_DEFAULT)

    # 패치 추출 (bbox 값이 float 로 저장된 경우 int 변환)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    patch = steel_gray[y1:y2, x1:x2]
    mask_patch = full_mask[y1:y2, x1:x2]

    # ── 이미지 패널 ──
    ax_patch = fig.add_subplot(gs_c[0, i * 2])
    ax_patch.imshow(patch, cmap="gray", aspect="auto", vmin=0, vmax=255)

    # 결함 마스크 오버레이
    if mask_patch.any():
        overlay_rgb = np.stack([patch] * 3, axis=-1).astype(np.float32)
        overlay_rgb[mask_patch > 0, 0] = 255
        overlay_rgb[mask_patch > 0, 1] = 80
        overlay_rgb[mask_patch > 0, 2] = 80
        ax_patch.imshow(
            overlay_rgb.astype(np.uint8), aspect="auto", alpha=0.45,
        )
        # 결함 bbox (defect_bbox)
        if "defect_bbox" in roi_row and roi_row["defect_bbox"] is not None:
            try:
                dbbox = safe_bbox(roi_row["defect_bbox"])
                if dbbox:
                    dx1, dy1, dx2, dy2 = dbbox
                    # defect_bbox는 원본 이미지 좌표 → 패치 로컬 좌표 변환
                    dx1 -= x1; dx2 -= x1; dy1 -= y1; dy2 -= y1
                    ax_patch.add_patch(Rectangle(
                        (dx1, dy1), dx2 - dx1, dy2 - dy1,
                        linewidth=1.5, edgecolor="#FF5252", facecolor="none",
                    ))
            except Exception:
                pass

    # ROI 테두리 (배경 유형 색상)
    for sp in ax_patch.spines.values():
        sp.set_linewidth(3.0)
        sp.set_edgecolor(col)
    ax_patch.set_xticks([]); ax_patch.set_yticks([])

    gc_col = int(x1 // GRID_SIZE)
    gc_row = int(y1 // GRID_SIZE)
    score  = roi_row.get("suitability_score", 0.0)
    cls_id = int(roi_row.get("class_id", 0))
    subtype = roi_row.get("defect_subtype", "-")
    stab   = roi_row.get("stability_score", 0.0)

    ax_patch.set_title(
        f"ROI #{i+1}",
        fontsize=8.5, fontweight="bold", color=col, pad=3,
        bbox=dict(fc="black", alpha=0.55, pad=2, boxstyle="round,pad=0.3"),
    )

    # ── 메타 텍스트 패널 ──
    ax_meta = fig.add_subplot(gs_c[0, i * 2 + 1])
    ax_meta.axis("off")

    lines = [
        ("Background",  bg_type.replace("_", " "), meta["color"]),
        ("Class",       f"Class {cls_id}",           "#E0E0E0"),
        ("Subtype",     subtype.replace("_", " "),    "#E0E0E0"),
        ("Suitability", f"{score:.3f}",               "#FFD600" if score >= 0.75 else "#A5D6A7"),
        ("Stability",   f"{stab:.3f}",                "#E0E0E0"),
        ("Grid",        f"[{gc_row}, {gc_col}]",      "#E0E0E0"),
        ("ROI size",    f"{x2-x1}×{y2-y1} px",        "#E0E0E0"),
    ]

    y_pos = 0.97
    ax_meta.text(0.05, y_pos, f"— ROI #{i+1} —",
                 transform=ax_meta.transAxes,
                 fontsize=7, fontweight="bold", color=col,
                 va="top")
    y_pos -= 0.14

    for label, value, vcolor in lines:
        ax_meta.text(0.05, y_pos, f"{label}:",
                     transform=ax_meta.transAxes,
                     fontsize=6, color="#9E9E9E", va="top")
        ax_meta.text(0.05, y_pos - 0.08, value,
                     transform=ax_meta.transAxes,
                     fontsize=6.5, color=vcolor, va="top", fontweight="bold",
                     bbox=dict(fc="#1E1E1E", alpha=0.0, pad=0))
        y_pos -= 0.18

    ax_meta.set_facecolor("#111111")
    ax_meta.patch.set_alpha(0.0)

# 행 C 타이틀 (별도 텍스트)
fig.text(
    0.5, gs_main[2].get_position(fig).y1 + 0.015,
    "(C)  Extracted ROI Patches (256×256 px)  │  "
    "Red overlay = RLE defect mask  │  "
    "Red box = defect_bbox  │  "
    "Border color = background type",
    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1A237E",
)

# ── 전체 타이틀 ──
fig.suptitle(
    f"CASDA Stage A — ROI Extraction: Sliding Grid & Background Classification  "
    f"[{DEMO_IMAGE_ID}]",
    fontsize=11, fontweight="bold", color="#0D47A1", y=0.96,
)

# ══════════════════════════════════════════════════════════════════════════════
# 저장
# ══════════════════════════════════════════════════════════════════════════════
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FILE, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved → {OUT_FILE}")
plt.show()
