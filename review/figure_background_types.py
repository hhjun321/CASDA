"""
figure_background_types.py  —  Stage A §4: Background Type Classification
==========================================================================
논문 §4 "배경 유형 분류 (BackgroundAnalyzer)" 시각화.

Layout  (3 rows × 5 columns):
  각 열 = 배경 유형 1개  (smooth / vertical_stripe / horizontal_stripe /
                           textured / complex_pattern)
  Row 0 : 실제 강재 이미지 패치 (256×256, grayscale)
  Row 1 : Sobel 방향 분석 시각화
            vertical_stripe   → Sobel-X (수평 기울기, 세로선 강조)
            horizontal_stripe → Sobel-Y (수직 기울기, 가로선 강조)
            기타              → √(Sx²+Sy²) 복합 에지
          false-color: 방향에 따라 R/G/B 채색
  Row 2 : 분류 의사결정 바 차트
            Variance / Edge Density / Sx ratio / Sy ratio
          + 판정 임계값 점선 + 분류 결과 배지

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_background_types.py
"""

import ast, sys
from pathlib import Path

try:
    PROJ_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJ_ROOT = Path("/content/CASDA")

sys.path.insert(0, str(PROJ_ROOT))

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
TRAIN_CSV    = DRIVE_DATA / "train.csv"
ROI_META     = DRIVE_DATA / "roi_patches/roi_metadata.csv"

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "background_types.png"

IMG_H, IMG_W = 256, 1600

# ──────────────────────────────────────────────────────────────────────────────
# 배경 유형 스타일
# ──────────────────────────────────────────────────────────────────────────────
BG_TYPES = [
    "smooth",
    "vertical_stripe",
    "horizontal_stripe",
    "textured",
    "complex_pattern",
]

BG_META = {
    "smooth":            dict(color="#1E88E5", label="smooth",
                              rule="Var<200 & EdgeDens<0.02",
                              sobel_mode="magnitude", sobel_label="√(Sx²+Sy²) ×0.3"),
    "vertical_stripe":   dict(color="#43A047", label="vertical\nstripe",
                              rule="Sx/tot>0.65 & EdgeDens>0.02",
                              sobel_mode="x",     sobel_label="Sobel-X (→ vertical lines)"),
    "horizontal_stripe": dict(color="#FB8C00", label="horizontal\nstripe",
                              rule="Sy/tot>0.65 & EdgeDens>0.02",
                              sobel_mode="y",     sobel_label="Sobel-Y (→ horizontal lines)"),
    "textured":          dict(color="#8E24AA", label="textured",
                              rule="Var>500 or EdgeDens>0.05",
                              sobel_mode="magnitude", sobel_label="√(Sx²+Sy²)"),
    "complex_pattern":   dict(color="#E53935", label="complex\npattern",
                              rule="EdgeDens>0.15",
                              sobel_mode="magnitude", sobel_label="√(Sx²+Sy²)"),
}

# false-color map per sobel mode
SOBEL_CMAP = {
    "x":         (0.15, 1.0, 0.15),   # green for Sobel-X
    "y":         (1.0, 0.65, 0.0),    # orange for Sobel-Y
    "magnitude": (0.3, 0.7, 1.0),     # blue for magnitude
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
        return [int(v) for v in ast.literal_eval(str(val))]
    except Exception:
        return None


def compute_bg_metrics(patch: np.ndarray):
    """배경 분류에 사용되는 4개 지표 계산."""
    p = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA)
    variance     = float(np.var(p.astype(np.float32)))
    edges        = cv2.Canny(p, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    sx = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(p, cv2.CV_64F, 0, 1, ksize=3)
    ex = float(np.mean(np.abs(sx)))
    ey = float(np.mean(np.abs(sy)))
    tot = ex + ey + 1e-7

    return dict(
        variance=variance,
        edge_density=edge_density,
        sx_ratio=ex / tot,
        sy_ratio=ey / tot,
    )


def classify_bg(m: dict) -> str:
    if m["variance"] < 200 and m["edge_density"] < 0.02:
        return "smooth"
    if m["sx_ratio"] > 0.65 and m["edge_density"] > 0.02:
        return "vertical_stripe"
    if m["sy_ratio"] > 0.65 and m["edge_density"] > 0.02:
        return "horizontal_stripe"
    if m["edge_density"] > 0.15:
        return "complex_pattern"
    if m["variance"] > 500 or m["edge_density"] > 0.05:
        return "textured"
    return "smooth"


def make_sobel_vis(patch: np.ndarray, mode: str) -> np.ndarray:
    """Sobel false-color visualization."""
    p = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_LINEAR) \
        if patch.shape[0] < 200 else patch.copy()
    sx = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(p, cv2.CV_64F, 0, 1, ksize=3)

    if mode == "x":
        channel = np.abs(sx)
    elif mode == "y":
        channel = np.abs(sy)
    else:
        channel = np.sqrt(sx**2 + sy**2)
        if mode == "magnitude_smooth":
            channel *= 0.3

    norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32) / 255.0
    rgb_tint = SOBEL_CMAP[mode]
    vis = np.zeros((*norm.shape, 3), dtype=np.float32)
    vis[:, :, 0] = norm * rgb_tint[0]
    vis[:, :, 1] = norm * rgb_tint[1]
    vis[:, :, 2] = norm * rgb_tint[2]
    # 배경에 원본 이미지 희미하게 합성
    gray_norm = p.astype(np.float32) / 255.0
    vis = vis * 0.7 + np.stack([gray_norm * 0.3] * 3, axis=-1)
    return np.clip(vis, 0, 1)


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
col_rle = [c for c in train_df.columns if "encoded" in c.lower() or "pixel" in c.lower()][0]

# ──────────────────────────────────────────────────────────────────────────────
# 배경 유형별 대표 패치 선택
#   • 결함 마스크 비중이 낮아 배경이 잘 보이는 ROI 우선
#   • 실제 분류 로직으로 교차 검증
# ──────────────────────────────────────────────────────────────────────────────
bg_col   = "background_type"   if "background_type"   in df_all.columns else None
suit_col = "suitability_score" if "suitability_score" in df_all.columns else df_all.columns[0]

selected = {}

for bgtype in BG_TYPES:
    if bg_col:
        pool = df_all[df_all[bg_col] == bgtype]
    else:
        pool = df_all

    # stability 높은 순 (배경이 명확한 이미지)
    s_col = "stability_score" if "stability_score" in pool.columns else suit_col
    pool = pool.sort_values(s_col, ascending=False)

    for _, row in pool.iterrows():
        x1, y1, x2, y2 = row["roi_bbox"]
        img_id = str(row.get("image_id", ""))
        img_path = TRAIN_IMAGES / img_id
        if not img_path.exists():
            continue

        gray_full = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray_full is None:
            continue

        # 결함 마스크 비중 계산
        rle_rows = train_df[train_df[col_img] == img_id]
        full_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for _, rr in rle_rows.iterrows():
            full_mask = np.maximum(full_mask, rle_to_mask(rr[col_rle]))
        mask_patch = full_mask[y1:y2, x1:x2]
        defect_ratio = mask_patch.mean()
        if defect_ratio > 0.15:     # 결함이 너무 많으면 배경 보기 어려움
            continue

        roi_gray = gray_full[y1:y2, x1:x2]
        metrics  = compute_bg_metrics(roi_gray)
        predicted = classify_bg(metrics)

        # 교차 검증: bg_col 없으면 predicted 기준, 있으면 metadata 기준
        if bg_col is None and predicted != bgtype:
            continue

        selected[bgtype] = dict(
            img_id=img_id, roi_gray=roi_gray, mask_patch=mask_patch,
            metrics=metrics, predicted=predicted,
            stability=float(row.get("stability_score", 0.0)),
        )
        print(f"  [{bgtype}] {img_id} | predicted={predicted} | "
              + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        break

    if bgtype not in selected:
        print(f"  WARNING: '{bgtype}' 패치를 찾지 못했습니다")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 생성
# ──────────────────────────────────────────────────────────────────────────────
N_COLS = len(BG_TYPES)
FIG_W, FIG_H = 15.0, 8.5

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
fig.suptitle(
    "CASDA Stage A §4 — Background Type Classification  (BackgroundAnalyzer)\n"
    "Sobel direction analysis + edge density → 5 background categories",
    fontsize=10, fontweight="bold", color="#0D47A1", y=0.99,
)

# 3행 × 5열 (상단 레이블 행 포함 → 실질 4행)
gs = gridspec.GridSpec(
    4, N_COLS,
    figure=fig,
    height_ratios=[0.18, 1.3, 1.3, 1.05],
    hspace=0.22, wspace=0.14,
    top=0.91, bottom=0.04, left=0.02, right=0.99,
)

ROW_LABELS = [
    None,
    "(A)  Original Patch  (256×256 px)",
    "(B)  Sobel Edge Analysis  (false-color)",
    "(C)  Classification Metrics",
]
for ri, rl in enumerate(ROW_LABELS):
    if rl:
        fig.text(
            0.015, gs[ri, 0].get_position(fig).y1 + 0.008,
            rl, fontsize=8.5, fontweight="bold", color="#37474F",
        )

for col_i, bgtype in enumerate(BG_TYPES):
    meta  = BG_META[bgtype]
    color = meta["color"]
    entry = selected.get(bgtype)

    # ── 열 헤더 (행 0) ──
    ax_hdr = fig.add_subplot(gs[0, col_i])
    ax_hdr.axis("off")
    ax_hdr.set_facecolor(color)
    ax_hdr.patch.set_alpha(0.85)
    ax_hdr.text(0.5, 0.5, meta["label"],
                transform=ax_hdr.transAxes,
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white")

    if entry is None:
        for ri in [1, 2, 3]:
            ax_x = fig.add_subplot(gs[ri, col_i])
            ax_x.axis("off")
            ax_x.text(0.5, 0.5, "No data", ha="center", va="center",
                      fontsize=9, color="#9E9E9E", transform=ax_x.transAxes)
        continue

    roi_gray  = entry["roi_gray"]
    mask_p    = entry["mask_patch"]
    mets      = entry["metrics"]
    predicted = entry["predicted"]

    # ── Row 1: 원본 패치 ──
    ax_img = fig.add_subplot(gs[1, col_i])
    ax_img.imshow(roi_gray, cmap="gray", vmin=0, vmax=255, aspect="equal")
    # 결함 마스크 (있으면 연하게)
    if mask_p.any():
        rgba = np.zeros((*mask_p.shape, 4), dtype=np.float32)
        rgba[mask_p > 0] = [1.0, 0.3, 0.3, 0.35]
        ax_img.imshow(rgba, aspect="equal")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_linewidth(2.5); sp.set_edgecolor(color)
    ax_img.text(
        2, roi_gray.shape[0] - 4,
        entry["img_id"][:16],
        fontsize=5, color="white",
        bbox=dict(fc="black", alpha=0.6, pad=1, boxstyle="round,pad=0.2"),
    )

    # ── Row 2: Sobel 시각화 ──
    ax_sob = fig.add_subplot(gs[2, col_i])
    sob_vis = make_sobel_vis(roi_gray, meta["sobel_mode"])
    ax_sob.imshow(sob_vis, aspect="equal")
    ax_sob.set_xticks([]); ax_sob.set_yticks([])
    for sp in ax_sob.spines.values():
        sp.set_linewidth(2.0); sp.set_edgecolor(color)
    ax_sob.text(
        2, sob_vis.shape[0] - 4,
        meta["sobel_label"],
        fontsize=5.5, color="white",
        bbox=dict(fc="black", alpha=0.62, pad=1.5, boxstyle="round,pad=0.25"),
    )

    # ── Row 3: 분류 의사결정 바 차트 ──
    ax_bar = fig.add_subplot(gs[3, col_i])

    metric_names = ["Variance\n(÷3000)", "Edge Density", "Sx ratio", "Sy ratio"]
    raw_vals = [
        mets["variance"],
        mets["edge_density"],
        mets["sx_ratio"],
        mets["sy_ratio"],
    ]
    # 정규화 표시 (0-1 스케일)
    disp_vals = [
        min(mets["variance"] / 3000.0, 1.0),
        min(mets["edge_density"] / 0.3, 1.0),
        mets["sx_ratio"],
        mets["sy_ratio"],
    ]
    bar_colors = ["#5C6BC0", "#26A69A", "#43A047", "#FB8C00"]

    y_pos = np.arange(len(metric_names))
    bars = ax_bar.barh(y_pos, disp_vals, color=bar_colors, height=0.5,
                       edgecolor="white", linewidth=0.8)

    # 임계값 점선
    thresholds_disp = [
        200 / 3000.0,  # Variance < 200
        0.02 / 0.3,    # EdgeDens 0.02 (smooth) or 0.15 (complex)
        0.65,          # Sx ratio > 0.65
        0.65,          # Sy ratio > 0.65
    ]
    for yi, thr in enumerate(thresholds_disp):
        ax_bar.axvline(
            thr,
            ymin=(yi - 0.3) / len(metric_names),
            ymax=(yi + 0.3) / len(metric_names),
            color="white", lw=1.2, linestyle="--", alpha=0.8,
        )

    # 값 레이블
    fmt = [
        f"{mets['variance']:.0f}",
        f"{mets['edge_density']:.3f}",
        f"{mets['sx_ratio']:.2f}",
        f"{mets['sy_ratio']:.2f}",
    ]
    for yi, (b, vd, lbl) in enumerate(zip(bars, disp_vals, fmt)):
        ax_bar.text(
            min(vd + 0.02, 0.97), yi,
            lbl, va="center", fontsize=7,
            fontweight="bold", color=bar_colors[yi],
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(metric_names, fontsize=6.5)
    ax_bar.set_xlim(0, 1.08)
    ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.tick_params(axis="x", labelsize=6)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.3)

    # 분류 결과 배지
    match = (predicted == bgtype)
    badge_col = "#43A047" if match else "#E53935"
    badge_sym = "✓" if match else "✗"
    ax_bar.text(
        0.5, -0.28,
        f"{badge_sym}  {predicted.replace('_',' ')}",
        transform=ax_bar.transAxes, ha="center", va="bottom",
        fontsize=7, fontweight="bold", color=badge_col,
        bbox=dict(fc="#F5F5F5", alpha=0.85, pad=2, boxstyle="round,pad=0.3"),
    )
    ax_bar.text(
        0.5, -0.44,
        f"Rule: {meta['rule']}",
        transform=ax_bar.transAxes, ha="center", va="bottom",
        fontsize=6, color="#757575", style="italic",
    )

# ──────────────────────────────────────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FILE, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved → {OUT_FILE}")
plt.show()
