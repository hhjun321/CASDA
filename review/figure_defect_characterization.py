"""
figure_defect_characterization.py  —  Stage A §3: Defect Shape Analysis
=========================================================================
논문 §3 "결함 형상 분석 (DefectCharacterizer)" 시각화.

Layout  (5 rows × 3 columns):
  각 행 = 결함 서브타입 1개  (linear_scratch / elongated / compact_blob /
                               irregular / general)
  Col 0 : 실제 ROI (grayscale) + 결함 마스크 오버레이 (red) + bbox
  Col 1 : 기하학적 분석 오버레이
            linear_scratch → PCA 주축 방향 (고유벡터 ellipse)
            compact_blob   → 볼록 껍질(Convex Hull) + 내부 채우기
            elongated      → 바운딩 박스 + 장축/단축 길이 표시
            irregular      → 결함 윤곽선 + 볼록 껍질 갭 강조
            general        → 바운딩 박스 + 무게중심
  Col 2 : 4개 지표 수평 바 차트 (Linearity / Solidity / Extent / Aspect Ratio)

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_defect_characterization.py
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
from matplotlib.patches import FancyArrowPatch, Ellipse, Rectangle
from skimage import measure

# ──────────────────────────────────────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
TRAIN_CSV    = DRIVE_DATA / "train.csv"
ROI_META     = DRIVE_DATA / "roi_patches/roi_metadata.csv"

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "defect_characterization.png"

IMG_H, IMG_W = 256, 1600

# ──────────────────────────────────────────────────────────────────────────────
# 서브타입 정의 및 스타일
# ──────────────────────────────────────────────────────────────────────────────
SUBTYPES = [
    "linear_scratch",
    "elongated",
    "compact_blob",
    "irregular",
    "general",
]

SUBTYPE_META = {
    "linear_scratch": dict(
        label="linear_scratch\n(Linearity > 0.7)",
        color="#1565C0",
        rule="Linearity > 0.7",
        viz="PCA principal axis",
    ),
    "elongated": dict(
        label="elongated\n(Aspect > 3.0)",
        color="#2E7D32",
        rule="Aspect Ratio > 3.0",
        viz="Bounding box + axes",
    ),
    "compact_blob": dict(
        label="compact_blob\n(Solidity > 0.8)",
        color="#E65100",
        rule="Solidity > 0.8",
        viz="Convex hull overlay",
    ),
    "irregular": dict(
        label="irregular\n(none of above)",
        color="#6A1B9A",
        rule="fallback",
        viz="Contour + hull gap",
    ),
    "general": dict(
        label="general\n(default)",
        color="#546E7A",
        rule="default",
        viz="Centroid + bbox",
    ),
}

METRIC_COLORS = {
    "Linearity":    "#1E88E5",
    "Solidity":     "#43A047",
    "Extent":       "#FB8C00",
    "Aspect Ratio": "#E53935",
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


def compute_metrics(mask_patch):
    """마스크로부터 4개 지표 계산."""
    labeled = measure.label(mask_patch, connectivity=2)
    regions = measure.regionprops(labeled)
    if not regions:
        return None, None
    reg = max(regions, key=lambda r: r.area)

    coords = reg.coords
    if len(coords) < 3:
        linearity = 0.0
    else:
        c = coords - coords.mean(axis=0)
        cov = np.cov(c.T)
        eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
        linearity = float((eigs[0] - eigs[1]) / eigs[0]) if eigs[0] > 1e-6 else 0.0

    return {
        "Linearity":    linearity,
        "Solidity":     float(reg.solidity),
        "Extent":       float(reg.extent),
        "Aspect Ratio": min(float(reg.major_axis_length / max(reg.minor_axis_length, 1e-6)), 10.0),
    }, reg


def draw_viz(ax, patch_gray, mask_patch, reg, subtype):
    """Col 1: 기하학적 분석 오버레이."""
    ax.imshow(patch_gray, cmap="gray", vmin=0, vmax=255, aspect="equal")

    # 공통: 결함 마스크 윤곽선
    contours, _ = cv2.findContours(
        mask_patch.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    for cnt in contours:
        pts = cnt[:, 0, :]   # (N, 2) x, y
        ax.plot(
            np.append(pts[:, 0], pts[0, 0]),
            np.append(pts[:, 1], pts[0, 1]),
            color="#FF5252", lw=1.2, alpha=0.85,
        )

    cy, cx = reg.centroid   # (row, col) → y, x

    if subtype == "linear_scratch":
        # PCA 주축 방향 화살표
        coords = reg.coords
        c = coords - coords.mean(axis=0)
        cov = np.cov(c.T)
        eigs, evecs = np.linalg.eigh(cov)
        idx = np.argsort(eigs)[::-1]
        evec1 = evecs[:, idx[0]]   # (row, col)
        half = min(reg.major_axis_length / 2, 80)
        ax.annotate(
            "", xy=(cx + evec1[1] * half, cy + evec1[0] * half),
            xytext=(cx - evec1[1] * half, cy - evec1[0] * half),
            arrowprops=dict(arrowstyle="<->", color="#FDD835", lw=2.0),
        )
        # 단축
        evec2 = evecs[:, idx[1]]
        half2 = min(reg.minor_axis_length / 2, 40)
        ax.annotate(
            "", xy=(cx + evec2[1] * half2, cy + evec2[0] * half2),
            xytext=(cx - evec2[1] * half2, cy - evec2[0] * half2),
            arrowprops=dict(arrowstyle="<->", color="#80CBC4", lw=1.5),
        )
        ax.text(3, 10, "Major axis (λ₁)", fontsize=6, color="#FDD835")
        ax.text(3, 18, "Minor axis (λ₂)", fontsize=6, color="#80CBC4")
        ax.text(3, 26, f"Lin=(λ₁-λ₂)/λ₁", fontsize=6, color="white", style="italic")

    elif subtype == "compact_blob":
        # 볼록 껍질
        if len(contours) > 0:
            all_pts = np.vstack(contours)
            hull = cv2.convexHull(all_pts)
            hull_pts = hull[:, 0, :]
            ax.fill(
                np.append(hull_pts[:, 0], hull_pts[0, 0]),
                np.append(hull_pts[:, 1], hull_pts[0, 1]),
                color="#FDD835", alpha=0.22,
            )
            ax.plot(
                np.append(hull_pts[:, 0], hull_pts[0, 0]),
                np.append(hull_pts[:, 1], hull_pts[0, 1]),
                color="#FDD835", lw=1.5, linestyle="--", alpha=0.9,
            )
        ax.text(3, 10, "Convex Hull", fontsize=6.5, color="#FDD835")
        ax.text(3, 18, "Solidity = Area / Hull Area", fontsize=6, color="white", style="italic")

    elif subtype == "elongated":
        minr, minc, maxr, maxc = reg.bbox
        ax.add_patch(Rectangle(
            (minc, minr), maxc - minc, maxr - minr,
            linewidth=1.5, edgecolor="#FDD835", facecolor="none",
        ))
        # 장축 길이 표시
        ax.annotate(
            "", xy=(maxc, cy), xytext=(minc, cy),
            arrowprops=dict(arrowstyle="<->", color="#FDD835", lw=1.5),
        )
        ax.annotate(
            "", xy=(cx, maxr), xytext=(cx, minr),
            arrowprops=dict(arrowstyle="<->", color="#80CBC4", lw=1.5),
        )
        ax.text(minc, cy - 5, f"W={maxc-minc}px", fontsize=6, color="#FDD835", ha="left")
        ax.text(cx + 2, (minr + maxr)//2, f"H={maxr-minr}px", fontsize=6, color="#80CBC4")
        ax.text(3, patch_gray.shape[0] - 10, f"AR = W/H", fontsize=6, color="white", style="italic")

    elif subtype == "irregular":
        # 윤곽선 + 볼록 껍질 갭 (붉은색)
        if len(contours) > 0:
            all_pts = np.vstack(contours)
            hull = cv2.convexHull(all_pts)
            hull_pts = hull[:, 0, :]
            ax.plot(
                np.append(hull_pts[:, 0], hull_pts[0, 0]),
                np.append(hull_pts[:, 1], hull_pts[0, 1]),
                color="#FDD835", lw=1.5, linestyle="--", alpha=0.9,
            )
        ax.text(3, 10, "Hull (dashed) vs Contour (solid)", fontsize=6, color="#FDD835")
        ax.text(3, 18, "Gap = low Solidity", fontsize=6, color="white", style="italic")

    else:  # general
        minr, minc, maxr, maxc = reg.bbox
        ax.add_patch(Rectangle(
            (minc, minr), maxc - minc, maxr - minr,
            linewidth=1.5, edgecolor="#FDD835", facecolor="none",
        ))
        ax.plot(cx, cy, "o", color="#FDD835", ms=6, markeredgecolor="white", mew=1)
        ax.text(3, 10, "Centroid + BBox", fontsize=6.5, color="#FDD835")

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
        sp.set_edgecolor(SUBTYPE_META[subtype]["color"])


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
# 서브타입별 대표 ROI 선택
# ──────────────────────────────────────────────────────────────────────────────
sub_col  = "defect_subtype"  if "defect_subtype"  in df_all.columns else None
suit_col = "suitability_score" if "suitability_score" in df_all.columns else df_all.columns[0]

# subtype → 정렬 기준 지표
SORT_BY = {
    "linear_scratch": "linearity",
    "elongated":      "aspect_ratio",
    "compact_blob":   "solidity",
    "irregular":      "extent",
    "general":        suit_col,
}

# elongated 는 얇아서 픽셀 수가 적음 → 임계를 낮게
MIN_MASK_PX = {
    "linear_scratch": 30,
    "elongated":      10,
    "compact_blob":   30,
    "irregular":      30,
    "general":        20,
}

def classify_subtype(m: dict, primary: str) -> str:
    """
    지표→서브타입 분류.  primary 를 우선 평가하여
    elongated 가 linear_scratch 에 가로채이지 않도록 함.
    """
    checks = {
        "elongated":      m["Aspect Ratio"] > 2.5,
        "linear_scratch": m["Linearity"]    > 0.7,
        "compact_blob":   m["Solidity"]     > 0.8,
        "irregular":      m["Linearity"] <= 0.4 and m["Solidity"] <= 0.7,
        "general":        True,
    }
    # primary 를 먼저 확인
    if checks.get(primary, False):
        return primary
    for name, cond in checks.items():
        if name != primary and cond:
            return name
    return "general"


def try_pool(pool_df, stype, min_px):
    """주어진 pool 에서 stype 에 맞는 ROI 하나를 찾아 반환. 없으면 None."""
    s_col = SORT_BY.get(stype, suit_col)
    if s_col not in pool_df.columns:
        s_col = suit_col
    if s_col not in pool_df.columns:
        s_col = pool_df.columns[0]

    for _, row in pool_df.sort_values(s_col, ascending=False).iterrows():
        bbox = row["roi_bbox"]
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        img_id   = str(row.get("image_id", ""))
        img_path = TRAIN_IMAGES / img_id
        if not img_path.exists():
            continue

        rle_rows = train_df[train_df[col_img] == img_id]
        if rle_rows.empty:
            continue
        full_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for _, rr in rle_rows.iterrows():
            full_mask = np.maximum(full_mask, rle_to_mask(rr[col_rle]))

        mask_patch = full_mask[y1:y2, x1:x2]
        if mask_patch.sum() < min_px:
            continue

        gray_full = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray_full is None:
            continue
        roi_gray = gray_full[y1:y2, x1:x2]

        metrics, reg = compute_metrics(mask_patch)
        if metrics is None:
            continue

        # sub_col 없을 때: 지표로 교차 검증 (primary 우선 평가)
        if sub_col is None:
            if classify_subtype(metrics, stype) != stype:
                continue

        return dict(
            img_id=img_id, roi_gray=roi_gray, mask_patch=mask_patch,
            metrics=metrics, reg=reg, row=row,
        )
    return None


selected = {}

for stype in SUBTYPES:
    min_px = MIN_MASK_PX[stype]
    result = None

    # 1차: metadata subtype 필터
    if sub_col:
        exact_pool = df_all[df_all[sub_col] == stype]
        result = try_pool(exact_pool, stype, min_px)

    # 2차: metadata 에 없거나 1차 실패 → 전체에서 지표 기준 탐색
    if result is None:
        result = try_pool(df_all, stype, min_px)

    # 3차: elongated 전용 추가 완화 (aspect_ratio > 2.0, min_px=5)
    if result is None and stype == "elongated":
        print(f"  [{stype}] 3차 완화 검색 중 (AR>2.0, min_px=5) ...")
        for _, row in df_all.iterrows():
            bbox = row["roi_bbox"]
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            img_id   = str(row.get("image_id", ""))
            img_path = TRAIN_IMAGES / img_id
            if not img_path.exists():
                continue
            rle_rows = train_df[train_df[col_img] == img_id]
            if rle_rows.empty:
                continue
            full_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            for _, rr in rle_rows.iterrows():
                full_mask = np.maximum(full_mask, rle_to_mask(rr[col_rle]))
            mask_patch = full_mask[y1:y2, x1:x2]
            if mask_patch.sum() < 5:
                continue
            gray_full = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray_full is None:
                continue
            metrics, reg = compute_metrics(mask_patch)
            if metrics is None or metrics["Aspect Ratio"] < 2.0:
                continue
            result = dict(
                img_id=img_id, roi_gray=gray_full[y1:y2, x1:x2],
                mask_patch=mask_patch, metrics=metrics, reg=reg, row=row,
            )
            break

    if result is not None:
        selected[stype] = result
        m = result["metrics"]
        print(f"  [{stype}] {result['img_id']} | "
              + "  ".join(f"{k}={v:.2f}" for k, v in m.items()))
    else:
        print(f"  WARNING: '{stype}' ROI를 찾지 못했습니다")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 생성
# ──────────────────────────────────────────────────────────────────────────────
N_ROWS = len(SUBTYPES)
FIG_W, FIG_H = 13.5, 10.5

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
fig.suptitle(
    "CASDA Stage A §3 — Defect Shape Analysis  (DefectCharacterizer)\n"
    "4 geometric metrics → 5 defect subtypes used for hint generation & prompt construction",
    fontsize=10, fontweight="bold", color="#0D47A1", y=0.995,
)

# 좌측 레이블 열 (0) + 이미지(1) + 기하분석(2) + 메트릭 바(3)
gs = gridspec.GridSpec(
    N_ROWS, 4,
    figure=fig,
    width_ratios=[0.65, 1.4, 1.4, 1.0],
    hspace=0.22, wspace=0.14,
    top=0.93, bottom=0.04, left=0.02, right=0.99,
)

# 컬럼 헤더 (첫 번째 행 위에 텍스트로)
COL_HEADERS = [
    None,
    "(A) ROI Image + Defect Mask",
    "(B) Geometric Analysis",
    "(C) Metric Values",
]

for row_i, stype in enumerate(SUBTYPES):
    meta = SUBTYPE_META[stype]
    color = meta["color"]
    entry = selected.get(stype)

    # ── 행 레이블 ──
    ax_lbl = fig.add_subplot(gs[row_i, 0])
    ax_lbl.axis("off")
    ax_lbl.set_facecolor("#F5F5F5")
    ax_lbl.patch.set_alpha(0.0)
    ax_lbl.text(0.5, 0.65, meta["label"].replace("\n", "\n"),
                transform=ax_lbl.transAxes,
                ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=color, rotation=0)
    ax_lbl.text(0.5, 0.22, f"Rule: {meta['rule']}",
                transform=ax_lbl.transAxes,
                ha="center", va="center",
                fontsize=6, color="#757575", style="italic")

    # 헤더 (첫 행만)
    if row_i == 0:
        for c_i, hdr in enumerate(COL_HEADERS):
            if hdr:
                fig.text(
                    gs[0, c_i].get_position(fig).x0 + 0.01,
                    gs[0, c_i].get_position(fig).y1 + 0.005,
                    hdr,
                    fontsize=8, fontweight="bold", color="#37474F",
                )

    if entry is None:
        for ci in [1, 2, 3]:
            ax_x = fig.add_subplot(gs[row_i, ci])
            ax_x.axis("off")
            ax_x.text(0.5, 0.5, "No data", ha="center", va="center",
                      fontsize=9, color="#9E9E9E", transform=ax_x.transAxes)
        continue

    roi_gray   = entry["roi_gray"]
    mask_patch = entry["mask_patch"]
    metrics    = entry["metrics"]
    reg        = entry["reg"]

    # ── Col 1: ROI 이미지 + 마스크 오버레이 ──
    ax_img = fig.add_subplot(gs[row_i, 1])
    ax_img.imshow(roi_gray, cmap="gray", vmin=0, vmax=255, aspect="equal")
    # 마스크 오버레이
    rgba = np.zeros((*mask_patch.shape, 4), dtype=np.float32)
    rgba[mask_patch > 0] = [1.0, 0.15, 0.15, 0.55]
    ax_img.imshow(rgba, aspect="equal")
    # 바운딩 박스
    minr, minc, maxr, maxc = reg.bbox
    ax_img.add_patch(Rectangle(
        (minc, minr), maxc - minc, maxr - minr,
        linewidth=1.5, edgecolor="#FDD835", facecolor="none", linestyle="--",
    ))
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_linewidth(2.5); sp.set_edgecolor(color)
    # 결함 면적 표시
    ax_img.text(
        2, roi_gray.shape[0] - 4,
        f"area={reg.area}px²  class={int(entry['row'].get('class_id', 0))}",
        fontsize=5.5, color="white",
        bbox=dict(fc="black", alpha=0.6, pad=1.5, boxstyle="round,pad=0.25"),
    )
    if row_i == 0:
        ax_img.set_title("(A) ROI + Mask", fontsize=8, fontweight="bold",
                         color="#37474F", pad=3)

    # ── Col 2: 기하학적 분석 ──
    ax_viz = fig.add_subplot(gs[row_i, 2])
    draw_viz(ax_viz, roi_gray, mask_patch, reg, stype)
    if row_i == 0:
        ax_viz.set_title("(B) Geometric Analysis", fontsize=8, fontweight="bold",
                         color="#37474F", pad=3)

    # ── Col 3: 메트릭 바 차트 ──
    ax_bar = fig.add_subplot(gs[row_i, 3])
    metric_names = ["Linearity", "Solidity", "Extent", "Aspect Ratio"]
    values_raw   = [metrics[m] for m in metric_names]
    # Aspect Ratio를 0-1로 정규화 표시 (10 기준)
    values_disp  = [
        metrics["Linearity"],
        metrics["Solidity"],
        metrics["Extent"],
        min(metrics["Aspect Ratio"] / 10.0, 1.0),
    ]
    bar_colors = [METRIC_COLORS[m] for m in metric_names]
    y_pos = np.arange(len(metric_names))

    bars = ax_bar.barh(y_pos, values_disp, color=bar_colors, height=0.55,
                       edgecolor="white", linewidth=0.8)

    # 임계값 선
    thresholds = {"Linearity": 0.7, "Solidity": 0.8, "Aspect Ratio": 0.3}  # AR=3/10
    thresh_map = [0.7, 0.8, None, 0.3]
    for yi, thr in enumerate(thresh_map):
        if thr is not None:
            ax_bar.axvline(thr, ymin=(yi - 0.35) / len(metric_names),
                           ymax=(yi + 0.35) / len(metric_names),
                           color="white", lw=1.2, linestyle="--", alpha=0.8)

    # 값 레이블
    for yi, (bar, vd, vr) in enumerate(zip(bars, values_disp, values_raw)):
        label = f"{vr:.2f}" if metric_names[yi] != "Aspect Ratio" else f"{vr:.1f}"
        ax_bar.text(
            min(vd + 0.02, 0.97), yi,
            label, va="center", fontsize=7, fontweight="bold",
            color=bar_colors[yi],
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(
        ["Linearity", "Solidity", "Extent", "AspectR\n(÷10)"],
        fontsize=7,
    )
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.tick_params(axis="x", labelsize=6)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.35)

    # 서브타입 판정 강조
    rule_met = (
        (stype == "linear_scratch" and metrics["Linearity"] > 0.7) or
        (stype == "elongated"      and metrics["Aspect Ratio"] > 3.0) or
        (stype == "compact_blob"   and metrics["Solidity"] > 0.8) or
        (stype in ("irregular", "general"))
    )
    badge_col = "#43A047" if rule_met else "#E53935"
    ax_bar.text(
        1.0, -0.18,
        f"→ {stype.replace('_',' ')}",
        transform=ax_bar.transAxes, ha="right", va="bottom",
        fontsize=6.5, fontweight="bold", color=badge_col,
    )

    if row_i == 0:
        ax_bar.set_title("(C) Metrics", fontsize=8, fontweight="bold",
                         color="#37474F", pad=3)

# ──────────────────────────────────────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FILE, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved → {OUT_FILE}")
plt.show()
