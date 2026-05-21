"""
figure_blend_comparison.py  —  Controlled Blending Comparison
==============================================================
[목적]
  Reviewer 설득용 figure:
  "동일 배경 × 동일 위치 × 동일 생성 ROI" 조건에서
  Direct Paste vs Poisson Blending 차이를 시각적으로 증명.

[방법]
  1. metadata_composed.json에서 class별 대표 샘플 자동 선별
  2. 선별된 (source_generated, source_background, roi_bbox)로
     양쪽 방법을 jitter=0, scale=1.0 고정하여 직접 재합성
     → 블렌딩 방식 외 모든 변수 동일
  3. 4-column figure 생성:
       [배경 + 위치] | [Direct Paste] | [Poisson Blend] | [경계 확대 ×2]

[경로 설정]
  아래 PATH 블록만 수정하면 됨. Colab Drive 마운트 후 실행.

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_blend_comparison.py
"""

import json
import re
import sys
from pathlib import Path

# ── PROJ_ROOT 설정 ──
# !python 실행 시: __file__ 사용 / Colab 셀 붙여넣기 시: 고정 경로 사용
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
from src.preprocessing.poisson_blender import PoissonBlender

# ──────────────────────────────────────────────────────────────────────────────
# 경로 설정 (Colab Drive 경로 — 필요 시 수정)
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
GENERATED_DIR= DRIVE_DATA / "augmented_images/generated"   # AUG_IMAGES/generated
HINT_DIR     = DRIVE_DATA / "controlnet_dataset/hints"      # CN_DATASET/hints
META_COMPOSED= DRIVE_DATA / "augmented_dataset/casda_composed/metadata.json"

OUT_DIR      = PROJ_ROOT / "review/figures"
OUT_FILE     = OUT_DIR / "blend_comparison.png"

# ──────────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES  = {0: "Class 1", 1: "Class 2", 2: "Class 3", 3: "Class 4"}
ZOOM_HALF_W  = 220   # ROI 중심 기준 좌우 ±px (확대 패널 너비 = ZOOM_HALF_W*2)


# ──────────────────────────────────────────────────────────────────────────────
# 샘플 선별
# ──────────────────────────────────────────────────────────────────────────────

def source_image_id(source_generated: str) -> str:
    """'c924ce298.jpg_class4_region0_gen0.png' → 'c924ce298.jpg'"""
    m = re.match(r"([^_]+\.[a-zA-Z]+)_", source_generated)
    return m.group(1) if m else source_generated


def sample_name_from_gen(source_generated: str) -> str:
    """'c924ce298.jpg_class4_region0_gen0.png' → 'c924ce298.jpg_class4_region0'"""
    m = re.match(r"(.+)_gen\d+\.png$", source_generated)
    return m.group(1) if m else Path(source_generated).stem


def hint_path(source_generated: str) -> Path:
    sname = sample_name_from_gen(source_generated)
    return HINT_DIR / f"{sname}_hint.png"


def is_roi_centered(entry: dict, margin: int = 120) -> bool:
    """ROI 삽입 위치가 이미지 가장자리에서 충분히 떨어져 있는지 확인."""
    roi = entry.get("roi_bbox")
    if not roi:
        return False
    x1, y1, x2, y2 = roi
    # jitter=0으로 재합성할 것이므로 원래 roi_bbox 기준으로 판단
    return x1 >= margin and x2 <= (1600 - margin)


def select_samples(metadata: list, n_per_class: int = 1) -> list:
    """
    class별로 n_per_class개의 대표 샘플 선별.
    기준: suitability_score 높고, ROI가 이미지 중앙에 있고, 힌트 파일 존재.
    """
    by_class: dict = {}
    for e in metadata:
        if not is_roi_centered(e):
            continue
        cls = e["class_id"]
        by_class.setdefault(cls, []).append(e)

    # suitability_score 내림차순 정렬
    for cls in by_class:
        by_class[cls].sort(key=lambda x: x.get("suitability_score", 0), reverse=True)

    selected = []
    for cls in sorted(by_class.keys()):
        count = 0
        for e in by_class[cls]:
            h = hint_path(e["source_generated"])
            g = GENERATED_DIR / e["source_generated"]
            if h.exists() and g.exists():
                selected.append(e)
                count += 1
                if count >= n_per_class:
                    break
        if count == 0:
            print(f"[WARN] Class {cls}: 유효 샘플 없음 (생성/힌트 파일 확인 필요)")

    return selected


# ──────────────────────────────────────────────────────────────────────────────
# 재합성 (Direct Paste / Poisson Blending)
# ──────────────────────────────────────────────────────────────────────────────

def compose_pair(entry: dict):
    """
    주어진 entry의 (source_generated, source_background, roi_bbox)로
    Direct Paste와 Poisson Blending 결과를 동일 조건에서 재합성.

    jitter_x=0, scale_factor=1.0 고정 → 블렌딩 방식만 다름.

    Returns:
        dict with keys: background, no_blend, composed,
                        src_img, roi_bbox, dest_x1, dest_x2
        실패 시 None.
    """
    gen_path  = GENERATED_DIR / entry["source_generated"]
    h_path    = hint_path(entry["source_generated"])
    bg_path   = TRAIN_IMAGES / entry["source_background"]
    src_path  = TRAIN_IMAGES / source_image_id(entry["source_generated"])

    # 이미지 로드
    gen_img  = cv2.imread(str(gen_path),  cv2.IMREAD_COLOR)
    hint_img = cv2.imread(str(h_path),    cv2.IMREAD_COLOR)
    bg_img   = cv2.imread(str(bg_path),   cv2.IMREAD_COLOR)
    src_img  = cv2.imread(str(src_path),  cv2.IMREAD_COLOR) if src_path.exists() else None

    if gen_img is None or hint_img is None or bg_img is None:
        print(f"[WARN] 이미지 로드 실패: {entry['source_generated']}")
        return None

    roi_bbox = tuple(entry["roi_bbox"])    # (x1, y1, x2, y2) in full image
    class_id = entry["class_id"]

    blender = PoissonBlender(dilation_px=8, mask_threshold=127)

    # ── Poisson Blending (jitter=0, scale=1.0) ──
    result_composed = blender.compose_single(
        gen_img, hint_img, bg_img,
        roi_bbox=roi_bbox,
        class_id=class_id,
        jitter_x=0,
        scale_factor=1.0,
        use_smooth_mask=True,
        smooth_ksize=21,
        smooth_sigma=7.0,
    )

    # ── Direct Paste (no_blend, 동일 조건) ──
    # PoissonBlender.extract_mask_from_hint → resize → paste
    mask_512 = blender.extract_mask_from_hint(hint_img)
    x1, y1, x2, y2 = roi_bbox
    roi_w, roi_h = x2 - x1, y2 - y1

    gen_resized  = cv2.resize(gen_img,   (roi_w, roi_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask_512,  (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    no_blend_img = bg_img.copy()
    mask_bool    = mask_resized > 127
    if mask_bool.any():
        no_blend_img[y1:y2, x1:x2][mask_bool] = gen_resized[mask_bool]

    # BGR → RGB
    def to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return {
        "background": to_rgb(bg_img),
        "no_blend":   to_rgb(no_blend_img),
        "composed":   to_rgb(result_composed.composited_image) if result_composed.success else to_rgb(no_blend_img),
        "src_img":    to_rgb(src_img) if src_img is not None else None,
        "roi_bbox":   roi_bbox,
        "success":    result_composed.success,
        "entry":      entry,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figure 생성
# ──────────────────────────────────────────────────────────────────────────────

def crop_roi_region(img: np.ndarray, roi_bbox: tuple, half_w: int = ZOOM_HALF_W) -> np.ndarray:
    """ROI 중심을 기준으로 ±half_w 범위 크롭 (full height)."""
    x1, y1, x2, y2 = roi_bbox
    cx = (x1 + x2) // 2
    h, w = img.shape[:2]
    crop_x1 = max(0, cx - half_w)
    crop_x2 = min(w, cx + half_w)
    return img[:, crop_x1:crop_x2]


def add_roi_box(ax, roi_bbox, img_w, img_h, color="#ffdd00", lw=2.0, label=None):
    x1, y1, x2, y2 = roi_bbox
    rect = mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(x1 + 3, y1 - 4, label, color=color, fontsize=6.5, fontweight="bold",
                bbox=dict(fc="black", alpha=0.5, pad=1.5, boxstyle="round,pad=0.2"))


def tag(ax, text, loc="tl", color="white", fontsize=7.5):
    x = 0.01 if "l" in loc else 0.99
    y = 0.98 if "t" in loc else 0.02
    ha = "left" if "l" in loc else "right"
    va = "top" if "t" in loc else "bottom"
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=fontsize, color=color,
            bbox=dict(fc="black", alpha=0.55, pad=2, boxstyle="round,pad=0.25"))


def make_figure(results: list):
    n = len(results)
    if n == 0:
        print("표시할 결과 없음")
        return

    # ── 레이아웃: n행 × 5열
    # col 0: 배경 + ROI 위치  (전체)
    # col 1: No-Blend 결과    (전체)
    # col 2: Poisson 결과     (전체)
    # col 3: No-Blend 경계확대 (크롭)
    # col 4: Poisson 경계확대  (크롭)
    COL_RATIO   = [4, 4, 4, 2, 2]          # 상대 너비 비율
    FULL_W_IN   = sum(COL_RATIO[:3]) * 1.5  # 전체 이미지 3개 너비 (인치)
    ZOOM_W_IN   = sum(COL_RATIO[3:]) * 1.5
    ROW_H_IN    = 1.4

    fig = plt.figure(figsize=(FULL_W_IN + ZOOM_W_IN + 0.3, n * ROW_H_IN + 1.2),
                     facecolor="white")

    outer_gs = gridspec.GridSpec(
        n, 5,
        figure=fig,
        width_ratios=COL_RATIO,
        hspace=0.12, wspace=0.04,
        left=0.05, right=0.97, top=0.88, bottom=0.10,
    )

    # 컬럼 헤더 (첫 행만)
    COL_HEADERS = [
        "Background (clean)\n+ paste location",
        "Direct Paste\n(w/o Blending)",
        "Poisson Blending\n(CASDA)",
        "Boundary\nDirect Paste",
        "Boundary\nPoisson Blend",
    ]
    COL_COLORS = ["#2c3e50", "#c0392b", "#2980b9", "#c0392b", "#2980b9"]

    for col_idx, (header, color) in enumerate(zip(COL_HEADERS, COL_COLORS)):
        ax_h = fig.add_subplot(outer_gs[0, col_idx])
        ax_h.set_title(header, fontsize=8.5, fontweight="bold", pad=4,
                       color="white",
                       bbox=dict(fc=color, boxstyle="round,pad=0.35", alpha=0.9))
        ax_h.axis("off")

    for row, res in enumerate(results):
        entry    = res["entry"]
        roi_bbox = res["roi_bbox"]
        cls_name = CLASS_NAMES.get(entry["class_id"], f"C{entry['class_id']}")
        subtype  = entry.get("defect_subtype", "")
        src_id   = source_image_id(entry["source_generated"])
        bg_id    = entry["source_background"]

        bg_img   = res["background"]
        nb_img   = res["no_blend"]
        co_img   = res["composed"]
        h_img, w_img = bg_img.shape[:2]

        nb_crop  = crop_roi_region(nb_img, roi_bbox)
        co_crop  = crop_roi_region(co_img, roi_bbox)

        def show(col, img, clip=False):
            ax = fig.add_subplot(outer_gs[row, col])
            ax.imshow(img, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            return ax

        # ── col 0: 배경 + ROI 위치 ──
        ax0 = show(0, bg_img)
        add_roi_box(ax0, roi_bbox, w_img, h_img, color="#ffdd00", lw=2.0,
                    label="paste region")
        tag(ax0, bg_id[:12], "tl")
        # 행 레이블
        ax0.set_ylabel(f"{cls_name}\n{subtype}", fontsize=7.5,
                       rotation=90, labelpad=5, va="center")

        # ── col 1: No-Blend 전체 ──
        ax1 = show(1, nb_img)
        add_roi_box(ax1, roi_bbox, w_img, h_img, color="#ff6666", lw=1.5)
        tag(ax1, "direct paste", "tl")

        # ── col 2: Poisson 전체 ──
        ax2 = show(2, co_img)
        add_roi_box(ax2, roi_bbox, w_img, h_img, color="#66aaff", lw=1.5)
        tag(ax2, "poisson blend", "tl")
        if not res["success"]:
            tag(ax2, "blend failed", "br", color="#ff9900")

        # ── col 3: No-Blend 크롭 ──
        ax3 = show(3, nb_crop)
        # 크롭 내 ROI 박스 (x 오프셋 조정)
        x1, y1, x2, y2 = roi_bbox
        cx = (x1 + x2) // 2
        crop_x1 = max(0, cx - ZOOM_HALF_W)
        ax3.add_patch(mpatches.Rectangle(
            (x1 - crop_x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="#ff6666", facecolor="none",
        ))

        # ── col 4: Poisson 크롭 ──
        ax4 = show(4, co_crop)
        ax4.add_patch(mpatches.Rectangle(
            (x1 - crop_x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="#66aaff", facecolor="none",
        ))

    # ── 전체 타이틀 ──
    fig.suptitle(
        "Controlled Comparison: Direct Paste vs Poisson Blending\n"
        "Same ROI · Same Background · Same Insertion Location  (jitter=0, scale=1.0)",
        fontsize=10, fontweight="bold", y=0.97,
    )

    # ── 범례 ──
    legend_patches = [
        mpatches.Patch(ec="#ffdd00", fc="none", lw=2.0, label="paste location (background)"),
        mpatches.Patch(ec="#ff6666", fc="none", lw=1.5, label="ROI region (direct paste)"),
        mpatches.Patch(ec="#66aaff", fc="none", lw=1.5, label="ROI region (poisson blend)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=8, frameon=True, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.01))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=180, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading metadata...")
    with open(META_COMPOSED) as f:
        metadata = json.load(f)
    print(f"  {len(metadata)} entries loaded")

    print("Selecting representative samples (1 per class)...")
    samples = select_samples(metadata, n_per_class=1)
    print(f"  {len(samples)} samples selected:")
    for s in samples:
        print(f"    [{CLASS_NAMES.get(s['class_id'])}] {s['source_generated']}"
              f"  bg={s['source_background']}"
              f"  score={s.get('suitability_score', 0):.3f}")

    if not samples:
        print("\n[ERROR] 선별된 샘플 없음. 경로 설정을 확인하세요.")
        print(f"  GENERATED_DIR : {GENERATED_DIR}")
        print(f"  HINT_DIR      : {HINT_DIR}")
        return

    print("\nRe-compositing (jitter=0, scale=1.0)...")
    results = []
    for s in samples:
        res = compose_pair(s)
        if res:
            status = "OK" if res["success"] else "blend_failed(fallback)"
            print(f"  {CLASS_NAMES.get(s['class_id'])}: {status}")
            results.append(res)

    if not results:
        print("[ERROR] 합성 결과 없음.")
        return

    print(f"\nGenerating figure ({len(results)} rows)...")
    make_figure(results)


if __name__ == "__main__":
    main()
