"""
figure_comparison_augmentation.py
====================================
Augmentation Method Visual Comparison

Layout: 4 rows (Class 1–4) × 3 cols
  Col 1: Raw (Original)  — original training image, cropped to defect region
  Col 2: Copy-Paste      — generated ROI direct-pasted onto clean background
  Col 3: CASDA (Ours)    — generated ROI Poisson-blended onto clean background

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_comparison_augmentation.py
"""

import ast
import json
import re
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────────────────────
try:
    PROJ_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJ_ROOT = Path("/content/CASDA")

DRIVE_DATA    = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES  = DRIVE_DATA / "train_images"
GENERATED_DIR = DRIVE_DATA / "augmented_images/generated"
HINT_DIR      = DRIVE_DATA / "controlnet_dataset/hints"
META_COMPOSED = DRIVE_DATA / "augmented_dataset/casda_composed/metadata.json"
ROI_META      = DRIVE_DATA / "roi_patches_v5.1/roi_metadata.csv"

OUT_DIR  = PROJ_ROOT / "review/figures"
OUT_FILE = OUT_DIR / "comparison_augmentation.jpg"

IMG_H, IMG_W = 256, 1600
ZOOM_HALF_W  = 300   # roi 중심 기준 좌우 ±px

CLASS_COLOR = {1: "#2196F3", 2: "#F44336", 3: "#4CAF50", 4: "#FF9800"}
COL_LABELS  = ["Raw (Original)", "Copy-Paste", "CASDA (Ours)"]
COL_COLORS  = ["#546E7A", "#C62828", "#1565C0"]

# ──────────────────────────────────────────────────────────────────────────────
# 인라인 블렌딩 유틸리티 (src.preprocessing.poisson_blender 대체)
# ──────────────────────────────────────────────────────────────────────────────

def extract_mask(hint_bgr: np.ndarray, threshold: int = 127) -> np.ndarray:
    """힌트 이미지 Red 채널(BGR index 2)에서 이진 마스크 추출."""
    _, mask = cv2.threshold(hint_bgr[:, :, 2], threshold, 255, cv2.THRESH_BINARY)
    return mask


def direct_paste(gen_bgr, hint_bgr, bg_gray, roi_bbox):
    """생성 ROI를 마스크 기반으로 단순 붙여넣기 (블렌딩 없음)."""
    x1, y1, x2, y2 = roi_bbox
    roi_w, roi_h = x2 - x1, y2 - y1

    mask_512 = extract_mask(hint_bgr)
    gen_gray  = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2GRAY)
    gen_r     = cv2.resize(gen_gray,  (roi_w, roi_h), interpolation=cv2.INTER_AREA)
    mask_r    = cv2.resize(mask_512,  (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    result = bg_gray.copy()
    result[y1:y2, x1:x2][mask_r > 127] = gen_r[mask_r > 127]
    return result


def poisson_blend(gen_bgr, hint_bgr, bg_gray, roi_bbox, dilation_px=8):
    """
    cv2.seamlessClone 기반 Poisson 블렌딩.
    실패 시 알파 블렌딩으로 폴백.
    """
    x1, y1, x2, y2 = roi_bbox
    roi_w, roi_h = x2 - x1, y2 - y1

    mask_512 = extract_mask(hint_bgr)
    gen_r    = cv2.resize(gen_bgr,   (roi_w, roi_h), interpolation=cv2.INTER_AREA)
    mask_r   = cv2.resize(mask_512,  (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    if np.count_nonzero(mask_r) == 0:
        return direct_paste(gen_bgr, hint_bgr, bg_gray, roi_bbox)

    # dilation + edge margin
    if dilation_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilation_px + 1, 2 * dilation_px + 1))
        mask_d = cv2.dilate(mask_r, k, iterations=1)
    else:
        mask_d = mask_r.copy()
    mask_d[[0, -1], :] = 0
    mask_d[:, [0, -1]] = 0

    # seamlessClone 크기 제약: source < target 필요
    src_h, src_w = mask_d.shape[:2]
    if src_h >= IMG_H or src_w >= IMG_W:
        cy = 1 if src_h >= IMG_H else 0
        cx = 1 if src_w >= IMG_W else 0
        gen_r  = gen_r[cy:src_h - cy, cx:src_w - cx]
        mask_d = mask_d[cy:src_h - cy, cx:src_w - cx]

    bg_color = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
    cx_paste = (x1 + x2) // 2
    cy_paste = (y1 + y2) // 2
    center   = (int(cx_paste), int(cy_paste))

    try:
        composited = cv2.seamlessClone(gen_r, bg_color, mask_d, center, cv2.NORMAL_CLONE)
        return cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        # 알파 블렌딩 폴백
        result = bg_gray.copy().astype(np.float32)
        gen_gray_r = cv2.cvtColor(gen_r, cv2.COLOR_BGR2GRAY).astype(np.float32)
        alpha = cv2.GaussianBlur(mask_d.astype(np.float32) / 255.0, (21, 21), 5.0)
        ph, pw = gen_r.shape[:2]
        ry1 = cy_paste - ph // 2
        rx1 = cx_paste - pw // 2
        ry2, rx2 = ry1 + ph, rx1 + pw
        region = result[ry1:ry2, rx1:rx2]
        result[ry1:ry2, rx1:rx2] = (alpha * gen_gray_r + (1 - alpha) * region)
        return result.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 공통 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def parse_gen_name(source_generated: str) -> dict:
    m = re.match(r"([^_]+\.[a-zA-Z0-9]+)_class(\d+)_region(\d+)_gen(\d+)\.png$",
                 source_generated)
    if m:
        return {"image_id": m.group(1),
                "class_id": int(m.group(2)),
                "region_id": int(m.group(3))}
    m2 = re.match(r"([^_]+\.[a-zA-Z0-9]+)_", source_generated)
    return {"image_id": m2.group(1) if m2 else source_generated,
            "class_id": 0, "region_id": 0}


def hint_path(source_generated: str) -> Path:
    m = re.match(r"(.+)_gen\d+\.png$", source_generated)
    stem = m.group(1) if m else Path(source_generated).stem
    return HINT_DIR / f"{stem}_hint.png"


def safe_bbox(val):
    try:
        return [int(v) for v in ast.literal_eval(str(val))]
    except Exception:
        return None


def crop_centered(img: np.ndarray, cx: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, cx - ZOOM_HALF_W)
    x2 = min(w, cx + ZOOM_HALF_W)
    return img[:, x1:x2]


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────
print("Loading metadata ...")
with open(META_COMPOSED) as f:
    metadata = json.load(f)
print(f"  {len(metadata)} entries")

print("Loading ROI metadata ...")
df_roi = pd.read_csv(ROI_META, sep=None, engine="python")
df_roi.columns = df_roi.columns.str.strip().str.lower().str.replace(" ", "_")
df_roi["roi_bbox"] = df_roi["roi_bbox"].apply(safe_bbox)
df_roi = df_roi.dropna(subset=["roi_bbox"])

by_class: dict = {}
for e in metadata:
    cls = int(e.get("class_id", 0))
    by_class.setdefault(cls, []).append(e)
for cls in by_class:
    by_class[cls].sort(key=lambda x: x.get("suitability_score", 0), reverse=True)

selected: dict = {}
for cls_id in [1, 2, 3, 4]:
    for e in by_class.get(cls_id, []):
        gen_p = GENERATED_DIR / e["source_generated"]
        h_p   = hint_path(e["source_generated"])
        bg_p  = TRAIN_IMAGES / e["source_background"]
        if gen_p.exists() and h_p.exists() and bg_p.exists():
            selected[cls_id] = e
            pinfo = parse_gen_name(e["source_generated"])
            print(f"  Class {cls_id}: {pinfo['image_id']}  "
                  f"region={pinfo['region_id']}  "
                  f"score={e.get('suitability_score', 0):.3f}")
            break
    if cls_id not in selected:
        print(f"  [WARN] Class {cls_id}: no valid sample found")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 생성
# ──────────────────────────────────────────────────────────────────────────────
N_ROWS = 4
N_COLS = 3
FIG_W  = 4.8 * N_COLS + 0.8
FIG_H  = 2.2 * N_ROWS + 0.8

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

gs = gridspec.GridSpec(
    N_ROWS, N_COLS,
    figure=fig,
    hspace=0.07, wspace=0.05,
    top=0.93, bottom=0.03, left=0.06, right=0.98,
)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for row_idx, cls_id in enumerate([1, 2, 3, 4]):
    cls_color = CLASS_COLOR[cls_id]

    if cls_id not in selected:
        for col_idx in range(N_COLS):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(np.full((IMG_H, ZOOM_HALF_W * 2), 64, dtype=np.uint8),
                      cmap="gray", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
        continue

    entry = selected[cls_id]
    pinfo = parse_gen_name(entry["source_generated"])

    gen_bgr  = cv2.imread(str(GENERATED_DIR / entry["source_generated"]), cv2.IMREAD_COLOR)
    hint_bgr = cv2.imread(str(hint_path(entry["source_generated"])),      cv2.IMREAD_COLOR)
    bg_gray  = cv2.imread(str(TRAIN_IMAGES / entry["source_background"]), cv2.IMREAD_GRAYSCALE)
    src_gray = cv2.imread(str(TRAIN_IMAGES / pinfo["image_id"]),          cv2.IMREAD_GRAYSCALE) \
               if (TRAIN_IMAGES / pinfo["image_id"]).exists() else None

    roi_bbox = tuple(entry["roi_bbox"])
    x1r, y1r, x2r, y2r = roi_bbox
    roi_cx = (x1r + x2r) // 2

    # ── Col 1: Raw (원본 학습 이미지)
    if src_gray is not None:
        mask_r = ((df_roi["image_id"] == pinfo["image_id"]) &
                  (df_roi["class_id"] == cls_id) &
                  (df_roi["region_id"] == pinfo["region_id"]))
        src_rows = df_roi[mask_r]
        if src_rows.empty:
            src_rows = df_roi[(df_roi["image_id"] == pinfo["image_id"]) &
                              (df_roi["class_id"] == cls_id)]
        src_cx = (src_rows.iloc[0]["roi_bbox"][0] + src_rows.iloc[0]["roi_bbox"][2]) // 2 \
                 if not src_rows.empty else IMG_W // 2
        raw_crop = crop_centered(src_gray, src_cx)
    else:
        raw_crop = np.full((IMG_H, ZOOM_HALF_W * 2), 96, dtype=np.uint8)

    # ── Col 2: Copy-Paste
    if gen_bgr is not None and hint_bgr is not None and bg_gray is not None:
        cp_img  = direct_paste(gen_bgr, hint_bgr, bg_gray, roi_bbox)
        cp_crop = crop_centered(cp_img, roi_cx)
    else:
        cp_crop = np.full((IMG_H, ZOOM_HALF_W * 2), 96, dtype=np.uint8)

    # ── Col 3: CASDA (Poisson blend)
    if gen_bgr is not None and hint_bgr is not None and bg_gray is not None:
        casda_img  = poisson_blend(gen_bgr, hint_bgr, bg_gray, roi_bbox)
        casda_crop = crop_centered(casda_img, roi_cx)
    else:
        casda_crop = np.full((IMG_H, ZOOM_HALF_W * 2), 96, dtype=np.uint8)

    panels = [raw_crop, cp_crop, casda_crop]

    for col_idx, panel in enumerate(panels):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.imshow(clahe.apply(panel), cmap="gray", aspect="auto", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])

        lw = 2.8 if col_idx == 2 else 2.0
        for sp in ax.spines.values():
            sp.set_linewidth(lw)
            sp.set_edgecolor(cls_color)

        if row_idx == 0:
            ax.set_title(COL_LABELS[col_idx], fontsize=13, fontweight="bold",
                         color=COL_COLORS[col_idx], pad=6)

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
