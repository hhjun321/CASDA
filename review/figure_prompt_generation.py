"""
figure_prompt_generation.py  —  Stage A §7: Hybrid Text Prompt Generation
==========================================================================
논문 §7 시각화.  generate_simple_prompt 기준 예시 + 3개 딕셔너리 참조 테이블.

Layout:
  (A) Simple prompt 구조 다이어그램
        "{base_desc} on {surface}, class N"

  (B) 5개 서브타입 × (ROI image | metadata | simple prompt) 예시 그리드

  (C) 3개 참조 테이블 (가로 배치)
        C1: DEFECT_DESCRIPTIONS  (subtype → base / detailed / characteristics)
        C2: BACKGROUND_DESCRIPTIONS  (bg_type → surface / texture / pattern)
        C3: SURFACE_QUALITY  (stability threshold → quality grade → vocabulary)

Run in Google Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python /content/CASDA/review/figure_prompt_generation.py
"""

import ast, sys, textwrap
from pathlib import Path

try:
    PROJ_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJ_ROOT = Path("/content/CASDA")

sys.path.insert(0, str(PROJ_ROOT))

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src.preprocessing.prompt_generator import PromptGenerator

# ──────────────────────────────────────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_DATA   = Path("/content/drive/MyDrive/data/Severstal")
TRAIN_IMAGES = DRIVE_DATA / "train_images"
TRAIN_CSV    = DRIVE_DATA / "train.csv"
ROI_META     = DRIVE_DATA / "roi_patches/roi_metadata.csv"

OUT_DIR  = Path("/content/CASDA/review/figures")
OUT_FILE = OUT_DIR / "prompt_generation.png"

IMG_H, IMG_W = 256, 1600

# ──────────────────────────────────────────────────────────────────────────────
# 5개 대표 예시 (서브타입 × 최적 배경 조합)
# ──────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    dict(defect_subtype="linear_scratch", background_type="vertical_stripe",
         class_id=2, stability=0.85, suitability=1.00,
         color="#1565C0"),
    dict(defect_subtype="elongated",      background_type="horizontal_stripe",
         class_id=1, stability=0.72, suitability=0.90,
         color="#2E7D32"),
    dict(defect_subtype="compact_blob",   background_type="smooth",
         class_id=3, stability=0.63, suitability=1.00,
         color="#E65100"),
    dict(defect_subtype="irregular",      background_type="complex_pattern",
         class_id=4, stability=0.55, suitability=1.00,
         color="#6A1B9A"),
    dict(defect_subtype="general",        background_type="textured",
         class_id=2, stability=0.48, suitability=0.70,
         color="#546E7A"),
]

# ──────────────────────────────────────────────────────────────────────────────
# 스타일 상수
# ──────────────────────────────────────────────────────────────────────────────
SUBTYPE_COLORS = {
    "linear_scratch": "#1565C0", "elongated": "#2E7D32",
    "compact_blob":   "#E65100", "irregular": "#6A1B9A",
    "general":        "#546E7A",
}
BG_COLORS = {
    "smooth":            "#1E88E5", "vertical_stripe":   "#43A047",
    "horizontal_stripe": "#FB8C00", "textured":          "#8E24AA",
    "complex_pattern":   "#E53935",
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


def _iter_pool(pool_df, train_df, col_img, col_rle, min_px: int):
    """pool_df 를 순회하며 mask 픽셀 수 >= min_px 인 첫 번째 (patch, mask) 반환."""
    for _, row in pool_df.iterrows():
        bbox = row.get("roi_bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        img_path = TRAIN_IMAGES / str(row.get("image_id", ""))
        if not img_path.exists():
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        rle_rows = train_df[train_df[col_img] == str(row["image_id"])]
        full_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for _, rr in rle_rows.iterrows():
            full_mask = np.maximum(full_mask, rle_to_mask(rr[col_rle]))
        mask = full_mask[y1:y2, x1:x2]
        if mask.sum() < min_px:
            continue
        return gray[y1:y2, x1:x2], mask
    return None, None


def load_roi(ex: dict):
    """
    주어진 subtype+bg 에 맞는 ROI 패치 로드. 3단계 폴백:
      1차: subtype + bg 완전 일치
      2차: subtype 만 일치 (bg 무시)
      3차: elongated 전용 — 전체에서 min_px=5 로 탐색
    """
    # elongated 는 얇아서 mask 픽셀 수가 적음
    min_px = 5 if ex["defect_subtype"] == "elongated" else 15

    try:
        df = pd.read_csv(ROI_META, sep=None, engine="python")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df["roi_bbox"] = df["roi_bbox"].apply(safe_bbox)
        df = df.dropna(subset=["roi_bbox"])

        train_df = pd.read_csv(TRAIN_CSV, sep=None, engine="python")
        train_df.columns = train_df.columns.str.strip()
        col_img = [c for c in train_df.columns if "image" in c.lower()][0]
        col_rle = [c for c in train_df.columns if "encoded" in c.lower() or "pixel" in c.lower()][0]

        sub_col = "defect_subtype"  if "defect_subtype"  in df.columns else None
        bg_col  = "background_type" if "background_type" in df.columns else None

        # 1차: subtype + bg 완전 일치
        pool = df.copy()
        if sub_col:
            pool = pool[pool[sub_col] == ex["defect_subtype"]]
        if bg_col:
            pool = pool[pool[bg_col]  == ex["background_type"]]
        patch, mask = _iter_pool(pool, train_df, col_img, col_rle, min_px)
        if patch is not None:
            return patch, mask

        # 2차: subtype 만 일치 (bg 무시)
        pool2 = df.copy()
        if sub_col:
            pool2 = pool2[pool2[sub_col] == ex["defect_subtype"]]
        patch, mask = _iter_pool(pool2, train_df, col_img, col_rle, min_px)
        if patch is not None:
            print(f"  [{ex['defect_subtype']}] 2차 폴백 (bg 무시) 성공")
            return patch, mask

        # 3차: elongated 전용 — 전체 데이터에서 min_px=5
        if ex["defect_subtype"] == "elongated":
            patch, mask = _iter_pool(df, train_df, col_img, col_rle, min_px=5)
            if patch is not None:
                print(f"  [elongated] 3차 폴백 (전체 탐색) 성공")
                return patch, mask

    except Exception as e:
        print(f"  [load_roi] {ex['defect_subtype']}: {e}")

    return None, None


def draw_table(ax, rows, col_widths, row_height=0.135, x0=0.01, y0=0.97,
               header_fc="#E0E0E0", odd_fc="#F9F9F9", even_fc="white",
               font_size=6.5):
    """ax 위에 텍스트 테이블을 그린다. rows[0] 은 헤더."""
    for ri, row_data in enumerate(rows):
        y = y0 - ri * row_height
        fc = header_fc if ri == 0 else (odd_fc if ri % 2 == 1 else even_fc)
        ax.add_patch(FancyBboxPatch(
            (x0 - 0.005, y - row_height + 0.012), sum(col_widths) + 0.01,
            row_height - 0.014,
            fc=fc, ec="#BDBDBD", lw=0.5, boxstyle="round,pad=0.01",
            transform=ax.transAxes, clip_on=False,
        ))
        x = x0
        for ci, (cell, cw) in enumerate(zip(row_data, col_widths)):
            is_hdr = ri == 0
            fw = "bold" if is_hdr else "normal"
            fc_txt = "#212121" if is_hdr else "#333333"
            ax.text(x + cw / 2, y - row_height / 2 + 0.01, str(cell),
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=font_size if not is_hdr else font_size + 0.3,
                    fontweight=fw, color=fc_txt, clip_on=False)
            x += cw


# ──────────────────────────────────────────────────────────────────────────────
# 프롬프트 생성 (detailed 스타일 — 논문 실제 사용)
# quality word 는 seed 고정으로 재현성 보장
# ──────────────────────────────────────────────────────────────────────────────
import random

QUALITY_FIRST = {          # SURFACE_QUALITY 각 등급의 첫 번째 단어 고정
    "high":   "pristine",
    "medium": "standard",
    "low":    "worn",
}

def get_quality_word(stability: float) -> tuple[str, str]:
    """(grade, word) 반환"""
    grade = "high" if stability >= 0.8 else ("medium" if stability >= 0.5 else "low")
    return grade, QUALITY_FIRST[grade]

pg_detailed = PromptGenerator(style="detailed")

for ex in EXAMPLES:
    grade, qword = get_quality_word(ex["stability"])
    ex["quality_grade"] = grade
    ex["quality_word"]  = qword if ex["stability"] >= 0.6 else None  # <0.6 → 생략

    # generate_detailed_prompt 의 random.choice 를 우회해 고정 단어 사용
    import unittest.mock as mock
    with mock.patch("random.choice", side_effect=lambda lst: lst[0]):
        ex["prompt"] = pg_detailed.generate_detailed_prompt(
            defect_subtype  = ex["defect_subtype"],
            background_type = ex["background_type"],
            class_id        = ex["class_id"],
            stability_score = ex["stability"],
            defect_metrics  = {},
        )
    print(f"  {ex['defect_subtype']:15s} | {ex['prompt']}")

# ROI 이미지 로드
print("\nLoading ROI images ...")
for ex in EXAMPLES:
    ex["patch"], ex["mask"] = load_roi(ex)
    print(f"  {ex['defect_subtype']:15s}: {'loaded' if ex['patch'] is not None else 'placeholder'}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 생성
# ──────────────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 16.5, 13.5
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
fig.suptitle(
    "CASDA Stage A §7 — Hybrid Text Prompt Generation  (generate_detailed_prompt  ←  actual use)\n"
    'Template:  "{detailed_desc}  on  {surface}  with  {texture}  ({quality_word} condition),  steel defect class {N}"',
    fontsize=10.5, fontweight="bold", color="#0D47A1", y=0.995,
)

gs_main = gridspec.GridSpec(
    3, 1,
    figure=fig,
    height_ratios=[0.85, 2.8, 3.2],
    hspace=0.36,
    top=0.94, bottom=0.025, left=0.03, right=0.985,
)

# ══════════════════════════════════════════════════════════════════════════════
# (A) 구조 다이어그램
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs_main[0])
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 1); ax_a.axis("off")
ax_a.set_title("(A)  Detailed Prompt Structure  (actual use style)",
               fontsize=9, fontweight="bold", color="#1A237E", pad=3, loc="left")

def fbox(ax, x, y, w, h, text, fc, ec, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec,
                                lw=1.8, boxstyle="round,pad=0.05"))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=ec, multialignment="center")

def farrow(ax, x1, x2, y=0.50, label=""):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#546E7A",
                                lw=1.5, mutation_scale=13))
    if label:
        ax.text((x1 + x2) / 2, y + 0.18, label,
                ha="center", va="bottom", fontsize=7, color="#546E7A")

# 5 컴포넌트 박스
fbox(ax_a, 0.05, 0.15, 1.75, 0.70, "① detailed\ndesc",         "#E3F2FD", "#1565C0", fs=7.5)
fbox(ax_a, 1.98, 0.15, 1.75, 0.70, "② surface\n(on …)",        "#E8F5E9", "#2E7D32", fs=7.5)
fbox(ax_a, 3.91, 0.15, 1.75, 0.70, "③ texture\n(with …)",      "#E8F5E9", "#388E3C", fs=7.5)
fbox(ax_a, 5.84, 0.15, 1.85, 0.70, "④ quality_word\n(stability→)",  "#FFFDE7", "#F57F17", fs=7.5)
fbox(ax_a, 7.87, 0.15, 1.10, 0.70, "⑤ class_id",               "#FFF3E0", "#E65100", fs=7.5)

# + 기호
for px in [1.83, 3.76, 5.69, 7.72]:
    ax_a.text(px, 0.50, "+", ha="center", va="center",
              fontsize=12, fontweight="bold", color="#546E7A")

farrow(ax_a, 9.07, 9.38, y=0.50)
fbox(ax_a, 9.40, 0.10, 0.57, 0.80, "Full\nPrompt", "#F3E5F5", "#4A148C", fs=7)

# 하단 sub-label
sub_info = [
    (0.925, "DEFECT_DESCRIPTIONS\n[subtype]['detailed']"),
    (2.855, "BACKGROUND_DESCRIPTIONS\n[bg_type]['surface']"),
    (4.785, "BACKGROUND_DESCRIPTIONS\n[bg_type]['texture']"),
    (6.765, "SURFACE_QUALITY\n[grade][0]"),
    (8.425, "class_id"),
]
for xc, txt in sub_info:
    ax_a.text(xc, 0.06, txt, ha="center", va="bottom", fontsize=5.5,
              color="#757575", fontstyle="italic", multialignment="center")

# ══════════════════════════════════════════════════════════════════════════════
# (B) 5 예시 그리드  (5 rows × 3 cols)
# ══════════════════════════════════════════════════════════════════════════════
pos_b = gs_main[1].get_position(fig)
fig.text(0.03, pos_b.y1 + 0.005,
         "(B)  Examples — 5 Defect Subtypes × Best-Matching Background",
         fontsize=9, fontweight="bold", color="#1A237E")

gs_b = gridspec.GridSpecFromSubplotSpec(
    5, 3,
    subplot_spec=gs_main[1],
    width_ratios=[0.65, 0.72, 2.5],
    wspace=0.10, hspace=0.18,
)

# 컬럼 헤더
col_hdrs = ["ROI Image", "Metadata", "Generated Prompt  (detailed style — actual use)"]
for ci, hdr in enumerate(col_hdrs):
    pos = gs_b[0, ci].get_position(fig)
    fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.005,
             hdr, ha="center", va="bottom",
             fontsize=7.5, fontweight="bold", color="#37474F")

for row_i, ex in enumerate(EXAMPLES):
    col = ex["color"]

    # ── Col 0: ROI 이미지 ──
    ax_img = fig.add_subplot(gs_b[row_i, 0])
    patch = ex.get("patch"); mask = ex.get("mask")
    if patch is not None:
        ax_img.imshow(patch, cmap="gray", vmin=0, vmax=255, aspect="equal")
        if mask is not None and mask.any():
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba[mask > 0] = [1.0, 0.2, 0.2, 0.50]
            ax_img.imshow(rgba, aspect="equal")
    else:
        ax_img.set_facecolor("#ECEFF1")
        ax_img.text(0.5, 0.5, ex["defect_subtype"].replace("_", "\n"),
                    transform=ax_img.transAxes, ha="center", va="center",
                    fontsize=6.5, color=col, fontweight="bold")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_linewidth(2.5); sp.set_edgecolor(col)

    # ── Col 1: 메타데이터 ──
    ax_meta = fig.add_subplot(gs_b[row_i, 1])
    ax_meta.axis("off")
    grade     = ex.get("quality_grade", "medium")
    qword     = ex.get("quality_word")
    grade_col = {"high": "#2E7D32", "medium": "#1565C0", "low": "#6A1B9A"}.get(grade, "#555")
    meta_lines = [
        (ex["defect_subtype"].replace("_", " "),           col,       "bold",   7.0),
        (f"bg: {ex['background_type'].replace('_',' ')}",
                              BG_COLORS.get(ex["background_type"], "#555"),
                                                                       "normal", 6.5),
        (f"class {ex['class_id']}",                        "#333",    "normal", 6.5),
        (f"stability = {ex['stability']:.2f}",             "#555",    "normal", 6.2),
        (f"→ {grade}  →  \"{qword}\"" if qword else
         f"→ {grade}  (condition omitted)",                grade_col, "bold",   6.5),
        (f"suitability = {ex['suitability']:.2f}",         "#777",    "normal", 6.0),
    ]
    y = 0.96
    for txt, c, fw, fs in meta_lines:
        ax_meta.text(0.05, y, txt, transform=ax_meta.transAxes,
                     ha="left", va="top", fontsize=fs, fontweight=fw, color=c)
        y -= 0.165

    # ── Col 2: 프롬프트 텍스트 박스 (5줄 컴포넌트 색상 강조) ──
    ax_p = fig.add_subplot(gs_b[row_i, 2])
    ax_p.axis("off")
    ax_p.set_facecolor("#F8F9FA")
    ax_p.patch.set_alpha(0.7)

    defect_info = PromptGenerator.DEFECT_DESCRIPTIONS.get(
        ex["defect_subtype"], {"detailed": "a defect"})
    bg_info = PromptGenerator.BACKGROUND_DESCRIPTIONS.get(
        ex["background_type"], {"surface": "metal surface", "texture": "uniform texture"})

    detailed = defect_info["detailed"]
    surface  = bg_info["surface"]
    texture  = bg_info["texture"]
    cls      = ex["class_id"]

    FONT = 8.0
    lines_p = [
        (f'"{detailed}',           "#1565C0", "bold",   "①"),
        (f' on {surface}',         "#2E7D32", "bold",   "②"),
        (f' with {texture}',       "#388E3C", "normal", "③"),
    ]
    if qword:
        lines_p.append((f' ({qword} condition),',              "#F57F17", "bold",   "④"))
    else:
        lines_p.append((' [condition omitted — stability<0.6]', "#BDBDBD", "normal", "④"))
    lines_p.append((f' steel defect class {cls}"',             "#E65100", "bold",   "⑤"))

    n = len(lines_p)
    step = 0.88 / n
    y_start = 0.96
    for i, (txt, c, fw, badge) in enumerate(lines_p):
        yp = y_start - i * step
        ax_p.text(0.03, yp, txt, transform=ax_p.transAxes,
                  ha="left", va="top", fontsize=FONT, fontweight=fw, color=c)
        ax_p.text(0.97, yp, badge, transform=ax_p.transAxes,
                  ha="right", va="top", fontsize=7.5, fontweight="bold", color=c)

    for sp in ax_p.spines.values():
        sp.set_linewidth(1.5); sp.set_edgecolor(col)

# ══════════════════════════════════════════════════════════════════════════════
# (C) 3개 참조 테이블
# ══════════════════════════════════════════════════════════════════════════════
pos_c = gs_main[2].get_position(fig)
fig.text(0.03, pos_c.y1 + 0.005,
         "(C)  Reference Dictionaries  (PromptGenerator class variables)",
         fontsize=9, fontweight="bold", color="#1A237E")

gs_c = gridspec.GridSpecFromSubplotSpec(
    1, 3,
    subplot_spec=gs_main[2],
    width_ratios=[1.25, 1.35, 0.90],
    wspace=0.14,
)

# ── C1: DEFECT_DESCRIPTIONS ──
ax_c1 = fig.add_subplot(gs_c[0, 0])
ax_c1.axis("off")
ax_c1.set_title("DEFECT_DESCRIPTIONS", fontsize=8, fontweight="bold",
                color="#1565C0", pad=4)

defect_rows = [
    ["subtype", "base", "detailed  ← used in figure"],
    ["linear_scratch", "a linear scratch defect",  "a high-linearity elongated scratch"],
    ["elongated",      "an elongated defect",       "a moderately elongated defect region"],
    ["compact_blob",   "a compact blob defect",     "a solid compact defect spot"],
    ["irregular",      "an irregular defect",       "an irregular defect with\ncomplex boundaries"],
    ["general",        "a defect",                  "a general surface defect"],
]
draw_table(ax_c1, defect_rows, col_widths=[0.27, 0.35, 0.38],
           y0=0.95, row_height=0.155, font_size=6.3)

# 서브타입 색상 적용
subtype_order = ["linear_scratch","elongated","compact_blob","irregular","general"]
for ri, stype in enumerate(subtype_order):
    ax_c1.text(0.01 + 0.27/2, 0.95 - (ri+1)*0.155 - 0.155/2 + 0.01,
               stype, transform=ax_c1.transAxes,
               ha="center", va="center", fontsize=6.3, fontweight="bold",
               color=SUBTYPE_COLORS.get(stype, "#333"))

# ── C2: BACKGROUND_DESCRIPTIONS ──
ax_c2 = fig.add_subplot(gs_c[0, 1])
ax_c2.axis("off")
ax_c2.set_title("BACKGROUND_DESCRIPTIONS", fontsize=8, fontweight="bold",
                color="#2E7D32", pad=4)

bg_rows = [
    ["bg_type",            "surface",                       "texture",               "pattern"],
    ["smooth",             "smooth metal surface",          "uniform texture",       "no visible pattern"],
    ["vertical_stripe",    "vertical striped metal surface","directional texture",   "vertical line pattern"],
    ["horizontal_stripe",  "horizontal striped\nmetal surface","directional texture","horizontal line pattern"],
    ["textured",           "textured metal surface",        "grainy texture",        "subtle surface texture"],
    ["complex_pattern",    "complex patterned\nmetal surface","multi-directional\ntexture","complex surface pattern"],
]
draw_table(ax_c2, bg_rows, col_widths=[0.24, 0.31, 0.24, 0.21],
           y0=0.95, row_height=0.155, font_size=6.1)

# 배경 유형 색상 적용
bg_order = ["smooth","vertical_stripe","horizontal_stripe","textured","complex_pattern"]
for ri, bg in enumerate(bg_order):
    ax_c2.text(0.01 + 0.24/2, 0.95 - (ri+1)*0.155 - 0.155/2 + 0.01,
               bg.replace("_","\n"), transform=ax_c2.transAxes,
               ha="center", va="center", fontsize=5.8, fontweight="bold",
               color=BG_COLORS.get(bg, "#333"))

# ── C3: SURFACE_QUALITY ──
ax_c3 = fig.add_subplot(gs_c[0, 2])
ax_c3.axis("off")
ax_c3.set_title("SURFACE_QUALITY\n(Detailed style / stability_score)",
                fontsize=8, fontweight="bold", color="#6A1B9A", pad=4)

surf_rows = [
    ["stability",     "grade",    "vocabulary"],
    ["≥ 0.8",         "high",     "pristine\nwell-maintained\nclean"],
    ["0.5 – 0.8",     "medium",   "standard\ntypical\nnormal"],
    ["< 0.5",         "low",      "worn\nweathered\naged"],
]
grade_colors_map = {"high": "#2E7D32", "medium": "#1565C0", "low": "#6A1B9A"}
grade_bg_map     = {"high": "#E8F5E9", "medium": "#E3F2FD", "low": "#F3E5F5"}

draw_table(ax_c3, surf_rows, col_widths=[0.28, 0.22, 0.50],
           y0=0.95, row_height=0.21, font_size=6.5,
           odd_fc="#F9F9F9", even_fc="white")

# grade 열 색상 재적용
grade_order = ["high", "medium", "low"]
for ri, grade in enumerate(grade_order):
    yy = 0.95 - (ri + 1) * 0.21 - 0.21/2 + 0.02
    ax_c3.add_patch(FancyBboxPatch(
        (0.01 + 0.28, yy - 0.085), 0.22, 0.165,
        fc=grade_bg_map[grade], ec=grade_colors_map[grade], lw=1.0,
        boxstyle="round,pad=0.01", transform=ax_c3.transAxes,
    ))
    ax_c3.text(0.01 + 0.28 + 0.11, yy,
               grade, transform=ax_c3.transAxes,
               ha="center", va="center", fontsize=7, fontweight="bold",
               color=grade_colors_map[grade])

# ── C3 하단: stability → grade → vocab 흐름 주석 ──
ax_c3.text(0.50, 0.02,
           "※ used in generate_detailed_prompt\n(not in simple)",
           transform=ax_c3.transAxes, ha="center", va="bottom",
           fontsize=6, color="#9E9E9E", fontstyle="italic",
           multialignment="center")

# ──────────────────────────────────────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FILE, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"\nSaved → {OUT_FILE}")
plt.show()
