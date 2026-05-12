# Rebuttal Report 1 — Comment 1: Six Figures (Colab Script)

Comment 1에서 약속한 6개 Figure를 Google Colab에서 생성하는 스크립트.

## Figure 목록

| Figure | 내용 | 데이터 소스 |
|--------|------|-------------|
| F1 | CASDA 4-stage 파이프라인 개요 flowchart | matplotlib만 사용 (이미지 불필요) |
| F2 | 3채널 hint 이미지 구성 (6-panel) | roi_patches + controlnet_dataset + augmented_images + casda_composed |
| F3 | Raw / Copy-Paste / CASDA 시각 비교 | train_images + copypaste_baseline + casda_composed |
| F4 | 결함 유형별 분류 예시 (4 types) | roi_patches |
| F5 | Per-class AP 막대 그래프 (3 모델) | 수치 데이터 (Table 10/11/12) |
| F6 | Ablation 결과 막대 그래프 | 수치 데이터 (Table 13) |

## 실행 방법

1. Google Colab에서 아래 코드 블록을 순서대로 실행
2. Google Drive가 마운트되어 있어야 함 (`from google.colab import drive; drive.mount('/content/drive')`)
3. CASDA 리포지토리가 `/content/CASDA`에 있어야 함
4. 생성된 figure는 `/content/CASDA/review/figures/`에 저장됨

---

## 설치 및 경로 설정

```python
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── 출력 디렉토리 ───────────────────────────────────────────────
OUT_DIR = "/content/CASDA/review/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Google Drive 데이터 경로 ────────────────────────────────────
DRIVE_DATA  = "/content/drive/MyDrive/data/Severstal"
ROI_DIR     = f"{DRIVE_DATA}/roi_patches"
CN_DATASET  = f"{DRIVE_DATA}/controlnet_dataset"
AUG_IMAGES  = f"{DRIVE_DATA}/augmented_images"
COPYPASTE   = f"{DRIVE_DATA}/augmented_dataset/copypaste_baseline"
CASDA_COMP  = f"{DRIVE_DATA}/augmented_dataset/casda_composed"
TRAIN_IMGS  = f"{DRIVE_DATA}/train_images"

# ── 공통 스타일 ─────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def load_image(path):
    """BGR→RGB 변환 포함 이미지 로드. 실패 시 None 반환."""
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

def find_first_image(directory, extensions=("*.png", "*.jpg")):
    """디렉토리에서 첫 번째 이미지 경로 반환."""
    for ext in extensions:
        files = sorted(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        if files:
            return files[0]
    return None

def placeholder_panel(ax, message, fontsize=9):
    """이미지 없을 때 플레이스홀더 표시."""
    ax.set_facecolor("#f0f0f0")
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=fontsize, color="#555555", transform=ax.transAxes,
            wrap=True, multialignment="center")
    ax.set_xticks([])
    ax.set_yticks([])

print("설정 완료. 다음 셀부터 Figure 생성.")
```

---

## Figure 1 — CASDA 파이프라인 개요 Flowchart

```python
fig1, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis("off")
fig1.patch.set_facecolor("white")

# ── 박스 정의 ───────────────────────────────────────────────────
stages = [
    {
        "x": 0.3, "label": "Stage A",
        "title": "ROI Extraction\n& Hint Prep",
        "hw": "CPU",
        "items": ["Crop defect ROIs", "Classify defect type",
                  "Build 3-ch hint (R/G/B)", "Compute suitability score"],
        "color": "#D6EAF8", "hw_color": "#2E86C1",
    },
    {
        "x": 3.8, "label": "Stage B",
        "title": "ControlNet Training\n& Generation",
        "hw": "GPU",
        "items": ["Fine-tune ControlNet", "(SD v1.5 + sd-controlnet-canny)",
                  "Generate synthetic ROIs", "Multi-phase validation"],
        "color": "#D5F5E3", "hw_color": "#1E8449",
    },
    {
        "x": 7.3, "label": "Stage C",
        "title": "Poisson Blending\n& Quality Pruning",
        "hw": "CPU",
        "items": ["Blend ROI → background", "Score: color/artifact/sharpness",
                  "Stratified top-k pruning", "CopyPaste baseline generation"],
        "color": "#FEF9E7", "hw_color": "#B7950B",
    },
    {
        "x": 10.8, "label": "Stage D",
        "title": "Evaluation\n& Benchmark",
        "hw": "GPU",
        "items": ["FID (ROI + full image)", "Train YOLO-MFD / EB-YOLOv8",
                  "Train DeepLabV3+", "Statistical hypothesis testing"],
        "color": "#FDEDEC", "hw_color": "#C0392B",
    },
]

BOX_W, BOX_H = 3.2, 3.4

for s in stages:
    x0, y0 = s["x"], 0.3
    # 메인 박스
    box = FancyBboxPatch((x0, y0), BOX_W, BOX_H,
                          boxstyle="round,pad=0.1",
                          facecolor=s["color"], edgecolor="#888888", linewidth=1.5)
    ax.add_patch(box)
    # 스테이지 레이블
    ax.text(x0 + BOX_W / 2, y0 + BOX_H - 0.18, s["label"],
            ha="center", va="top", fontsize=11, fontweight="bold", color="#222222")
    # 타이틀
    ax.text(x0 + BOX_W / 2, y0 + BOX_H - 0.55, s["title"],
            ha="center", va="top", fontsize=9, color="#333333")
    # 하드웨어 배지
    hw_box = FancyBboxPatch((x0 + 0.05, y0 + BOX_H - 1.05), 0.65, 0.28,
                             boxstyle="round,pad=0.05",
                             facecolor=s["hw_color"], edgecolor="none")
    ax.add_patch(hw_box)
    ax.text(x0 + 0.375, y0 + BOX_H - 0.91, s["hw"],
            ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    # 항목 목록
    for j, item in enumerate(s["items"]):
        ax.text(x0 + 0.2, y0 + BOX_H - 1.25 - j * 0.52, f"• {item}",
                ha="left", va="top", fontsize=7.5, color="#444444")

# ── 화살표 ──────────────────────────────────────────────────────
for i in range(len(stages) - 1):
    x_start = stages[i]["x"] + BOX_W
    x_end   = stages[i + 1]["x"]
    y_mid   = 0.3 + BOX_H / 2
    ax.annotate("", xy=(x_end, y_mid), xytext=(x_start, y_mid),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=2.0, mutation_scale=18))

# ── 입출력 레이블 ────────────────────────────────────────────────
ax.text(0.15, 0.3 + BOX_H / 2, "Severstal\nRaw Images",
        ha="center", va="center", fontsize=7, color="#666666",
        style="italic")
ax.text(13.85, 0.3 + BOX_H / 2, "AP / Dice\nResults",
        ha="center", va="center", fontsize=7, color="#666666",
        style="italic")

ax.set_title("Figure 1. Overview of the CASDA Data-Augmentation Pipeline",
             fontsize=12, fontweight="bold", pad=12)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "F1_pipeline_overview.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.show()
```

---

## Figure 2 — 3채널 Hint 이미지 구성

```python
# ── 이미지 로드 ─────────────────────────────────────────────────
# ROI 패치: roi_patches 내 첫 번째 이미지
roi_path = find_first_image(ROI_DIR)

# ControlNet 데이터셋에서 hint/image 쌍 탐색
# 구조 가정: controlnet_dataset/{split}/conditioning_images/*.png
#             controlnet_dataset/{split}/images/*.png
cn_hint_path = find_first_image(os.path.join(CN_DATASET, "train", "conditioning_images"))
if cn_hint_path is None:
    cn_hint_path = find_first_image(CN_DATASET)

cn_gen_path = find_first_image(AUG_IMAGES)
blended_path = find_first_image(CASDA_COMP)

# ── 이미지 로드 ─────────────────────────────────────────────────
roi_img     = load_image(roi_path)     if roi_path     else None
hint_img    = load_image(cn_hint_path) if cn_hint_path else None
gen_img     = load_image(cn_gen_path)  if cn_gen_path  else None
blended_img = load_image(blended_path) if blended_path else None

# hint 이미지에서 R/G/B 채널 분해
r_ch = g_ch = b_ch = None
if hint_img is not None:
    r_ch = np.stack([hint_img[:, :, 0],
                     np.zeros_like(hint_img[:, :, 0]),
                     np.zeros_like(hint_img[:, :, 0])], axis=2)
    g_ch = np.stack([np.zeros_like(hint_img[:, :, 1]),
                     hint_img[:, :, 1],
                     np.zeros_like(hint_img[:, :, 1])], axis=2)
    b_ch = np.stack([np.zeros_like(hint_img[:, :, 2]),
                     np.zeros_like(hint_img[:, :, 2]),
                     hint_img[:, :, 2]], axis=2)

# ── 플롯 ────────────────────────────────────────────────────────
panels = [
    ("Source ROI\n(w/ defect)",   roi_img),
    ("R: Defect\nGeometry Mask",  r_ch),
    ("G: Background\nStructure",  g_ch),
    ("B: Background\nTexture",    b_ch),
    ("ControlNet\nGenerated",     gen_img),
    ("Poisson\nBlended Result",   blended_img),
]

fig2, axes = plt.subplots(1, 6, figsize=(18, 3.5))
fig2.suptitle("Figure 2. Construction of the 3-Channel Hint Image for ControlNet",
              fontsize=11, fontweight="bold")

for ax, (label, img) in zip(axes, panels):
    if img is not None:
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        placeholder_panel(ax, f"{label}\n(이미지 없음)", fontsize=8)
    ax.set_title(label, fontsize=8.5, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#aaaaaa")

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "F2_hint_construction.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.show()
```

---

## Figure 3 — Raw / Copy-Paste / CASDA 시각 비교

```python
# ── 이미지 로드 ─────────────────────────────────────────────────
raw_path   = find_first_image(TRAIN_IMGS)
cp_path    = find_first_image(COPYPASTE)
casda_path = find_first_image(CASDA_COMP)

raw_img   = load_image(raw_path)   if raw_path   else None
cp_img    = load_image(cp_path)    if cp_path    else None
casda_img = load_image(casda_path) if casda_path else None

# ── 결함 영역 크롭 (±50px) ──────────────────────────────────────
def crop_defect_region(img, pad=50):
    """이미지에서 가장 밝은 연결 영역 (결함) 주변 ±pad 크롭."""
    if img is None:
        return None
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # 중앙 256×256 크롭으로 fallback
            h, w = img.shape[:2]
            cy, cx = h // 2, w // 2
            return img[max(0, cy-128):cy+128, max(0, cx-128):cx+128]
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img.shape[1], x + bw + pad)
        y1 = min(img.shape[0], y + bh + pad)
        return img[y0:y1, x0:x1]
    except Exception:
        h, w = img.shape[:2]
        return img[:min(h, 256), :min(w, 256)]

raw_crop   = crop_defect_region(raw_img)
cp_crop    = crop_defect_region(cp_img)
casda_crop = crop_defect_region(casda_img)

# ── 플롯 ────────────────────────────────────────────────────────
fig3, axes = plt.subplots(1, 3, figsize=(12, 4))
fig3.suptitle("Figure 3. Qualitative Comparison of Augmentation Strategies",
              fontsize=11, fontweight="bold")

for ax, (label, img) in zip(axes, [
    ("(a) Raw", raw_crop),
    ("(b) Copy-Paste", cp_crop),
    ("(c) CASDA", casda_crop),
]):
    if img is not None:
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        placeholder_panel(ax, f"{label}\n(이미지 없음)")
    ax.set_title(label, fontsize=10, pad=5)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "F3_visual_comparison.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.show()
```

---

## Figure 4 — 결함 유형별 분류 예시

```python
# ── 유형별 첫 번째 이미지 탐색 ──────────────────────────────────
# 디렉토리 구조 가정 1: roi_patches/{linear_scratch, irregular, compact_blob, general}/
# 디렉토리 구조 가정 2: roi_patches/ 평탄 구조 — 파일명에 유형 포함

DEFECT_TYPES = [
    ("linear_scratch",  "Linear Scratch",  "linearity > 0.85\naspect ratio > 5.0"),
    ("irregular",       "Irregular",       "solidity < 0.7\nnon-linear"),
    ("compact_blob",    "Compact Blob",    "solidity ≥ 0.7\naspect ratio ≤ 5.0"),
    ("general",         "General",         "catch-all\n(other morphologies)"),
]

def find_type_image(roi_dir, type_name):
    """유형 이름 포함 디렉토리 또는 파일명에서 이미지 탐색."""
    # 하위 디렉토리 탐색
    subdir = os.path.join(roi_dir, type_name)
    if os.path.isdir(subdir):
        img = find_first_image(subdir)
        if img:
            return img
    # 파일명 패턴 탐색
    for ext in ("*.png", "*.jpg"):
        files = sorted(glob.glob(os.path.join(roi_dir, f"*{type_name}*"), recursive=False))
        files += sorted(glob.glob(os.path.join(roi_dir, "**", f"*{type_name}*"), recursive=True))
        if files:
            return files[0]
    return None

fig4, axes = plt.subplots(1, 4, figsize=(14, 3.8))
fig4.suptitle("Figure 4. Representative ROI Patches for Each Defect Type Category",
              fontsize=11, fontweight="bold")

for ax, (type_key, type_label, annot_text) in zip(axes, DEFECT_TYPES):
    img_path = find_type_image(ROI_DIR, type_key)
    img = load_image(img_path) if img_path else None

    if img is not None:
        # 128×128으로 리사이즈 (스케일 통일)
        try:
            import cv2
            img_resized = cv2.resize(img, (128, 128))
            ax.imshow(img_resized, cmap="gray" if img_resized.ndim == 2 else None)
        except Exception:
            ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        # 형태 정보 오버레이
        ax.text(0.04, 0.96, annot_text, transform=ax.transAxes,
                fontsize=7, va="top", ha="left", color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
    else:
        placeholder_panel(ax, f"{type_label}\n(이미지 없음)\n\n{annot_text}", fontsize=7.5)

    ax.set_title(type_label, fontsize=9.5, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#aaaaaa")

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "F4_defect_types.png")
fig4.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved: {out4}")
plt.show()
```

---

## Figure 5 — Per-Class AP 막대 그래프 (3 모델)

```python
METHODS   = ["Raw", "Trad.", "Copy-Paste", "CASDA"]
CLASSES   = ["Class 1", "Class 2", "Class 3", "Class 4"]
BAR_COLORS = ["steelblue", "darkorange", "mediumseagreen", "crimson"]

# Table 10: YOLO-MFD per-class AP@0.5
yolo_mfd = np.array([
    [0.5090, 0.4525, 0.6806, 0.5764],  # Raw
    [0.5126, 0.2444, 0.6379, 0.5572],  # Trad.
    [0.5076, 0.3811, 0.6501, 0.5641],  # Copy-Paste
    [0.5298, 0.5317, 0.6886, 0.5839],  # CASDA
])

# Table 11: EB-YOLOv8 per-class AP@0.5
eb_yolo = np.array([
    [0.5397, 0.4895, 0.6875, 0.6107],
    [0.4204, 0.5200, 0.5617, 0.5202],
    [0.5642, 0.4464, 0.6889, 0.6461],
    [0.5445, 0.4663, 0.6842, 0.6452],
])

# Table 12: DeepLabV3+ per-class Dice
deeplab = np.array([
    [0.4926, 0.5170, 0.7296, 0.7767],
    [0.5458, 0.5246, 0.7398, 0.7831],
    [0.5152, 0.4892, 0.7238, 0.7706],
    [0.5150, 0.4961, 0.7213, 0.7603],
])

SUBTITLES = ["YOLO-MFD (mAP@0.5)", "EB-YOLOv8 (mAP@0.5)", "DeepLabV3+ (Dice)"]
DATASETS  = [yolo_mfd, eb_yolo, deeplab]
Y_LIMS    = [(0.20, 0.75), (0.38, 0.73), (0.44, 0.82)]

fig5, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
fig5.suptitle("Figure 5. Per-Class AP / Dice Score by Augmentation Method",
              fontsize=12, fontweight="bold", y=1.01)

x = np.arange(len(CLASSES))
total_width = 0.72
bar_w = total_width / len(METHODS)
offsets = np.linspace(-(total_width / 2) + bar_w / 2,
                      (total_width / 2) - bar_w / 2, len(METHODS))

for ax, data, subtitle, ylim in zip(axes, DATASETS, SUBTITLES, Y_LIMS):
    for i, (method, color) in enumerate(zip(METHODS, BAR_COLORS)):
        edge_kw = dict(edgecolor="black", linewidth=1.2) if method == "CASDA" else {}
        ax.bar(x + offsets[i], data[i], width=bar_w, color=color,
               label=method, **edge_kw)
    ax.set_title(subtitle, fontsize=11, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_ylim(ylim)
    ax.set_ylabel("Score", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

handles = [mpatches.Patch(color=c, label=m,
                          edgecolor="black" if m == "CASDA" else "none",
                          linewidth=1.2 if m == "CASDA" else 0)
           for m, c in zip(METHODS, BAR_COLORS)]
fig5.legend(handles=handles, loc="lower center", ncol=4,
            fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
out5 = os.path.join(OUT_DIR, "F5_per_class_AP.png")
fig5.savefig(out5, dpi=150, bbox_inches="tight")
print(f"Saved: {out5}")
plt.show()
```

---

## Figure 6 — Ablation 결과 막대 그래프

```python
ABL_METHODS = ["CASDA-Full", "w/o Pruning", "w/o Blending"]
ABL_MAP     = [0.5835, 0.5826, 0.4697]          # Table 13 overall mAP
ABL_COLORS  = ["seagreen", "steelblue", "crimson"]

# 가장 영향받은 클래스 AP (Table 13)
# Class 2: CASDA-Full=0.5317, w/o Pruning=0.5317(변화없음), w/o Blending=0.2958
# Class 4: CASDA-Full=0.5839, w/o Pruning=0.5572, w/o Blending=0.5572(변화없음)
CLASS2_AP = [0.5317, 0.5317, 0.2958]
CLASS4_AP = [0.5839, 0.5572, 0.5572]

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(11, 4.5))
fig6.suptitle("Figure 6. YOLO-MFD Ablation Study Results (Table 13)",
              fontsize=11, fontweight="bold", y=1.01)

x3 = np.arange(len(ABL_METHODS))

# Subplot 1: Overall mAP
bars1 = ax6a.bar(x3, ABL_MAP, color=ABL_COLORS, width=0.5,
                 edgecolor="black", linewidth=0.8)
ax6a.set_title("Overall mAP@0.5", fontsize=11, pad=6)
ax6a.set_xticks(x3)
ax6a.set_xticklabels(ABL_METHODS, fontsize=9)
ax6a.set_ylabel("mAP@0.5", fontsize=10)
ax6a.set_ylim(0.43, 0.61)
ax6a.tick_params(axis="y", labelsize=9)
ax6a.grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(bars1, ABL_MAP):
    ax6a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
              f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Subplot 2: Most-affected class AP
cls_labels = ["Class 2\n(w/o Blending\nmost affected)",
              "Class 4\n(w/o Pruning\nmost affected)"]
cls_data   = np.array([CLASS2_AP, CLASS4_AP])   # (2, 3)
x2, sub_w, sub_off = np.arange(2), 0.22, np.array([-0.22, 0, 0.22])

for i, (method, color) in enumerate(zip(ABL_METHODS, ABL_COLORS)):
    b = ax6b.bar(x2 + sub_off[i], cls_data[:, i], width=sub_w,
                 color=color, label=method, edgecolor="black", linewidth=0.8)
    for bar, v in zip(b, cls_data[:, i]):
        ax6b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                  f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)

ax6b.set_title("Most-Affected Class AP", fontsize=11, pad=6)
ax6b.set_xticks(x2)
ax6b.set_xticklabels(cls_labels, fontsize=8.5)
ax6b.set_ylabel("AP@0.5", fontsize=10)
ax6b.set_ylim(0.22, 0.65)
ax6b.tick_params(axis="y", labelsize=9)
ax6b.grid(axis="y", linestyle="--", alpha=0.4)

handles6 = [mpatches.Patch(color=c, label=m, edgecolor="black", linewidth=0.8)
            for m, c in zip(ABL_METHODS, ABL_COLORS)]
fig6.legend(handles=handles6, loc="lower center", ncol=3,
            fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
out6 = os.path.join(OUT_DIR, "F6_ablation.png")
fig6.savefig(out6, dpi=150, bbox_inches="tight")
print(f"Saved: {out6}")
plt.show()
```

---

## 출력 파일 확인

```python
print("=== 생성된 Figure 파일 ===")
for fname in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, fname)
    size_kb = os.path.getsize(fpath) // 1024
    print(f"  {fname}  ({size_kb} KB)")
```

---

## 주의사항

- **F2, F3, F4**는 실제 이미지 파일이 Drive 경로에 존재해야 함. 없으면 placeholder로 표시됨.
- F2는 ROI 패치와 ControlNet dataset 이미지의 **대응 관계**가 이상적. 현재 스크립트는 첫 번째 사용 가능 파일을 사용.
- F3에서 3개 방법의 **동일 배경** 이미지를 사용하면 비교 효과가 극대화됨. 필요시 `raw_path`, `cp_path`, `casda_path`를 동일 파일명 기준으로 수동 지정할 것.
- F4의 결함 유형별 이미지는 `roi_patches/{type_name}/` 하위 디렉토리 구조를 우선 탐색.
