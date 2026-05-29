"""
Stage 0: CASDA Dataset Analysis

Analyzes the full Severstal Steel Defect Detection dataset to derive
data-driven thresholds for 19 hardcoded values across:
  - DefectCharacterizer.classify_defect_subtype()
  - BackgroundAnalyzer.classify_patch()
  - Pipeline parameters (num_images_per_class, compositions-per-roi, etc.)

Outputs (ANALYSIS_DIR/):
  class_distribution.json
  morphological_features.csv
  background_features.csv
  defect_bg_matrix_4x5.json
  threshold_recommendations.json
  recommended_config.yaml
  analysis_report.md
  figures/  (9 PNG files)

Usage (Google Colab Pro):
    !python scripts/analyze_dataset.py \\
        --image_dir /content/drive/MyDrive/data/Severstal/train_images \\
        --train_csv /content/drive/MyDrive/data/Severstal/train.csv \\
        --output_dir /content/drive/MyDrive/data/Severstal/analysis \\
        --num_workers 8

    # Quick test (first 100 images):
    !python scripts/analyze_dataset.py \\
        --image_dir ... --train_csv ... --output_dir ... \\
        --max_images 100 --num_workers 2
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.background_characterization import BackgroundAnalyzer
from src.analysis.defect_characterization import DefectCharacterizer
from src.analysis.roi_suitability import ROISuitabilityEvaluator
from src.utils.rle_utils import build_image_index, get_all_masks_for_image

# ──────────────────────────────────────────────────────────────────────────────
# Column schemas (positional tuples — single source of truth)
# ──────────────────────────────────────────────────────────────────────────────

MORPH_COLS = (
    "image_id", "class_id", "region_id", "area",
    "linearity", "solidity", "extent", "aspect_ratio",
    "defect_subtype", "background_type",
)
# Index shortcuts
_M_CLS  = 1; _M_AREA = 3; _M_LIN = 4; _M_SOL = 5
_M_EXT  = 6; _M_ASP  = 7; _M_SUB = 8; _M_BG  = 9

BG_COLS = (
    "image_id", "grid_i", "grid_j",
    "background_type", "stability_score",
    "variance",
    "edge_vertical", "edge_horizontal", "edge_total", "edge_magnitude_std",
    "v_ratio", "h_ratio",
    "high_freq_ratio", "total_energy", "center_energy",
)
# Index shortcuts
_B_BGT  = 3; _B_VAR  = 5; _B_ETOT = 8
_B_VRAT = 10; _B_HRAT = 11; _B_HFQ = 12

BG_TYPES  = ["smooth", "textured", "vertical_stripe", "horizontal_stripe", "complex_pattern"]
SUBTYPES  = ["linear_scratch", "irregular", "elongated", "compact_blob", "general"]
CLASS_IDS = [1, 2, 3, 4]

# Current hardcoded values — used as fallback and for "current" column in report
HARDCODED = {
    "HIGH_LINEARITY":        0.85,
    "HIGH_ASPECT_RATIO":     5.0,
    "LOW_ASPECT_RATIO":      2.0,
    "HIGH_SOLIDITY":         0.9,
    "LOW_SOLIDITY":          0.7,
    "elongated_linearity":   0.6,
    "variance_threshold":    100.0,
    "edge_threshold":        0.3,
    "total_strength":        1.0,
    "stripe_ratio_v":        1.5,
    "stripe_ratio_h":        1.5,
    "high_freq_ratio":       0.3,
    "min_suitability":       0.5,
    "min_quality_score":     0.5,
}

# ──────────────────────────────────────────────────────────────────────────────
# Worker — per-process initializer + analysis function
# ──────────────────────────────────────────────────────────────────────────────

_W: dict = {}  # module-level globals populated by initializer


def _init_worker(train_csv_str: str, image_dir_str: str, grid_size: int) -> None:
    """Called once per worker process. Parses CSV and creates analyzer instances."""
    df = pd.read_csv(train_csv_str)
    _W["index"]   = build_image_index(df)
    _W["df"]      = df
    _W["img_dir"] = Path(image_dir_str)
    _W["dc"]      = DefectCharacterizer()
    _W["ba"]      = BackgroundAnalyzer(grid_size=grid_size)
    _W["gs"]      = grid_size


def _analyze_one_image(image_id: str):
    """
    Analyze one image.
    Returns (image_id, morph_tuples, bg_tuples, cooccur_tuples).
    Tuples are positional, matching MORPH_COLS / BG_COLS.
    """
    dc  = _W["dc"];  ba  = _W["ba"]
    gs  = _W["gs"];  idx = _W["index"];  df = _W["df"]
    img_path = _W["img_dir"] / image_id

    if not img_path.exists():
        return image_id, [], [], []

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return image_id, [], [], []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # ── Background: analyze_image for classification + raw features per patch ──
    bg_analysis = ba.analyze_image(img_rgb)
    grid_h, grid_w = bg_analysis["grid_shape"]

    bg_tuples = []
    for i in range(grid_h):
        for j in range(grid_w):
            patch = gray[i * gs: (i + 1) * gs, j * gs: (j + 1) * gs]
            if patch.size == 0:
                continue
            var  = ba.compute_variance(patch)
            edge = ba.compute_edge_directions(patch)
            freq = ba.compute_frequency_spectrum(patch)
            tot  = edge["total"] + 1e-6
            v_r  = edge["vertical"] / tot
            h_r  = edge["horizontal"] / tot
            bg_tuples.append((
                image_id, i, j,
                bg_analysis["background_map"][i, j],
                round(float(bg_analysis["stability_map"][i, j]), 6),
                round(var, 4),
                round(edge["vertical"], 4), round(edge["horizontal"], 4),
                round(edge["total"], 4),    round(edge["magnitude_std"], 4),
                round(v_r, 6), round(h_r, 6),
                round(freq["high_freq_ratio"], 6),
                round(freq["total_energy"], 2), round(freq["center_energy"], 2),
            ))

    # ── Morphological: all defect instances, bg_type at centroid ──────────────
    masks = get_all_masks_for_image(image_id, df, shape=(h, w), image_index=idx)
    morph_tuples   = []
    cooccur_tuples = []

    for class_id, mask in masks.items():
        for m in dc.analyze_all_defects_in_mask(mask, class_id):
            subtype = dc.classify_defect_subtype(m)
            cx, cy  = m["centroid"]
            bg_info = ba.get_background_at_location(bg_analysis, int(cx), int(cy))
            bg_at   = bg_info["background_type"] if bg_info else "unknown"

            morph_tuples.append((
                image_id, class_id, m["region_id"], m["area"],
                round(m["linearity"], 6),    round(m["solidity"], 6),
                round(m["extent"], 6),        round(m["aspect_ratio"], 6),
                subtype, bg_at,
            ))
            cooccur_tuples.append((class_id, subtype, bg_at))

    return image_id, morph_tuples, bg_tuples, cooccur_tuples


# ──────────────────────────────────────────────────────────────────────────────
# Threshold derivation
# ──────────────────────────────────────────────────────────────────────────────

def _valley_threshold(values: np.ndarray, bins: int = 64) -> float | None:
    """Find bimodal separator via histogram valley. Returns None if not bimodal."""
    hist, edges = np.histogram(values, bins=bins)
    centers     = (edges[:-1] + edges[1:]) / 2
    smoothed    = uniform_filter1d(hist.astype(float), size=3)

    peaks, _ = find_peaks(smoothed, prominence=smoothed.max() * 0.05)
    if len(peaks) < 2:
        return None

    top2 = sorted(peaks, key=lambda i: smoothed[i])[-2:]
    lp, rp = sorted(top2)

    inv = smoothed.max() - smoothed
    valleys, _ = find_peaks(inv, prominence=smoothed.max() * 0.1)
    inner = [v for v in valleys if lp < v < rp]
    if not inner:
        return None

    best = min(inner, key=lambda i: smoothed[i])
    return float(centers[best])


def _otsu_1d(values: np.ndarray) -> float | None:
    """Otsu's method for 1-D continuous values. Returns None on failure."""
    n = 256
    hist, edges = np.histogram(values, bins=n)
    centers     = (edges[:-1] + edges[1:]) / 2
    total       = int(hist.sum())
    if total == 0:
        return None

    sum_total = float(np.dot(centers, hist))
    sum_bg = w_bg = 0.0
    best_var = 0.0
    threshold = None

    for i in range(n):
        w_bg  += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg  += centers[i] * hist[i]
        mean_bg  = sum_bg / w_bg
        mean_fg  = (sum_total - sum_bg) / w_fg
        var      = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var  = var
            threshold = centers[i]

    return threshold


def derive_threshold(
    values: np.ndarray,
    hard_default: float,
    pct_fallback: int = 75,
) -> tuple[float, str]:
    """
    Ladder: valley → Otsu_1D → percentile → hardcoded.
    Returns (recommended_value, method_string).
    """
    finite = values[np.isfinite(values)] if len(values) else values
    if len(finite) < 50:
        return hard_default, "hardcoded(insufficient_n)"

    v = _valley_threshold(finite)
    if v is not None and float(finite.min()) < v < float(finite.max()):
        return round(v, 4), "valley"

    o = _otsu_1d(finite)
    if o is not None and float(finite.min()) < o < float(finite.max()):
        return round(float(o), 4), "otsu_1d"

    p = float(np.percentile(finite, pct_fallback))
    return round(p, 4), f"p{pct_fallback}"


# ──────────────────────────────────────────────────────────────────────────────
# 4×5 matrix + generation parameter builders
# ──────────────────────────────────────────────────────────────────────────────

def build_matrix_and_params(cooccur, class_counts):
    """
    Build 4×5 (class × bg_type) co-occurrence matrix and derive:
      - target_synthetic  (deficit-weighted synthesis targets)
      - compatibility_score (0.6 × MATCHING_RULES + 0.4 × empirical)
      - num_images_per_class, compositions_per_roi, per_class_cap, rare_class_threshold
    """
    bg_idx = {bg: i for i, bg in enumerate(BG_TYPES)}

    M           = np.zeros((4, 5), dtype=np.int64)
    subtype_dist = {c: Counter() for c in CLASS_IDS}

    for class_id, subtype, bg in cooccur:
        if 1 <= class_id <= 4 and bg in bg_idx:
            M[class_id - 1, bg_idx[bg]] += 1
        if 1 <= class_id <= 4:
            subtype_dist[class_id][subtype] += 1

    row_tot = M.sum(axis=1, keepdims=True).clip(1)
    P       = M / row_tot                           # P(bg | class)
    global_bg = M.sum(axis=0) / max(int(M.sum()), 1)

    rules = ROISuitabilityEvaluator.MATCHING_RULES

    matrix_out = {}
    compat_out = {}
    target_out = {}

    for c in CLASS_IDS:
        deficit = np.clip(global_bg - P[c - 1], 0, None)
        dsum    = float(deficit.sum())
        dominant = subtype_dist[c].most_common(1)[0][0] if subtype_dist[c] else "general"
        rule_row = rules.get(dominant, rules["general"])

        row = {}
        compat_row = {}
        target_row = {}
        for j, bg in enumerate(BG_TYPES):
            comp = round(0.6 * rule_row.get(bg, 0.5) + 0.4 * float(P[c - 1, j]), 4)
            tgt  = round(float(deficit[j] / dsum), 4) if dsum > 0 else round(1.0 / 5, 4)
            row[bg] = {
                "count_real":          int(M[c - 1, j]),
                "target_synthetic":    tgt,
                "compatibility_score": comp,
            }
            compat_row[bg] = comp
            target_row[bg] = tgt

        matrix_out[f"class_{c}"]  = row
        compat_out[c]             = compat_row
        target_out[c]             = target_row

    # Pipeline parameters derived from class distribution
    counts_arr = np.array([class_counts.get(c, 0) for c in CLASS_IDS], dtype=float)
    nz = counts_arr[counts_arr > 0]
    per_class_cap         = int(np.percentile(nz, 90))  if len(nz) else 1200
    rare_class_threshold  = int(np.percentile(nz, 25))  if len(nz) else 200

    max_cnt = max(class_counts.values()) if class_counts else 1
    num_images = {}
    for c in CLASS_IDS:
        cnt = class_counts.get(c, 0)
        num_images[c] = max(1, round(max_cnt / cnt)) if cnt > 0 else 10

    avg_deficit = float(np.clip(global_bg - P, 0, None).mean())
    comp_per_roi = max(3, min(10, round(5 + avg_deficit * 20)))

    return matrix_out, {
        "num_images_per_class":  {str(c): num_images[c] for c in CLASS_IDS},
        "compositions_per_roi":  comp_per_roi,
        "per_class_cap":         per_class_cap,
        "rare_class_threshold":  rare_class_threshold,
        "compatibility_matrix":  compat_out,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figure rendering (9 figures)
# ──────────────────────────────────────────────────────────────────────────────

def _savefig(figures_dir: Path, fname: str) -> None:
    path = figures_dir / fname
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  {fname}")


def render_figures(
    class_counts, subtype_counts, morph_arrays,
    bg_type_counts, bg_arrays, matrix_out,
    area_by_class, threshold_recs, figures_dir: Path,
) -> None:

    def _vline(ax, key, label_prefix="rec"):
        if key in threshold_recs:
            val = threshold_recs[key]["recommended"]
            ax.axvline(val, color="red", linestyle="--", lw=1.4,
                       label=f"{label_prefix}={val}")
            ax.legend(fontsize=8)

    # Fig 1: Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [class_counts.get(c, 0) for c in CLASS_IDS]
    bars   = ax.bar([f"Class {c}" for c in CLASS_IDS], counts,
                    color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Instances"); ax.set_title("Defect Class Distribution")
    plt.tight_layout()
    _savefig(figures_dir, "fig1_class_distribution.png")

    # Fig 2: Subtype distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sub_counts = [subtype_counts.get(s, 0) for s in SUBTYPES]
    bars = ax.bar(SUBTYPES, sub_counts, color="coral", edgecolor="white")
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Count"); ax.set_title("Defect Subtype Distribution")
    plt.xticks(rotation=18, ha="right"); plt.tight_layout()
    _savefig(figures_dir, "fig2_subtype_distribution.png")

    # Fig 3: Morphological histograms (2×2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    specs = [
        ("linearity",    "HIGH_LINEARITY",    axes[0, 0]),
        ("aspect_ratio", "HIGH_ASPECT_RATIO", axes[0, 1]),
        ("solidity",     "HIGH_SOLIDITY",     axes[1, 0]),
        ("extent",       None,                axes[1, 1]),
    ]
    for fname, tkey, ax in specs:
        arr = morph_arrays.get(fname, [])
        if len(arr):
            ax.hist(arr, bins=50, color="mediumseagreen", edgecolor="white", alpha=0.85)
        if tkey:
            _vline(ax, tkey)
        ax.set_title(fname); ax.set_xlabel("Value"); ax.set_ylabel("Count")
    fig.suptitle("Morphological Feature Distributions", fontweight="bold")
    plt.tight_layout()
    _savefig(figures_dir, "fig3_morph_features_hist.png")

    # Fig 4: Variance histogram (SMOOTH boundary)
    fig, ax = plt.subplots(figsize=(7, 4))
    arr = np.array(bg_arrays.get("variance", []))
    if len(arr):
        clip = np.percentile(arr, 99)
        ax.hist(arr[arr <= clip], bins=60, color="slateblue",
                edgecolor="white", alpha=0.85)
    _vline(ax, "variance_threshold")
    ax.set_title("Background Variance (SMOOTH boundary)")
    ax.set_xlabel("Variance"); ax.set_ylabel("Patch count")
    _savefig(figures_dir, "fig4_bg_variance_hist.png")

    # Fig 5: Edge histograms (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    edge_specs = [
        ("edge_total", "total_strength",  axes[0], "Edge Total Strength"),
        ("v_ratio",    "edge_threshold",  axes[1], "v_ratio (VERTICAL_STRIPE)"),
        ("h_ratio",    "edge_threshold",  axes[2], "h_ratio (HORIZONTAL_STRIPE)"),
    ]
    for fname, tkey, ax, title in edge_specs:
        arr = np.array(bg_arrays.get(fname, []))
        if len(arr):
            ax.hist(arr, bins=50, color="darkorange", edgecolor="white", alpha=0.85)
        _vline(ax, tkey)
        ax.set_title(title); ax.set_xlabel("Value"); ax.set_ylabel("Count")
    fig.suptitle("Edge Feature Distributions", fontweight="bold")
    plt.tight_layout()
    _savefig(figures_dir, "fig5_bg_edge_hist.png")

    # Fig 6: High-frequency ratio (COMPLEX_PATTERN boundary)
    fig, ax = plt.subplots(figsize=(7, 4))
    arr = np.array(bg_arrays.get("high_freq_ratio", []))
    if len(arr):
        ax.hist(arr, bins=50, color="mediumpurple", edgecolor="white", alpha=0.85)
    _vline(ax, "high_freq_ratio")
    ax.set_title("High-Frequency Ratio (COMPLEX_PATTERN boundary)")
    ax.set_xlabel("high_freq_ratio"); ax.set_ylabel("Patch count")
    _savefig(figures_dir, "fig6_bg_freq_hist.png")

    # Fig 7: Background type distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    bt_counts = [bg_type_counts.get(bg, 0) for bg in BG_TYPES]
    bars = ax.bar(BG_TYPES, bt_counts, color="teal", edgecolor="white")
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Grid Cell Count"); ax.set_title("Background Type Distribution")
    plt.xticks(rotation=18, ha="right"); plt.tight_layout()
    _savefig(figures_dir, "fig7_bg_type_distribution.png")

    # Fig 8: 4×5 Defect–Background heatmap
    fig, ax = plt.subplots(figsize=(9, 5))
    heat = np.array([
        [matrix_out[f"class_{c}"][bg]["count_real"] for bg in BG_TYPES]
        for c in CLASS_IDS
    ], dtype=float)
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="count_real")
    ax.set_xticks(range(5)); ax.set_xticklabels(BG_TYPES, rotation=18, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels([f"Class {c}" for c in CLASS_IDS])
    for i in range(4):
        for j in range(5):
            color = "white" if heat[i, j] > heat.max() * 0.6 else "black"
            ax.text(j, i, int(heat[i, j]), ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_title("Defect × Background Co-occurrence (4×5 Matrix)")
    plt.tight_layout()
    _savefig(figures_dir, "fig8_defect_bg_heatmap.png")

    # Fig 9: Per-class area boxplot (log scale)
    fig, ax = plt.subplots(figsize=(7, 4))
    data   = [np.log1p(area_by_class.get(c, [1])) for c in CLASS_IDS]
    labels = [f"Class {c}" for c in CLASS_IDS]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_ylabel("log(1 + area)  [px²]")
    ax.set_title("Defect Area Distribution by Class")
    _savefig(figures_dir, "fig9_per_class_area_box.png")


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def render_report(
    n_images, n_defects, n_bg_cells,
    class_counts, subtype_counts, bg_type_counts,
    threshold_recs, matrix_out, param_recs,
    output_dir: Path,
) -> None:
    total = sum(class_counts.values()) or 1
    lines = [
        "# CASDA Dataset Analysis Report (Stage 0)",
        "",
        f"> Images analyzed: **{n_images:,}** | "
        f"Defect instances: **{n_defects:,}** | "
        f"BG grid cells: **{n_bg_cells:,}**",
        "",
        "---",
        "",
        "## 1. Class Distribution",
        "",
        "| Class | Instances | % of total |",
        "|-------|-----------|------------|",
    ]
    for c in CLASS_IDS:
        cnt = class_counts.get(c, 0)
        lines.append(f"| Class {c} | {cnt:,} | {100 * cnt / total:.1f}% |")

    lines += [
        "",
        "![Class Distribution](figures/fig1_class_distribution.png)",
        "![Subtype Distribution](figures/fig2_subtype_distribution.png)",
        "",
        "---",
        "",
        "## 2. Morphological Feature Distributions",
        "",
        "![Morphological Features](figures/fig3_morph_features_hist.png)",
        "",
        "---",
        "",
        "## 3. Background Feature Distributions",
        "",
        "![Variance](figures/fig4_bg_variance_hist.png)",
        "![Edge](figures/fig5_bg_edge_hist.png)",
        "![Frequency](figures/fig6_bg_freq_hist.png)",
        "![BG Type](figures/fig7_bg_type_distribution.png)",
        "",
        "---",
        "",
        "## 4. Defect–Background Co-occurrence (4×5 Matrix)",
        "",
        "![Heatmap](figures/fig8_defect_bg_heatmap.png)",
        "![Area](figures/fig9_per_class_area_box.png)",
        "",
        "---",
        "",
        "## 5. Recommended Thresholds",
        "",
        "| Parameter | Current | Recommended | Method | Derivable |",
        "|-----------|---------|-------------|--------|-----------|",
    ]
    for name, info in threshold_recs.items():
        flag = "✓" if info.get("derivable", True) else "✗ Stage A 이후"
        lines.append(
            f"| `{name}` | {info['current']} | {info['recommended']} "
            f"| {info['method']} | {flag} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Generation Plan",
        "",
        "| Parameter | Recommended |",
        "|-----------|-------------|",
        f"| `num_images_per_class` | {param_recs['num_images_per_class']} |",
        f"| `compositions_per_roi` | {param_recs['compositions_per_roi']} |",
        f"| `per_class_cap` | {param_recs['per_class_cap']} |",
        f"| `rare_class_threshold` | {param_recs['rare_class_threshold']} |",
        "",
        "---",
        "",
        "## 7. Artifacts",
        "",
        f"All outputs in: `{output_dir}`",
        "",
        "- `class_distribution.json`",
        "- `morphological_features.csv`",
        "- `background_features.csv`",
        "- `defect_bg_matrix_4x5.json`",
        "- `threshold_recommendations.json`",
        "- `recommended_config.yaml`  ← apply to Stage A~C before re-running",
        "- `figures/` (9 PNG files)",
    ]

    (output_dir / "analysis_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("  analysis_report.md")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0: CASDA Dataset Analysis"
    )
    parser.add_argument("--image_dir",   required=True,
                        help="Severstal training images directory")
    parser.add_argument("--train_csv",   required=True,
                        help="Path to train.csv (RLE annotations)")
    parser.add_argument("--output_dir",  required=True,
                        help="Output directory for analysis artifacts")
    parser.add_argument("--num_workers", type=int,
                        default=min(os.cpu_count() or 4, 8),
                        help="Worker processes (default: min(cpu_count, 8))")
    parser.add_argument("--grid_size",   type=int, default=64,
                        help="Background grid size in pixels (default: 64)")
    parser.add_argument("--max_images",  type=int, default=None,
                        help="Max images to process — for smoke testing")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    image_dir   = Path(args.image_dir)
    train_csv   = Path(args.train_csv)
    output_dir  = Path(args.output_dir)
    figures_dir = output_dir / "figures"

    if not image_dir.exists():
        print(f"Error: image_dir not found: {image_dir}"); return
    if not train_csv.exists():
        print(f"Error: train_csv not found: {train_csv}"); return

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("CASDA Stage 0 — Dataset Analysis")
    print("=" * 70)
    print(f"  image_dir  : {image_dir}")
    print(f"  train_csv  : {train_csv}")
    print(f"  output_dir : {output_dir}")
    print(f"  workers    : {args.num_workers}")
    print(f"  grid_size  : {args.grid_size}")
    print(f"  max_images : {args.max_images or 'all'}")
    print("=" * 70)

    # ── [1/8] Image list ─────────────────────────────────────────────────────
    print("\n[1/8] Building image list...")
    all_ids = sorted(
        p.name for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if args.max_images:
        all_ids = all_ids[:args.max_images]
    print(f"  {len(all_ids):,} images")

    # ── [2/8] Parallel analysis + streaming write ─────────────────────────────
    print("\n[2/8] Analyzing images (streaming write)...")

    morph_csv = output_dir / "morphological_features.csv"
    bg_csv    = output_dir / "background_features.csv"

    class_counts   = Counter()
    subtype_counts = Counter()
    bg_type_counts = Counter()
    area_by_class  = defaultdict(list)
    cooccur        = []

    morph_arrays = {k: [] for k in ("linearity", "solidity", "extent", "aspect_ratio")}
    bg_arrays    = {k: [] for k in ("variance", "edge_total", "v_ratio", "h_ratio", "high_freq_ratio")}

    n_morph = n_bg = n_failed = 0

    with (
        open(morph_csv, "w", newline="", encoding="utf-8") as mf,
        open(bg_csv,    "w", newline="", encoding="utf-8") as bf,
    ):
        mw = csv.writer(mf); mw.writerow(MORPH_COLS)
        bw = csv.writer(bf); bw.writerow(BG_COLS)

        # in-flight 제한: IPC 큐 메모리 누적 방지
        max_inflight = args.num_workers * 2
        ids_iter     = iter(all_ids)
        pending: dict = {}
        done_count   = 0

        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(str(train_csv), str(image_dir), args.grid_size),
        ) as ex:
            pbar = tqdm(total=len(all_ids), desc="images")

            # seed initial batch
            for iid in ids_iter:
                pending[ex.submit(_analyze_one_image, iid)] = iid
                if len(pending) >= max_inflight:
                    break

            while pending:
                fut = next(as_completed(pending))
                iid = pending.pop(fut)

                try:
                    _, morph_rows, bg_rows, cooccur_rows = fut.result()
                except Exception as e:
                    print(f"\n  [WARN] {iid}: {e}")
                    n_failed += 1
                else:
                    mw.writerows(morph_rows)
                    bw.writerows(bg_rows)
                    cooccur.extend(cooccur_rows)
                    n_morph += len(morph_rows)
                    n_bg    += len(bg_rows)

                    for r in morph_rows:
                        c = r[_M_CLS]
                        class_counts[c]             += 1
                        subtype_counts[r[_M_SUB]]   += 1
                        area_by_class[c].append(r[_M_AREA])
                        morph_arrays["linearity"].append(r[_M_LIN])
                        morph_arrays["solidity"].append(r[_M_SOL])
                        morph_arrays["extent"].append(r[_M_EXT])
                        morph_arrays["aspect_ratio"].append(r[_M_ASP])

                    for r in bg_rows:
                        bg_type_counts[r[_B_BGT]] += 1
                        bg_arrays["variance"].append(r[_B_VAR])
                        bg_arrays["edge_total"].append(r[_B_ETOT])
                        bg_arrays["v_ratio"].append(r[_B_VRAT])
                        bg_arrays["h_ratio"].append(r[_B_HRAT])
                        bg_arrays["high_freq_ratio"].append(r[_B_HFQ])

                done_count += 1
                pbar.update(1)
                if done_count % 500 == 0:
                    mf.flush(); bf.flush()

                # refill pipeline
                nxt = next(ids_iter, None)
                if nxt is not None:
                    pending[ex.submit(_analyze_one_image, nxt)] = nxt

            pbar.close()

    print(f"  morphological instances : {n_morph:,}")
    print(f"  background grid cells   : {n_bg:,}")
    if n_failed:
        print(f"  failed images           : {n_failed}")

    for k in list(morph_arrays):
        tmp = morph_arrays[k]
        morph_arrays[k] = np.array(tmp, dtype=np.float64)
        del tmp
    for k in list(bg_arrays):
        tmp = bg_arrays[k]
        bg_arrays[k] = np.array(tmp, dtype=np.float64)
        del tmp

    # ── [3/8] Class distribution ─────────────────────────────────────────────
    print("\n[3/8] Class distribution...")
    class_dist = {}
    for c in CLASS_IDS:
        areas = area_by_class.get(c, [])
        class_dist[f"class_{c}"] = {
            "count": int(class_counts.get(c, 0)),
            "area_mean": round(float(np.mean(areas)), 1) if areas else 0,
            "area_p25":  round(float(np.percentile(areas, 25)), 1) if areas else 0,
            "area_p75":  round(float(np.percentile(areas, 75)), 1) if areas else 0,
        }
    class_dist["subtype_counts"] = dict(subtype_counts)
    class_dist["bg_type_counts"] = dict(bg_type_counts)

    with open(output_dir / "class_distribution.json", "w") as f:
        json.dump(class_dist, f, indent=2)
    print(f"  {dict(class_counts)}")

    # ── [4/8] Threshold derivation ───────────────────────────────────────────
    print("\n[4/8] Deriving thresholds...")

    arr = morph_arrays
    bar = bg_arrays
    stripe_ratios = np.concatenate([
        bar["v_ratio"] / (bar["h_ratio"] + 1e-6),
        bar["h_ratio"] / (bar["v_ratio"] + 1e-6),
    ])

    threshold_specs = [
        # (name,                  array,                  default,  pct,  derivable)
        ("HIGH_LINEARITY",        arr["linearity"],        0.85,    85,   True),
        ("HIGH_ASPECT_RATIO",     arr["aspect_ratio"],     5.0,     90,   True),
        ("LOW_ASPECT_RATIO",      arr["aspect_ratio"],     2.0,     25,   True),
        ("HIGH_SOLIDITY",         arr["solidity"],         0.9,     90,   True),
        ("LOW_SOLIDITY",          arr["solidity"],         0.7,     25,   True),
        ("elongated_linearity",   arr["linearity"],        0.6,     50,   True),
        ("variance_threshold",    bar["variance"],         100.0,   50,   True),
        ("edge_threshold",        bar["v_ratio"],          0.3,     75,   True),
        ("total_strength",        bar["edge_total"],       1.0,     10,   True),
        ("stripe_ratio_v",        stripe_ratios,           1.5,     75,   True),
        ("stripe_ratio_h",        stripe_ratios,           1.5,     75,   True),
        ("high_freq_ratio",       bar["high_freq_ratio"],  0.3,     50,   True),
        ("min_suitability",       np.array([]),            0.5,     25,   False),
        ("min_quality_score",     np.array([]),            0.5,     25,   False),
    ]

    threshold_recs = {}
    for name, data, default, pct, derivable in threshold_specs:
        if derivable:
            val, method = derive_threshold(data, default, pct)
        else:
            val, method = default, "hardcoded(not_derivable_in_stage0)"
        threshold_recs[name] = {
            "current":     default,
            "recommended": val,
            "method":      method,
            "derivable":   derivable,
        }
        flag = "✓" if derivable else "✗"
        print(f"  {flag} {name:<26} {default!s:<7} → {val!s:<9} [{method}]")

    with open(output_dir / "threshold_recommendations.json", "w") as f:
        json.dump(threshold_recs, f, indent=2)

    # ── [5/8] 4×5 matrix + generation parameters ─────────────────────────────
    print("\n[5/8] 4×5 matrix + generation parameters...")
    matrix_out, param_recs = build_matrix_and_params(cooccur, class_counts)

    with open(output_dir / "defect_bg_matrix_4x5.json", "w") as f:
        json.dump(matrix_out, f, indent=2)
    print(f"  num_images_per_class : {param_recs['num_images_per_class']}")
    print(f"  compositions_per_roi : {param_recs['compositions_per_roi']}")
    print(f"  per_class_cap        : {param_recs['per_class_cap']}")
    print(f"  rare_class_threshold : {param_recs['rare_class_threshold']}")

    # ── [6/8] recommended_config.yaml ────────────────────────────────────────
    print("\n[6/8] Writing recommended_config.yaml...")
    config = {
        "stage0_analysis": {
            "n_images":   len(all_ids),
            "n_defects":  n_morph,
            "n_bg_cells": n_bg,
        },
        "defect_characterizer_thresholds": {
            k: threshold_recs[k]["recommended"]
            for k in [
                "HIGH_LINEARITY", "HIGH_ASPECT_RATIO", "LOW_ASPECT_RATIO",
                "HIGH_SOLIDITY", "LOW_SOLIDITY", "elongated_linearity",
            ]
        },
        "background_analyzer_thresholds": {
            k: threshold_recs[k]["recommended"]
            for k in [
                "variance_threshold", "edge_threshold", "total_strength",
                "stripe_ratio_v", "stripe_ratio_h", "high_freq_ratio",
            ]
        },
        "pipeline_parameters": {
            "min_suitability":       threshold_recs["min_suitability"]["recommended"],
            "per_class_cap":         param_recs["per_class_cap"],
            "rare_class_threshold":  param_recs["rare_class_threshold"],
            "num_images_per_class":  param_recs["num_images_per_class"],
            "compositions_per_roi":  param_recs["compositions_per_roi"],
            "min_quality_score":     threshold_recs["min_quality_score"]["recommended"],
            "compatibility_matrix":  {
                str(c): v for c, v in param_recs["compatibility_matrix"].items()
            },
        },
        "notes": {
            "min_suitability":   "Stage 0 도출 불가 — Stage A 후 suitability 분포에서 재산정",
            "min_quality_score": "Stage 0 도출 불가 — Stage C 후 quality 분포에서 재산정",
        },
    }
    with open(output_dir / "recommended_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False,
                       default_flow_style=False)

    # ── [7/8] Figures ─────────────────────────────────────────────────────────
    print("\n[7/8] Rendering figures...")
    render_figures(
        class_counts   = {c: class_counts.get(c, 0) for c in CLASS_IDS},
        subtype_counts = subtype_counts,
        morph_arrays   = morph_arrays,
        bg_type_counts = bg_type_counts,
        bg_arrays      = bg_arrays,
        matrix_out     = matrix_out,
        area_by_class  = area_by_class,
        threshold_recs = threshold_recs,
        figures_dir    = figures_dir,
    )

    # ── [8/8] Report ──────────────────────────────────────────────────────────
    print("\n[8/8] Writing report...")
    render_report(
        n_images       = len(all_ids),
        n_defects      = n_morph,
        n_bg_cells     = n_bg,
        class_counts   = {c: class_counts.get(c, 0) for c in CLASS_IDS},
        subtype_counts = subtype_counts,
        bg_type_counts = bg_type_counts,
        threshold_recs = threshold_recs,
        matrix_out     = matrix_out,
        param_recs     = param_recs,
        output_dir     = output_dir,
    )

    print("\n" + "=" * 70)
    print(f"Stage 0 complete.")
    print(f"Artifacts : {output_dir}")
    print(f"Next step : review recommended_config.yaml → apply to Stage A~C")
    print("=" * 70)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)  # Colab/Linux 명시
    main()
