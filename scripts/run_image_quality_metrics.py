#!/usr/bin/env python3
"""
Image Quality Metrics: KID + LPIPS

Computes three complementary synthesis quality metrics:
  FID  — distributional distance (InceptionV3, existing baseline)
  KID  — distributional distance, unbiased estimator (preferred for n~2,200)
  LPIPS (realism)   — mean perceptual distance between generated and real patches
  LPIPS (diversity) — mean pairwise distance within generated patches

All metrics are computed per defect class on held-out ROI patches.

Dependencies:
  pip install torch-fidelity   # KID
  pip install lpips            # LPIPS

Input:
  train_images/                   — real training images (reference)
  augmented_images/generated/     — CASDA generated ROI patches
  roi_patches_v5.1/roi_metadata.csv
  augmented_dataset/casda_composed/metadata.json

Output:
  <output-dir>/kid_results.json
  <output-dir>/lpips_results.json
  <output-dir>/quality_metrics_table.md
  <output-dir>/quality_metrics_table.tex

Usage:
  python scripts/run_image_quality_metrics.py \\
    --config         /content/CASDA/configs/benchmark_experiment.yaml \\
    --data-dir       /content/drive/MyDrive/data/Severstal/train_images \\
    --csv            /content/drive/MyDrive/data/Severstal/train.csv \\
    --casda-roi-dir  /content/drive/MyDrive/data/Severstal/augmented_images/generated \\
    --roi-meta       /content/drive/MyDrive/data/Severstal/roi_patches_v5.1/roi_metadata.csv \\
    --meta-composed  /content/drive/MyDrive/data/Severstal/augmented_dataset/casda_composed/metadata.json \\
    --metrics        kid lpips \\
    --output-dir     /content/drive/MyDrive/data/Severstal/fid_results
"""

import argparse
import ast
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torch_fidelity
    HAS_TORCH_FIDELITY = True
except ImportError:
    HAS_TORCH_FIDELITY = False

try:
    import lpips as lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


# ============================================================================
# ROI patch extraction utilities
# ============================================================================

def safe_bbox(val) -> Optional[List[int]]:
    try:
        return [int(v) for v in ast.literal_eval(str(val))]
    except Exception:
        return None


def load_roi_meta(roi_meta_path: Path) -> pd.DataFrame:
    df = pd.read_csv(roi_meta_path, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["roi_bbox"] = df["roi_bbox"].apply(safe_bbox)
    df = df.dropna(subset=["roi_bbox"])
    return df


def extract_roi_patch(img_path: Path, bbox: List[int],
                      target_size: int = 256) -> Optional[np.ndarray]:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    patch = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)  # KID/LPIPS expect 3-channel


def collect_real_patches(
    df_roi: pd.DataFrame,
    data_dir: Path,
    class_id: int,
    target_size: int = 256,
    max_samples: int = 500,
) -> List[np.ndarray]:
    rows = df_roi[df_roi["class_id"] == class_id].copy()
    rows = rows.sample(min(len(rows), max_samples), random_state=42)
    patches = []
    for _, row in rows.iterrows():
        p = extract_roi_patch(data_dir / row["image_id"], row["roi_bbox"], target_size)
        if p is not None:
            patches.append(p)
    return patches


def collect_generated_patches(
    metadata: List[Dict],
    gen_dir: Path,
    class_id: int,
    target_size: int = 256,
    max_samples: int = 500,
) -> List[np.ndarray]:
    entries = [e for e in metadata if int(e.get("class_id", 0)) == class_id]
    entries.sort(key=lambda x: x.get("suitability_score", 0), reverse=True)
    entries = entries[:max_samples]
    patches = []
    for e in entries:
        src = gen_dir / e["source_generated"]
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        patches.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return patches


# ============================================================================
# KID computation via torch-fidelity
# ============================================================================

def _save_patches_to_temp(patches: List[np.ndarray], tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(patches):
        cv2.imwrite(str(tmp_dir / f"{i:05d}.png"),
                    cv2.cvtColor(p, cv2.COLOR_RGB2BGR))


def compute_kid_class(
    real_patches: List[np.ndarray],
    gen_patches:  List[np.ndarray],
    class_id:     int,
    tmp_root:     Path,
) -> Dict:
    if not HAS_TORCH_FIDELITY:
        return {"error": "torch_fidelity not installed. Run: pip install torch-fidelity"}
    if len(real_patches) < 10 or len(gen_patches) < 10:
        return {"error": f"Insufficient patches: real={len(real_patches)}, gen={len(gen_patches)}"}

    real_dir = tmp_root / f"class{class_id}_real"
    gen_dir  = tmp_root / f"class{class_id}_gen"
    _save_patches_to_temp(real_patches, real_dir)
    _save_patches_to_temp(gen_patches,  gen_dir)

    try:
        out = torch_fidelity.calculate_metrics(
            input1=str(real_dir),
            input2=str(gen_dir),
            kid=True,
            fid=False,
            isc=False,
            prc=False,
            verbose=False,
        )
        return {
            "kid_mean": float(out.get("kernel_inception_distance_mean", 0.0)),
            "kid_std":  float(out.get("kernel_inception_distance_std",  0.0)),
            "n_real":   len(real_patches),
            "n_gen":    len(gen_patches),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# LPIPS computation
# ============================================================================

def _patches_to_tensor(patches: List[np.ndarray]) -> "torch.Tensor":
    """Convert list of HxWx3 uint8 RGB numpy arrays to NCHW float [-1,1]."""
    arr = np.stack(patches).astype(np.float32) / 127.5 - 1.0  # [N, H, W, C]
    arr = arr.transpose(0, 3, 1, 2)                            # [N, C, H, W]
    return torch.from_numpy(arr)


def compute_lpips_class(
    real_patches: List[np.ndarray],
    gen_patches:  List[np.ndarray],
    class_id:     int,
    loss_fn,
    batch_size:   int = 32,
) -> Dict:
    if not HAS_LPIPS or not HAS_TORCH:
        return {"error": "lpips or torch not installed. Run: pip install lpips"}
    if len(real_patches) < 2 or len(gen_patches) < 2:
        return {"error": f"Insufficient patches: real={len(real_patches)}, gen={len(gen_patches)}"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Realism: mean LPIPS(generated[i], real[i]) ─────────────────────────
    n_pairs = min(len(gen_patches), len(real_patches))
    gen_sub  = gen_patches[:n_pairs]
    real_sub = real_patches[:n_pairs]

    realism_scores = []
    for start in range(0, n_pairs, batch_size):
        g = _patches_to_tensor(gen_sub[start:start + batch_size]).to(device)
        r = _patches_to_tensor(real_sub[start:start + batch_size]).to(device)
        with torch.no_grad():
            scores = loss_fn(g, r).flatten().cpu().numpy()
        realism_scores.extend(scores.tolist())

    # ── Diversity: mean pairwise LPIPS within generated patches ────────────
    n_div = min(len(gen_patches), 200)
    gen_div = gen_patches[:n_div]
    diversity_scores = []
    indices = [(i, j) for i in range(n_div) for j in range(i + 1, n_div)]
    random.seed(42)
    if len(indices) > 2000:
        indices = random.sample(indices, 2000)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        ga = _patches_to_tensor([gen_div[i] for i, _ in batch_idx]).to(device)
        gb = _patches_to_tensor([gen_div[j] for _, j in batch_idx]).to(device)
        with torch.no_grad():
            scores = loss_fn(ga, gb).flatten().cpu().numpy()
        diversity_scores.extend(scores.tolist())

    return {
        "lpips_realism_mean":   float(np.mean(realism_scores)),
        "lpips_realism_std":    float(np.std(realism_scores)),
        "lpips_diversity_mean": float(np.mean(diversity_scores)),
        "lpips_diversity_std":  float(np.std(diversity_scores)),
        "n_realism_pairs":      n_pairs,
        "n_diversity_pairs":    len(diversity_scores),
    }


# ============================================================================
# Table generation
# ============================================================================

def generate_quality_table_md(
    kid_results:   Dict[int, Dict],
    lpips_results: Dict[int, Dict],
) -> str:
    lines = [
        "| Class | KID↓ (×10³) | LPIPS↓ (realism) | LPIPS↑ (diversity) |",
        "|-------|-------------|------------------|--------------------|",
    ]
    for cls_id in [1, 2, 3, 4]:
        kid  = kid_results.get(cls_id, {})
        lp   = lpips_results.get(cls_id, {})

        kid_s = (f"{kid['kid_mean']*1000:.3f} ± {kid['kid_std']*1000:.3f}"
                 if "kid_mean" in kid else f"Error: {kid.get('error', '?')}")
        rlp_s = (f"{lp['lpips_realism_mean']:.4f} ± {lp['lpips_realism_std']:.4f}"
                 if "lpips_realism_mean" in lp else f"Error: {lp.get('error', '?')}")
        dlp_s = (f"{lp['lpips_diversity_mean']:.4f} ± {lp['lpips_diversity_std']:.4f}"
                 if "lpips_diversity_mean" in lp else f"—")

        lines.append(f"| Class {cls_id} | {kid_s} | {rlp_s} | {dlp_s} |")

    lines.append("\n*KID scaled ×10³ for readability. LPIPS uses AlexNet backbone.*")
    return "\n".join(lines)


def generate_quality_table_tex(
    kid_results:   Dict[int, Dict],
    lpips_results: Dict[int, Dict],
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Synthesis Quality Metrics per Defect Class (CASDA vs. real patches)}",
        r"\label{tab:quality_metrics}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Class & KID$\downarrow$ ($\times 10^3$) & LPIPS$\downarrow$ (realism)"
        r" & LPIPS$\uparrow$ (diversity) \\",
        r"\midrule",
    ]
    for cls_id in [1, 2, 3, 4]:
        kid = kid_results.get(cls_id, {})
        lp  = lpips_results.get(cls_id, {})

        kid_s = (f"${kid['kid_mean']*1000:.3f} \\pm {kid['kid_std']*1000:.3f}$"
                 if "kid_mean" in kid else "—")
        rlp_s = (f"${lp['lpips_realism_mean']:.4f} \\pm {lp['lpips_realism_std']:.4f}$"
                 if "lpips_realism_mean" in lp else "—")
        dlp_s = (f"${lp['lpips_diversity_mean']:.4f} \\pm {lp['lpips_diversity_std']:.4f}$"
                 if "lpips_diversity_mean" in lp else "—")

        lines.append(f"Class {cls_id} & {kid_s} & {rlp_s} & {dlp_s} \\\\")

    lines += [
        r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize KID $= $ Kernel Inception Distance"
        r" (unbiased, polynomial kernel). LPIPS backbone: AlexNet.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute KID + LPIPS synthesis quality metrics")
    parser.add_argument("--config",        default=None, help="benchmark_experiment.yaml (optional)")
    parser.add_argument("--data-dir",      required=True, help="Real training images dir")
    parser.add_argument("--csv",           default=None,  help="train.csv (for class filtering)")
    parser.add_argument("--casda-roi-dir", required=True, help="CASDA generated ROI dir (generated/)")
    parser.add_argument("--roi-meta",      required=True, help="roi_metadata.csv")
    parser.add_argument("--meta-composed", required=True, help="casda_composed/metadata.json")
    parser.add_argument("--metrics",       nargs="+",     default=["kid", "lpips"],
                        choices=["kid", "lpips"],
                        help="Which metrics to compute (default: kid lpips)")
    parser.add_argument("--output-dir",    required=True, help="Output directory")
    parser.add_argument("--target-size",   type=int, default=256,
                        help="ROI patch resize target (default: 256)")
    parser.add_argument("--max-samples",   type=int, default=500,
                        help="Max patches per class per group (default: 500)")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch is required.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metadata ...")
    df_roi = load_roi_meta(Path(args.roi_meta))
    with open(args.meta_composed) as f:
        metadata = json.load(f)

    data_dir    = Path(args.data_dir)
    gen_dir     = Path(args.casda_roi_dir)

    kid_results:   Dict[int, Dict] = {}
    lpips_results: Dict[int, Dict] = {}

    # Load LPIPS model once
    lpips_model = None
    if "lpips" in args.metrics:
        if not HAS_LPIPS:
            print("WARNING: lpips not installed. Skipping LPIPS. Run: pip install lpips",
                  file=sys.stderr)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading LPIPS model (AlexNet) on {device} ...")
            lpips_model = lpips_lib.LPIPS(net="alex").to(device)
            lpips_model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        for cls_id in [1, 2, 3, 4]:
            print(f"\n--- Class {cls_id} ---")

            real_patches = collect_real_patches(
                df_roi, data_dir, cls_id, args.target_size, args.max_samples)
            gen_patches  = collect_generated_patches(
                metadata, gen_dir, cls_id, args.target_size, args.max_samples)

            print(f"  Real patches: {len(real_patches)}  |  Generated: {len(gen_patches)}")

            if "kid" in args.metrics:
                if not HAS_TORCH_FIDELITY:
                    kid_results[cls_id] = {
                        "error": "torch_fidelity not installed. Run: pip install torch-fidelity"}
                else:
                    print(f"  Computing KID ...")
                    kid_results[cls_id] = compute_kid_class(
                        real_patches, gen_patches, cls_id, tmp_root)
                    r = kid_results[cls_id]
                    if "kid_mean" in r:
                        print(f"  KID: {r['kid_mean']*1000:.3f} ± {r['kid_std']*1000:.3f} (×10³)")
                    else:
                        print(f"  KID error: {r.get('error')}")

            if "lpips" in args.metrics and lpips_model is not None:
                print(f"  Computing LPIPS ...")
                lpips_results[cls_id] = compute_lpips_class(
                    real_patches, gen_patches, cls_id, lpips_model)
                r = lpips_results[cls_id]
                if "lpips_realism_mean" in r:
                    print(f"  LPIPS realism:   {r['lpips_realism_mean']:.4f} ± {r['lpips_realism_std']:.4f}")
                    print(f"  LPIPS diversity: {r['lpips_diversity_mean']:.4f} ± {r['lpips_diversity_std']:.4f}")
                else:
                    print(f"  LPIPS error: {r.get('error')}")

    # Save results
    if kid_results:
        kid_path = output_dir / "kid_results.json"
        with open(kid_path, "w") as f:
            json.dump({str(k): v for k, v in kid_results.items()}, f, indent=2)
        print(f"\nSaved → {kid_path}")

    if lpips_results:
        lpips_path = output_dir / "lpips_results.json"
        with open(lpips_path, "w") as f:
            json.dump({str(k): v for k, v in lpips_results.items()}, f, indent=2)
        print(f"Saved → {lpips_path}")

    # Tables
    md_path  = output_dir / "quality_metrics_table.md"
    tex_path = output_dir / "quality_metrics_table.tex"
    md_path.write_text(generate_quality_table_md(kid_results, lpips_results), encoding="utf-8")
    tex_path.write_text(generate_quality_table_tex(kid_results, lpips_results), encoding="utf-8")
    print(f"Saved → {md_path}")
    print(f"Saved → {tex_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
