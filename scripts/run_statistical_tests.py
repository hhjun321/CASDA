#!/usr/bin/env python3
"""
Statistical Significance Tests for CASDA Hypotheses

Reads aggregated multi-seed results and FID results, then runs:
  H3: Architecture Independence   — Friedman test across 3 models
  H4: Class 2 Improvement         — Wilcoxon signed-rank (paired)
  H5: FID Superiority             — Wilcoxon signed-rank on class-level FID
  H6: Augmentation Ratio          — Wilcoxon signed-rank (paired)

Multiple comparisons corrected with Benjamini-Hochberg FDR (alpha=0.05).
Effect sizes reported as Cohen's d.

Input:
  aggregated_results.json     (from aggregate_multiseed_results.py)
  fid_results.json            (from run_fid.py, for H5)

Output:
  hypothesis_test_results.json   — raw test statistics
  significance_table.md          — paper-ready summary table
  significance_table.tex         — LaTeX version

Usage:
  python scripts/run_statistical_tests.py \\
    --aggregated-results /content/drive/.../multiseed_aggregated/aggregated_results.json \\
    --fid-results        /content/drive/.../fid_results/fid_results.json \\
    --output-dir         /content/drive/.../benchmark_results/statistical_tests \\
    --alpha 0.05
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon, friedmanchisquare
    HAS_SCIPY = True
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Helpers
# ============================================================================

def get_values(aggregated: Dict, model: str, group: str, metric: str) -> List[float]:
    """Extract per-seed values for (model, group, metric)."""
    key = f"{model}|{group}"
    if key not in aggregated:
        raise KeyError(f"Key not found in aggregated results: {key}")
    return aggregated[key]["metrics"][metric]["values"]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d for paired samples (a - b)."""
    diff = a - b
    if diff.std(ddof=1) < 1e-12:
        return float("inf") if diff.mean() > 0 else 0.0
    return float(diff.mean() / diff.std(ddof=1))


def effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def bh_correction(pvalues: List[float], alpha: float = 0.05) -> List[float]:
    """Benjamini-Hochberg FDR correction. Returns corrected p-values."""
    n = len(pvalues)
    order   = np.argsort(pvalues)
    ranks   = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    corrected = np.minimum(1.0, np.array(pvalues) * n / ranks)
    # Enforce monotonicity (from largest rank downward)
    for i in range(n - 2, -1, -1):
        corrected[order[i]] = min(corrected[order[i]], corrected[order[i + 1]])
    return corrected.tolist()


# ============================================================================
# Hypothesis tests
# ============================================================================

MODELS = ["yolo_mfd", "eb_yolov8", "deeplabv3plus"]
CASDA_GROUP    = "casda_composed_pruning"
BASELINE_GROUP = "baseline_raw"
COPYPASTE_GROUP = "copypaste"


def test_h3_architecture(aggregated: Dict) -> Dict:
    """
    H3: Architecture Independence
    Friedman test on (mAP_casda - mAP_baseline) deltas across 3 models.
    If p < alpha → architectures differ in CASDA benefit (reject H3).
    If p >= alpha → no significant difference → CASDA benefit is architecture-independent.
    """
    deltas_per_model = []
    available_models = []
    for model in MODELS:
        try:
            casda_vals    = np.array(get_values(aggregated, model, CASDA_GROUP,    "mAP@0.5"))
            baseline_vals = np.array(get_values(aggregated, model, BASELINE_GROUP, "mAP@0.5"))
            deltas_per_model.append(casda_vals - baseline_vals)
            available_models.append(model)
        except KeyError:
            pass

    if len(deltas_per_model) < 3:
        return {"error": f"Need 3 models, found {len(available_models)}: {available_models}"}

    try:
        stat, p = friedmanchisquare(*deltas_per_model)
    except Exception as e:
        return {"error": str(e)}

    return {
        "hypothesis": "H3",
        "test":       "Friedman",
        "statistic":  float(stat),
        "p_raw":      float(p),
        "models":     available_models,
        "deltas_mean": [float(np.mean(d)) for d in deltas_per_model],
    }


def test_h4_class2(aggregated: Dict) -> Dict:
    """
    H4: Class 2 Improvement
    Wilcoxon signed-rank: casda_composed_pruning vs baseline_raw on Class2 AP.
    Averaged across all available models.
    """
    all_casda    = []
    all_baseline = []
    for model in MODELS:
        try:
            all_casda    += get_values(aggregated, model, CASDA_GROUP,    "Class2")
            all_baseline += get_values(aggregated, model, BASELINE_GROUP, "Class2")
        except KeyError:
            pass

    if not all_casda:
        return {"error": "No Class2 data found"}

    a = np.array(all_casda)
    b = np.array(all_baseline)
    diff = a - b
    if np.all(diff == 0):
        return {"error": "All differences are zero — Wilcoxon not applicable"}

    try:
        stat, p = wilcoxon(a, b, alternative="greater")
    except Exception as e:
        return {"error": str(e)}

    d = cohens_d(a, b)
    return {
        "hypothesis": "H4",
        "test":       "Wilcoxon signed-rank",
        "statistic":  float(stat),
        "p_raw":      float(p),
        "cohens_d":   d,
        "effect":     effect_label(d),
        "n":          len(a),
    }


def test_h5_fid(fid_data: Optional[Dict]) -> Dict:
    """
    H5: FID Superiority
    Wilcoxon signed-rank on per-class FID: CASDA vs CopyPaste.
    Uses class-level FID values from run_fid.py output.
    """
    if fid_data is None:
        return {"error": "FID results not provided (use --fid-results)"}

    # Expected structure: {"casda": {"class1": x, ...}, "copypaste": {"class1": x, ...}}
    # or: list of {"group": ..., "class": ..., "fid": ...}
    casda_fids    = []
    copypaste_fids = []

    # Try flat dict first
    if "casda" in fid_data and "copypaste" in fid_data:
        for cls in ["class1", "class2", "class3", "class4"]:
            c = fid_data["casda"].get(cls)
            cp = fid_data["copypaste"].get(cls)
            if c is not None and cp is not None:
                casda_fids.append(float(c))
                copypaste_fids.append(float(cp))
    else:
        # Try list format
        for entry in fid_data if isinstance(fid_data, list) else []:
            grp = entry.get("group", "")
            val = entry.get("fid", entry.get("fid_score"))
            if val is None:
                continue
            if "casda" in grp.lower():
                casda_fids.append(float(val))
            elif "copypaste" in grp.lower() or "copy" in grp.lower():
                copypaste_fids.append(float(val))

    if len(casda_fids) < 2 or len(casda_fids) != len(copypaste_fids):
        return {
            "error": (f"Insufficient paired FID data: "
                      f"casda={len(casda_fids)}, copypaste={len(copypaste_fids)}")
        }

    a = np.array(casda_fids)
    b = np.array(copypaste_fids)
    # Lower FID is better; we test casda < copypaste
    try:
        stat, p = wilcoxon(a, b, alternative="less")
    except Exception as e:
        return {"error": str(e)}

    d = cohens_d(b, a)  # effect of copypaste being higher
    return {
        "hypothesis": "H5",
        "test":       "Wilcoxon signed-rank",
        "statistic":  float(stat),
        "p_raw":      float(p),
        "cohens_d":   d,
        "effect":     effect_label(d),
        "n":          len(a),
        "casda_fid_mean":    float(a.mean()),
        "copypaste_fid_mean": float(b.mean()),
    }


def test_h6_ratio(aggregated: Dict) -> Dict:
    """
    H6: Augmentation Ratio
    Wilcoxon signed-rank: casda_composed_pruning vs copypaste on mAP@0.5.
    """
    all_casda    = []
    all_copypaste = []
    for model in MODELS:
        try:
            all_casda     += get_values(aggregated, model, CASDA_GROUP,     "mAP@0.5")
            all_copypaste += get_values(aggregated, model, COPYPASTE_GROUP, "mAP@0.5")
        except KeyError:
            pass

    if not all_casda:
        return {"error": "No mAP@0.5 data found for H6"}

    a = np.array(all_casda)
    b = np.array(all_copypaste)
    diff = a - b
    if np.all(diff == 0):
        return {"error": "All differences are zero — Wilcoxon not applicable"}

    try:
        stat, p = wilcoxon(a, b, alternative="greater")
    except Exception as e:
        return {"error": str(e)}

    d = cohens_d(a, b)
    return {
        "hypothesis": "H6",
        "test":       "Wilcoxon signed-rank",
        "statistic":  float(stat),
        "p_raw":      float(p),
        "cohens_d":   d,
        "effect":     effect_label(d),
        "n":          len(a),
    }


# ============================================================================
# FDR correction and table generation
# ============================================================================

def apply_bh(results: List[Dict], alpha: float) -> List[Dict]:
    """Apply Benjamini-Hochberg FDR correction across hypotheses."""
    valid   = [r for r in results if "p_raw" in r and "error" not in r]
    invalid = [r for r in results if "p_raw" not in r or "error" in r]

    if not valid:
        return results

    pvals     = [r["p_raw"] for r in valid]
    corrected = bh_correction(pvals, alpha)

    for r, p_corr in zip(valid, corrected):
        r["p_bh"]     = p_corr
        r["supported"] = p_corr < alpha

    for r in invalid:
        r["p_bh"]     = None
        r["supported"] = None

    return valid + invalid


def _sig_stars(p: Optional[float]) -> str:
    if p is None:
        return "—"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def generate_md_table(results: List[Dict]) -> str:
    lines = [
        "| Hypothesis | Test | Statistic | p-value (BH-corr) | Effect (d) | Result |",
        "|-----------|------|-----------|-------------------|------------|--------|",
    ]
    for r in sorted(results, key=lambda x: x.get("hypothesis", "Z")):
        h    = r.get("hypothesis", "—")
        err  = r.get("error")
        if err:
            lines.append(f"| {h} | — | — | — | — | Error: {err} |")
            continue
        test = r.get("test", "—")
        stat = r.get("statistic")
        stat_s = f"{stat:.3f}" if stat is not None else "—"
        p_bh   = r.get("p_bh")
        p_s    = f"p={p_bh:.4f} {_sig_stars(p_bh)}" if p_bh is not None else "—"
        d      = r.get("cohens_d")
        d_s    = f"d={d:.2f} ({r.get('effect','')})" if d is not None else "—"
        sup    = r.get("supported")
        sup_s  = "Supported" if sup else ("Not supported" if sup is not None else "—")
        lines.append(f"| {h} | {test} | {stat_s} | {p_s} | {d_s} | {sup_s} |")

    lines.append("\n*Significance: \\* p<0.05, \\*\\* p<0.01 (Benjamini-Hochberg FDR, α=0.05)*")
    return "\n".join(lines)


def generate_tex_table(results: List[Dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hypothesis Test Results (Benjamini-Hochberg FDR corrected, $\alpha=0.05$)}",
        r"\label{tab:significance}",
        r"\begin{tabular}{llcccl}",
        r"\toprule",
        r"Hypothesis & Test & Statistic & $p$ (BH) & Effect ($d$) & Result \\",
        r"\midrule",
    ]
    for r in sorted(results, key=lambda x: x.get("hypothesis", "Z")):
        h   = r.get("hypothesis", "—")
        err = r.get("error")
        if err:
            lines.append(f"{h} & \\multicolumn{{5}}{{l}}{{Error: {err}}} \\\\")
            continue
        test = r.get("test", "—").replace("Wilcoxon signed-rank", "Wilcoxon")
        stat = r.get("statistic")
        stat_s = f"{stat:.3f}" if stat is not None else "—"
        p_bh   = r.get("p_bh")
        p_s    = f"{p_bh:.4f}" if p_bh is not None else "—"
        d      = r.get("cohens_d")
        d_s    = f"{d:.2f} ({r.get('effect','')})" if d is not None else "—"
        sup    = r.get("supported")
        sup_s  = "Supported" if sup else ("Not supported" if sup is not None else "—")
        lines.append(f"{h} & {test} & {stat_s} & {p_s} & {d_s} & {sup_s} \\\\")

    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\footnotesize $^{*}p<0.05$, $^{**}p<0.01$"
        r" (Benjamini-Hochberg FDR corrected).} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Statistical tests for CASDA hypotheses")
    parser.add_argument("--aggregated-results", required=True,
                        help="Path to aggregated_results.json")
    parser.add_argument("--fid-results",        default=None,
                        help="Path to fid_results.json (required for H5)")
    parser.add_argument("--output-dir",          required=True,
                        help="Output directory")
    parser.add_argument("--alpha",               type=float, default=0.05,
                        help="FDR significance threshold (default: 0.05)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    print(f"Loading aggregated results: {args.aggregated_results}")
    with open(args.aggregated_results) as f:
        aggregated = json.load(f)

    fid_data = None
    if args.fid_results:
        print(f"Loading FID results: {args.fid_results}")
        with open(args.fid_results) as f:
            fid_data = json.load(f)

    # Run tests
    print("\nRunning hypothesis tests ...")
    results = [
        test_h3_architecture(aggregated),
        test_h4_class2(aggregated),
        test_h5_fid(fid_data),
        test_h6_ratio(aggregated),
    ]

    # FDR correction
    results = apply_bh(results, args.alpha)

    # Print summary
    print("\n=== Hypothesis Test Results ===")
    for r in sorted(results, key=lambda x: x.get("hypothesis", "Z")):
        h = r.get("hypothesis", "?")
        if "error" in r:
            print(f"  {h}: ERROR — {r['error']}")
            continue
        p_bh  = r.get("p_bh")
        d     = r.get("cohens_d")
        sup   = r.get("supported")
        print(f"  {h}: p(BH)={p_bh:.4f}  d={d:.3f if d else '—'}  "
              f"{'Supported' if sup else 'Not supported'}")

    # Save raw results
    raw_path = output_dir / "hypothesis_test_results.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {raw_path}")

    # Save tables
    md_path  = output_dir / "significance_table.md"
    tex_path = output_dir / "significance_table.tex"
    md_path.write_text(generate_md_table(results), encoding="utf-8")
    tex_path.write_text(generate_tex_table(results), encoding="utf-8")
    print(f"Saved → {md_path}")
    print(f"Saved → {tex_path}")


if __name__ == "__main__":
    main()
