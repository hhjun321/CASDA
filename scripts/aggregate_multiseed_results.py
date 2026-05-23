#!/usr/bin/env python3
"""
Multi-Seed Benchmark Results Aggregator

Reads per-seed benchmark_results.json files and produces mean ± std tables
in both Markdown and LaTeX formats.

Input structure:
  <results-dir>/seed_42/benchmark_results.json
  <results-dir>/seed_123/benchmark_results.json
  <results-dir>/seed_456/benchmark_results.json

Output:
  <output-dir>/aggregated_results.json    — raw aggregation data
  <output-dir>/table_mean_std.md          — Markdown mean±std table
  <output-dir>/table_mean_std.tex         — LaTeX mean±std table

Usage:
  python scripts/aggregate_multiseed_results.py \\
    --results-dirs \\
      /content/drive/.../benchmark_results/multiseed/seed_42 \\
      /content/drive/.../benchmark_results/multiseed/seed_123 \\
      /content/drive/.../benchmark_results/multiseed/seed_456 \\
    --output-dir /content/drive/.../benchmark_results/multiseed_aggregated
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================================
# Metrics to aggregate
# ============================================================================
SCALAR_METRICS = ["mAP@0.5", "dice_mean"]
CLASS_METRICS  = ["Class1", "Class2", "Class3", "Class4"]


# ============================================================================
# Data loading
# ============================================================================

def load_seed_results(results_dir: Path) -> List[Dict]:
    """Load benchmark_results.json from a single seed directory."""
    path = results_dir / "benchmark_results.json"
    if not path.exists():
        print(f"  [WARN] Not found: {path}", file=sys.stderr)
        return []
    with open(path) as f:
        return json.load(f)


def extract_metrics(entry: Dict) -> Dict[str, float]:
    """Flatten one benchmark result entry into a flat metric dict."""
    m = entry.get("metrics", {})
    cap = m.get("class_ap", {})
    flat: Dict[str, float] = {}
    for key in SCALAR_METRICS:
        flat[key] = float(m.get(key, 0.0))
    for cls in CLASS_METRICS:
        flat[cls] = float(cap.get(cls, 0.0))
    return flat


# ============================================================================
# Aggregation
# ============================================================================

def aggregate(results_dirs: List[Path]) -> Dict:
    """
    Returns nested dict:
      aggregated[(model, group)] = {
        "metric_name": {"values": [v1, v2, v3], "mean": x, "std": y},
        ...
      }
    """
    # Collect per-seed values: {(model, group): {metric: [v_seed1, v_seed2, ...]}}
    raw: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for i, d in enumerate(results_dirs):
        entries = load_seed_results(d)
        if not entries:
            print(f"  [WARN] No entries in {d}", file=sys.stderr)
            continue

        seed_label = d.name  # e.g. "seed_42"
        print(f"  Loaded {len(entries)} entries from {seed_label}")

        for entry in entries:
            model = entry.get("model", "unknown")
            group = entry.get("dataset", "unknown")
            key   = (model, group)

            if key not in raw:
                raw[key] = {m: [] for m in (SCALAR_METRICS + CLASS_METRICS)}

            flat = extract_metrics(entry)
            for metric, val in flat.items():
                raw[key][metric].append(val)

    # Compute mean ± std
    aggregated: Dict = {}
    for (model, group), metrics in raw.items():
        agg: Dict = {}
        for metric, values in metrics.items():
            arr = np.array(values, dtype=float)
            agg[metric] = {
                "values": list(arr),
                "mean":   float(np.mean(arr)),
                "std":    float(np.std(arr, ddof=0)),
                "n":      len(arr),
            }
        aggregated[f"{model}|{group}"] = {"model": model, "group": group, "metrics": agg}

    return aggregated


# ============================================================================
# Table generation
# ============================================================================

def _fmt_md(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _fmt_tex(mean: float, std: float, bold: bool = False) -> str:
    s = f"${mean:.4f} \\pm {std:.4f}$"
    return f"\\textbf{{{s}}}" if bold else s


def _best_group(aggregated: Dict, model: str, metric: str) -> str:
    """Return the group key with highest mean for (model, metric)."""
    best_val  = -1.0
    best_group = ""
    for entry in aggregated.values():
        if entry["model"] != model:
            continue
        v = entry["metrics"].get(metric, {}).get("mean", 0.0)
        if v > best_val:
            best_val   = v
            best_group = entry["group"]
    return best_group


def generate_markdown_table(aggregated: Dict) -> str:
    lines: List[str] = []
    header = ("| Model | Group | mAP@0.5 | Dice | "
              "Class1 AP | Class2 AP | Class3 AP | Class4 AP | N seeds |")
    sep    = ("|-------|-------|---------|------|"
              "-----------|-----------|-----------|-----------|---------|")
    lines += [header, sep]

    for entry in sorted(aggregated.values(), key=lambda e: (e["model"], e["group"])):
        m = entry["metrics"]
        row = (
            f"| {entry['model']:<12} | {entry['group']:<24} | "
            f"{_fmt_md(m['mAP@0.5']['mean'], m['mAP@0.5']['std'])} | "
            f"{_fmt_md(m['dice_mean']['mean'], m['dice_mean']['std'])} | "
            f"{_fmt_md(m['Class1']['mean'], m['Class1']['std'])} | "
            f"{_fmt_md(m['Class2']['mean'], m['Class2']['std'])} | "
            f"{_fmt_md(m['Class3']['mean'], m['Class3']['std'])} | "
            f"{_fmt_md(m['Class4']['mean'], m['Class4']['std'])} | "
            f"{m['mAP@0.5']['n']} |"
        )
        lines.append(row)

    n_seeds = next(iter(aggregated.values()))["metrics"]["mAP@0.5"]["n"]
    lines.append(f"\n*Results reported as mean ± std over {n_seeds} random seeds.*")
    return "\n".join(lines)


def generate_latex_table(aggregated: Dict) -> str:
    lines: List[str] = []
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CASDA Benchmark Results (mean $\pm$ std over 3 seeds)}",
        r"\label{tab:benchmark_multiseed}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Model & Group & mAP@0.5 & Dice & C1 AP & C2 AP & C3 AP & C4 AP \\",
        r"\midrule",
    ]

    prev_model = ""
    for entry in sorted(aggregated.values(), key=lambda e: (e["model"], e["group"])):
        model = entry["model"]
        group = entry["group"]
        m     = entry["metrics"]

        if model != prev_model and prev_model:
            lines.append(r"\midrule")
        prev_model = model

        model_str = model if model != prev_model else ""

        best_map  = _best_group(aggregated, model, "mAP@0.5") == group
        best_dice = _best_group(aggregated, model, "dice_mean") == group

        row = (
            f"{model} & {group} & "
            f"{_fmt_tex(m['mAP@0.5']['mean'], m['mAP@0.5']['std'], best_map)} & "
            f"{_fmt_tex(m['dice_mean']['mean'], m['dice_mean']['std'], best_dice)} & "
            f"{_fmt_tex(m['Class1']['mean'], m['Class1']['std'])} & "
            f"{_fmt_tex(m['Class2']['mean'], m['Class2']['std'])} & "
            f"{_fmt_tex(m['Class3']['mean'], m['Class3']['std'])} & "
            f"{_fmt_tex(m['Class4']['mean'], m['Class4']['std'])} \\\\"
        )
        lines.append(row)

    n_seeds = next(iter(aggregated.values()))["metrics"]["mAP@0.5"]["n"]
    lines += [
        r"\bottomrule",
        r"\multicolumn{8}{l}{\footnotesize Mean $\pm$ std over "
        + str(n_seeds)
        + r" random seeds. Bold: best mAP / Dice per model.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed benchmark results")
    parser.add_argument("--results-dirs", nargs="+", required=True,
                        help="Per-seed result directories (each must contain benchmark_results.json)")
    parser.add_argument("--output-dir",   required=True,
                        help="Output directory for aggregated tables")
    args = parser.parse_args()

    results_dirs = [Path(d) for d in args.results_dirs]
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating {len(results_dirs)} seed directories ...")
    aggregated = aggregate(results_dirs)

    if not aggregated:
        print("ERROR: No results aggregated. Check input directories.", file=sys.stderr)
        sys.exit(1)

    # Save raw aggregation
    raw_path = output_dir / "aggregated_results.json"
    with open(raw_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved → {raw_path}")

    # Markdown table
    md_path = output_dir / "table_mean_std.md"
    md_path.write_text(generate_markdown_table(aggregated), encoding="utf-8")
    print(f"Saved → {md_path}")

    # LaTeX table
    tex_path = output_dir / "table_mean_std.tex"
    tex_path.write_text(generate_latex_table(aggregated), encoding="utf-8")
    print(f"Saved → {tex_path}")

    # Print summary
    print("\n=== Summary (mAP@0.5 mean ± std) ===")
    for entry in sorted(aggregated.values(), key=lambda e: (e["model"], e["group"])):
        m = entry["metrics"]["mAP@0.5"]
        print(f"  {entry['model']:<14} {entry['group']:<28} "
              f"{m['mean']:.4f} ± {m['std']:.4f}  (n={m['n']})")


if __name__ == "__main__":
    main()
