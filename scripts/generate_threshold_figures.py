"""
논문 Comment 3 대응 — 임계값 분포 근거 Figure 생성
========================================================
fig3_morph_thresh_hist.png  : 형태학적 특징 분포 + 임계값 (분류 + hint 생성)
fig4_bg_variance_thresh_hist.png : 배경 분산 분포 + variance_threshold
fig5_bg_edge_thresh_hist.png     : 배경 엣지 분포 + edge_threshold

Usage (Colab):
    !python $SCRIPTS/generate_threshold_figures.py \
        --morph_csv   $ANALYSIS_DIR/morphological_features.csv \
        --bg_csv      $ANALYSIS_DIR/background_features.csv \
        --output_dir  $ANALYSIS_DIR/figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker


# ── 논문 분류 임계값 (defect_type 수식 기준) ────────────────────────────────
PAPER_THRESHOLDS = {
    'linearity':         {'value': 0.85,  'label': 'λ=0.85 (HIGH_LINEARITY)',     'color': '#c0392b'},
    'aspect_ratio_high': {'value': 5.0,   'label': 'α=5.0 (HIGH_ASPECT_RATIO)',   'color': '#c0392b'},
    'aspect_ratio_low':  {'value': 2.0,   'label': 'α=2.0 (LOW_ASPECT_RATIO)',    'color': '#2980b9'},
    'solidity_high':     {'value': 0.9,   'label': 'σ=0.9 (HIGH_SOLIDITY)',       'color': '#c0392b'},
    'solidity_irr':      {'value': 0.70,  'label': 'σ=0.7 (IRREGULAR_SOLIDITY)', 'color': '#d35400'},
    'variance':          {'value': 14.09, 'label': 'τ=14.09 (variance_threshold)', 'color': '#c0392b'},
    'edge':              {'value': 0.607, 'label': 'ε=0.607 (edge_threshold)',     'color': '#c0392b'},
}

# ── hint 생성 임계값 (3-channel hint image 생성 기준) ────────────────────────
# R채널: λ>0.7 → skeleton+edge, σ>0.8 → filled mask, else → edge-based
# B채널: smooth 배경에서 α=0.5 감쇠 (rendering 파라미터, 분포 근거 불필요)
HINT_THRESHOLDS = {
    'hint_linearity': {'value': 0.70, 'label': 'λ=0.70 (R-ch: skeleton+edge)',  'color': '#8e44ad'},
    'hint_solidity':  {'value': 0.80, 'label': 'σ=0.8 (R-ch: filled mask)',     'color': '#6c3483'},
}

# ── 데이터 기반 도출값 (비교용 보조선) ────────────────────────────────────────
DATA_THRESHOLDS = {
    'linearity':         0.7012,
    'aspect_ratio_high': 4.979,
    'aspect_ratio_low':  1.831,
    'solidity_high':     0.9089,
    'solidity_irr':      0.75,
}

# ── hint 임계값에 대응하는 데이터 기반 도출값 ─────────────────────────────────
HINT_DATA_MAP = {
    'hint_linearity': 0.7012,   # λ Otsu boundary; identical comparison target as DATA_THRESHOLDS['linearity']
}

# 겹침 판단 기준: axis range의 2% 이내이면 시각적으로 구분 불가 → callout으로 대체
OVERLAP_FRAC = 0.02

SUBTYPE_COLORS = {
    'linear_scratch': '#2ecc71',
    'compact_blob':   '#3498db',
    'irregular':      '#e74c3c',
    'general':        '#95a5a6',
}


def _add_vline(ax, x, label, color, linestyle='-', alpha=0.85, ymax=0.92):
    ax.axvline(x=x, color=color, linestyle=linestyle,
               linewidth=1.8, alpha=alpha, zorder=5)
    ylim = ax.get_ylim()
    y_pos = ylim[0] + (ylim[1] - ylim[0]) * ymax
    ax.text(x, y_pos, f' {label}', color=color, fontsize=7.5,
            rotation=90, va='top', ha='right', fontweight='bold')


def _add_alignment_callout(ax, x, data_v, color, xlim, ymax_frac=0.68):
    """Near-overlapping 쌍에 대해 두 값과 Δ를 callout box로 명시한다.

    두 선을 겹쳐 그리는 대신 paper line 하나를 그리고,
    데이터 기반 값과 차이를 옆에 주석으로 표시해 alignment를 명확히 전달한다.
    """
    x_range = xlim[1] - xlim[0]
    ylim = ax.get_ylim()
    y_ref = ylim[0] + (ylim[1] - ylim[0]) * ymax_frac
    delta = abs(data_v - x)

    # 선이 축 우측 절반에 있으면 박스를 왼쪽에, 아니면 오른쪽에 배치
    mid = (xlim[0] + xlim[1]) / 2
    if x > mid:
        x_text = x - x_range * 0.04
        ha = 'right'
    else:
        x_text = x + x_range * 0.04
        ha = 'left'

    ax.annotate(
        f'≈ data: {data_v:.4f}\n(Δ={delta:.4f})',
        xy=(x, y_ref),
        xytext=(x_text, y_ref),
        fontsize=6.5, color='#5d6d7e',
        ha=ha, va='center',
        arrowprops=dict(arrowstyle='-', color='#95a5a6', lw=0.8),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=color, alpha=0.88, linewidth=0.8),
        zorder=8
    )


def fig3_morph_thresh(morph_csv: Path, output_dir: Path):
    df = pd.read_csv(morph_csv)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        'Morphological Feature Distributions with Classification and Hint Generation Thresholds\n'
        '(Severstal training set, N=19,958 defect instances)',
        fontsize=11, fontweight='bold', y=1.02
    )

    # (col, xlabel, xlim, bins, paper_keys, hint_keys)
    specs = [
        ('linearity',
         'Linearity (λ)',
         (0, 1), 50,
         [('linearity', '-')],
         [('hint_linearity', '-')]),

        ('aspect_ratio',
         'Aspect Ratio (α)',
         (0, 20), 60,
         [('aspect_ratio_high', '-'), ('aspect_ratio_low', '--')],
         []),

        ('solidity',
         'Solidity (σ)',
         (0.1, 1), 50,
         [('solidity_high', '-'), ('solidity_irr', '--')],
         [('hint_solidity', '-')]),
    ]

    for ax, (col, xlabel, xlim, bins, paper_keys, hint_keys) in zip(axes, specs):
        # 서브타입별 스택 히스토그램
        subtypes = ['linear_scratch', 'compact_blob', 'irregular', 'general']
        data_by_sub = [df[df['defect_subtype'] == s][col].dropna().values
                       for s in subtypes]
        bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)

        bottoms = np.zeros(bins)
        for stype, data in zip(subtypes, data_by_sub):
            counts, _ = np.histogram(data, bins=bin_edges)
            ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
                   bottom=bottoms, color=SUBTYPE_COLORS[stype],
                   alpha=0.75, align='edge', label=stype.replace('_', ' '))
            bottoms += counts

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(xlim)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.3)

        # 분류 임계값 (빨강/파랑) + 데이터 기반 도출값 비교
        for key, ls in paper_keys:
            info = PAPER_THRESHOLDS[key]
            _add_vline(ax, info['value'], info['label'], info['color'], ls)

            if key in DATA_THRESHOLDS:
                dv = DATA_THRESHOLDS[key]
                pv = info['value']
                x_range = xlim[1] - xlim[0]

                if abs(dv - pv) / x_range < OVERLAP_FRAC:
                    # 시각적으로 겹치는 쌍 → callout annotation
                    _add_alignment_callout(ax, pv, dv, info['color'], xlim)
                else:
                    # 잘 분리된 쌍 → 회색 점선
                    ax.axvline(x=dv, color='#7f8c8d', linestyle=':', linewidth=1.2,
                               alpha=0.7, zorder=4)
                    ax.text(dv, ax.get_ylim()[1] * 0.75,
                            f' {dv:.3f}\n(data)', color='#7f8c8d',
                            fontsize=6.0, rotation=90, va='top', ha='left')

        # hint 생성 임계값 (보라색) + 데이터 기반 도출값 비교
        for key, ls in hint_keys:
            info = HINT_THRESHOLDS[key]
            _add_vline(ax, info['value'], info['label'], info['color'], ls,
                       alpha=0.80, ymax=0.55)

            if key in HINT_DATA_MAP:
                dv = HINT_DATA_MAP[key]
                pv = info['value']
                x_range = xlim[1] - xlim[0]

                if abs(dv - pv) / x_range < OVERLAP_FRAC:
                    _add_alignment_callout(ax, pv, dv, info['color'], xlim,
                                           ymax_frac=0.35)

    # 공통 범례
    handles = [mpatches.Patch(color=SUBTYPE_COLORS[s], alpha=0.75,
                               label=s.replace('_', ' '))
               for s in ['linear_scratch', 'compact_blob', 'irregular', 'general']]
    red_line    = plt.Line2D([0], [0], color='#c0392b', lw=1.8,
                              label='Classification threshold')
    blue_line   = plt.Line2D([0], [0], color='#2980b9', lw=1.8, linestyle='--',
                              label='Classification threshold (lower)')
    purple_line = plt.Line2D([0], [0], color='#8e44ad', lw=1.8,
                              label='Hint generation threshold (R-ch)')
    grey_line   = plt.Line2D([0], [0], color='#7f8c8d', lw=1.2,
                              linestyle=':', label='Data-driven value (visually separated)')
    fig.legend(handles=handles + [red_line, blue_line, purple_line, grey_line],
               loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.10), frameon=True)

    plt.tight_layout()
    out_path = output_dir / 'fig3_morph_thresh_hist.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def fig4_variance_thresh(bg_csv: Path, output_dir: Path):
    df = pd.read_csv(bg_csv)
    if 'variance' not in df.columns:
        print(f'[WARN] variance column not found in {bg_csv}. Skipping fig4.')
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    data = df['variance'].dropna()
    data_clip = data[data < np.percentile(data, 98)]
    ax.hist(data_clip, bins=80, color='#5dade2', alpha=0.75, edgecolor='white', lw=0.3)
    ax.set_xlabel('Background Variance', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Background Variance Distribution\n'
                 'with variance_threshold (τ)', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{int(x):,}' if x >= 10000 else f'{int(x)}'))

    info = PAPER_THRESHOLDS['variance']
    _add_vline(ax, info['value'], info['label'], info['color'])

    plt.tight_layout()
    out_path = output_dir / 'fig4_bg_variance_thresh_hist.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def fig5_edge_thresh(bg_csv: Path, output_dir: Path):
    df = pd.read_csv(bg_csv)

    edge_col = None
    for candidate in ['edge_density', 'v_ratio', 'edge_total', 'total_strength']:
        if candidate in df.columns:
            edge_col = candidate
            break

    if edge_col is None:
        print(f'[WARN] No edge column found in {bg_csv}. Columns: {list(df.columns)[:10]}')
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    data = df[edge_col].dropna()
    data_clip = data[data < np.percentile(data, 98)]
    ax.hist(data_clip, bins=80, color='#58d68d', alpha=0.75, edgecolor='white', lw=0.3)
    ax.set_xlabel(f'{edge_col}', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Background Edge Feature Distribution\n'
                 'with edge_threshold (ε)', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{int(x):,}' if x >= 10000 else f'{int(x)}'))

    info = PAPER_THRESHOLDS['edge']
    _add_vline(ax, info['value'], info['label'], info['color'])

    plt.tight_layout()
    out_path = output_dir / 'fig5_bg_edge_thresh_hist.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--morph_csv',   required=True)
    parser.add_argument('--bg_csv',      default=None)
    parser.add_argument('--output_dir',  required=True)
    args = parser.parse_args()

    morph_csv  = Path(args.morph_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not morph_csv.exists():
        print(f'Error: {morph_csv} not found'); sys.exit(1)

    print('=== Generating threshold distribution figures ===')
    fig3_morph_thresh(morph_csv, output_dir)

    if args.bg_csv:
        bg_csv = Path(args.bg_csv)
        if bg_csv.exists():
            fig4_variance_thresh(bg_csv, output_dir)
            fig5_edge_thresh(bg_csv, output_dir)
        else:
            print(f'[WARN] bg_csv not found: {bg_csv}')

    print('\nDone. Files saved to:', output_dir)


if __name__ == '__main__':
    # Colab/IPython 셀에서 직접 실행하면 sys.argv에 커널 런처 인수가 들어와
    # argparse의 required 인수 파싱이 실패한다.
    # 셀 실행 시에는 !python 명령어를 사용해야 한다.
    _in_ipython = 'ipykernel' in sys.modules or 'google.colab' in sys.modules
    if _in_ipython:
        print(
            '[ERROR] IPython/Colab 셀에서 직접 실행되었습니다.\n'
            '아래 !python 명령어를 새 셀에서 실행하세요:\n\n'
            '  !python $SCRIPTS/generate_threshold_figures.py \\\n'
            '      --morph_csv  $ANALYSIS_DIR/morphological_features.csv \\\n'
            '      --bg_csv     $ANALYSIS_DIR/background_features.csv \\\n'
            '      --output_dir $ANALYSIS_DIR/figures'
        )
    else:
        main()
