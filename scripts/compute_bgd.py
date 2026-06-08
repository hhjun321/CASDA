"""
compute_bgd.py — Boundary Gradient Discontinuity (BGD) 측정

Reviewer 2 Comment 9 대응:
CASDA(Seamless/Poisson Blending) vs Copy-Paste(Direct Paste)의
결함 경계면 gradient 불연속성을 비교하여 합성 자연스러움을 정량화.

BGD = mean(|∇I|) at boundary band pixels
boundary band = dilate(mask, k) - erode(mask, k)

낮은 BGD = 더 자연스러운 경계 (가설: CASDA < Copy-Paste)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# Worker process globals (initializer로 1회 설정)
_W: dict = {}


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CASDA vs Copy-Paste Boundary Gradient Discontinuity 측정"
    )
    parser.add_argument(
        "--casda-dir", required=True, type=str,
        help="CASDA 합성 이미지 디렉토리 (images/ masks/ metadata.json 포함)",
    )
    parser.add_argument(
        "--copypaste-dir", required=True, type=str,
        help="Copy-Paste 합성 이미지 디렉토리 (images/ masks/ metadata.json 포함)",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str,
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--hint-dir", default=None, type=str,
        help="힌트 이미지 디렉토리 — mask_path 없을 때 R채널 폴백용 (선택)",
    )
    parser.add_argument(
        "--kernel-size", default=5, type=int,
        help="Boundary band 폭 파라미터 — band 폭 ≈ ±kernel_size px (기본: 5)",
    )
    parser.add_argument(
        "--sample-size", default=None, type=int,
        help="그룹당 최대 샘플 수 — 미지정 시 min(두 그룹 유효 N)으로 자동 균형",
    )
    parser.add_argument(
        "--seed", default=42, type=int,
        help="샘플링 재현성 seed (기본: 42)",
    )
    parser.add_argument(
        "--workers", default=4, type=int,
        help="병렬 처리 worker 수 (기본: 4)",
    )
    return parser.parse_args()


# ── Worker initializer ────────────────────────────────────────────────────────

def _init_worker(kernel_size: int, hint_dir_str: str) -> None:
    """각 worker 프로세스에서 1회 실행. 커널과 hint 경로를 _W에 저장."""
    k = kernel_size * 2 + 1
    _W["kernel"] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    _W["hint_dir"] = Path(hint_dir_str) if hint_dir_str else None


# ── Per-image computation ─────────────────────────────────────────────────────

def _load_mask(image_abs: str, mask_abs: str) -> Optional[np.ndarray]:
    """마스크 로드: mask_abs 1차 → hint R채널 폴백 → None."""
    # 1차: metadata.json의 mask_path (최종 좌표계, 가장 정확)
    mask_path = Path(mask_abs)
    if mask_path.exists():
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return m

    # 2차: hint 이미지 R채널 폴백 (512×512 ROI 좌표계 — 최후 수단)
    hint_dir = _W.get("hint_dir")
    if hint_dir is not None:
        img_stem = Path(image_abs).stem
        # _gen{n}_comp{n} 등 합성 접미사 제거 → 원본 sample 이름 추출
        sample_stem = re.sub(r"(_gen\d+)?(_comp\d+)?(_mask)?$", "", img_stem)
        seen: set = set()
        for candidate in [img_stem, sample_stem]:
            if candidate in seen:
                continue
            seen.add(candidate)
            hint_path = hint_dir / f"{candidate}.png"
            if hint_path.exists():
                hint = cv2.imread(str(hint_path))
                if hint is not None:
                    r_channel = hint[:, :, 2]  # BGR → R (index 2)
                    _, m = cv2.threshold(r_channel, 127, 255, cv2.THRESH_BINARY)
                    return m

    return None


def _compute_one(task: dict) -> tuple:
    """단일 이미지 BGD 계산. (name, bgd | None, skip_reason | None) 반환."""
    name = task["name"]

    # worker 초기화 실패 방어
    kernel = _W.get("kernel")
    if kernel is None:
        return name, None, "worker_not_initialized"

    # 이미지 로드
    img = cv2.imread(task["image_abs"])
    if img is None:
        return name, None, "no_image"

    # 마스크 로드
    mask = _load_mask(task["image_abs"], task["mask_abs"])
    if mask is None:
        return name, None, "no_mask"

    # 해상도 불일치 방어 (compose가 항상 1600×256 맞추지만 방어적으로)
    h, w = img.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 이진화
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if np.count_nonzero(mask_bin) == 0:
        return name, None, "empty_mask"

    # Boundary band = dilate - erode (포화 연산으로 음수 방지)
    dilated = cv2.dilate(mask_bin, kernel)
    eroded = cv2.erode(mask_bin, kernel)
    band = cv2.subtract(dilated, eroded)  # 0 or 255

    if np.count_nonzero(band) == 0:
        return name, None, "empty_band"

    # Sobel gradient magnitude (grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    bgd = float(grad_mag[band > 0].mean())
    return name, bgd, None


# ── Group loading ─────────────────────────────────────────────────────────────

def load_group(dataset_dir: Path, label: str) -> list[dict]:
    """metadata.json → task 리스트 (절대경로). 누락 필드 entry는 경고 후 건너뜀."""
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json 없음: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        entries = json.load(f)

    tasks = []
    skipped = 0
    for entry in entries:
        image_rel = entry.get("image_path", "")
        mask_rel = entry.get("mask_path", "")
        if not image_rel or not mask_rel:
            print(
                f"[WARN] [{label}] image_path/mask_path 누락 entry 건너뜀: {entry}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        tasks.append({
            "name": Path(image_rel).stem,
            "label": label,
            "image_abs": str(dataset_dir / image_rel),
            "mask_abs": str(dataset_dir / mask_rel),
        })

    if skipped:
        print(f"  [{label}] 누락 필드로 {skipped}개 entry 건너뜀", file=sys.stderr)

    return tasks


# ── Parallel group computation ────────────────────────────────────────────────

def compute_group(
    tasks: list[dict],
    workers: int,
    kernel_size: int,
    hint_dir: Optional[Path],
    label: str,
) -> tuple[list[float], dict[str, int]]:
    """BGD 병렬 계산. (scores, skip_counts) 반환."""
    hint_str = str(hint_dir) if hint_dir else ""
    scores: list[float] = []
    skip_counts: dict[str, int] = {}

    max_inflight = max(workers * 2, 8)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(kernel_size, hint_str),
    ) as ex:
        pbar = tqdm(total=len(tasks), desc=f"  {label}")
        tasks_iter = iter(tasks)
        pending: dict = {}

        # 초기 배치 제출
        for task in tasks_iter:
            pending[ex.submit(_compute_one, task)] = task["name"]
            if len(pending) >= max_inflight:
                break

        while pending:
            # wait + FIRST_COMPLETED: O(N) per completion (as_completed 재생성 대비)
            done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)

            for fut in done:
                try:
                    _, bgd, skip_reason = fut.result()
                except Exception as exc:
                    skip_counts["worker_exception"] = skip_counts.get("worker_exception", 0) + 1
                    print(f"\n  [WARN] worker exception: {exc}", file=sys.stderr)
                else:
                    if bgd is not None:
                        scores.append(bgd)
                    elif skip_reason:
                        skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1

                del pending[fut]
                pbar.update(1)

                nxt = next(tasks_iter, None)
                if nxt is not None:
                    pending[ex.submit(_compute_one, nxt)] = nxt["name"]

        pbar.close()

    if skip_counts:
        total = sum(skip_counts.values())
        print(f"    skipped {total}/{len(tasks)}: {skip_counts}")

    return scores, skip_counts


# ── Stats & sampling ──────────────────────────────────────────────────────────

def subsample(scores: list[float], n: int, seed: int) -> list[float]:
    """점수 계산 후 N 균형. len(scores) <= n이면 그대로 반환."""
    if len(scores) <= n:
        return scores
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(scores), n, replace=False))
    return [scores[int(i)] for i in idx]  # numpy.int64 → int 명시 변환


def summarize(scores: list[float]) -> dict:
    if not scores:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "n": 0, "min": 0.0, "max": 0.0}
    arr = np.array(scores, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "n": len(arr),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ── Output writers ────────────────────────────────────────────────────────────

def _mkdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(
    output_dir: Path,
    casda_raw: list[float],
    cp_raw: list[float],
    casda_bal: list[float],
    cp_bal: list[float],
    casda_skips: dict,
    cp_skips: dict,
    kernel_size: int,
    seed: int,
) -> None:
    result = {
        "kernel_size": kernel_size,
        "seed": seed,
        "casda": {
            "raw": {**summarize(casda_raw), "scores": casda_raw},
            "balanced": {**summarize(casda_bal), "scores": casda_bal},
            "skip_counts": casda_skips,
        },
        "copypaste": {
            "raw": {**summarize(cp_raw), "scores": cp_raw},
            "balanced": {**summarize(cp_bal), "scores": cp_bal},
            "skip_counts": cp_skips,
        },
    }
    out = output_dir / "bgd_results.json"
    _mkdir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  saved: {out.name}")


def write_summary_txt(
    output_dir: Path,
    casda_bal: list[float],
    cp_bal: list[float],
    kernel_size: int,
) -> None:
    cs = summarize(casda_bal)
    cp = summarize(cp_bal)
    delta = cp["mean"] - cs["mean"]
    ratio = cs["mean"] / cp["mean"] if cp["mean"] > 0 else float("nan")

    lines = [
        "BGD (Boundary Gradient Discontinuity) Summary",
        "=" * 56,
        f"kernel_size = {kernel_size}  |  balanced N = {cs['n']} per group",
        "",
        f"{'Method':<22} {'Mean':>8} {'±Std':>8} {'Median':>8} {'N':>6}",
        "-" * 56,
        f"{'CASDA (Seamless Blend)':<22} {cs['mean']:>8.3f} {cs['std']:>8.3f} {cs['median']:>8.3f} {cs['n']:>6}",
        f"{'Copy-Paste (Direct)':<22} {cp['mean']:>8.3f} {cp['std']:>8.3f} {cp['median']:>8.3f} {cp['n']:>6}",
        "-" * 56,
        f"  Δ (Copy-Paste − CASDA) : {delta:+.3f}",
        f"  Ratio (CASDA / CP)     : {ratio:.4f}",
        "",
        "Hypothesis: CASDA BGD < Copy-Paste BGD",
        f"Result    : {'SUPPORTED' if delta > 0 else 'NOT SUPPORTED'} (Δ = {delta:+.3f})",
    ]

    out = output_dir / "bgd_summary.txt"
    _mkdir(out)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved: {out.name}")
    print()
    print("\n".join(lines))


def write_boxplot(
    output_dir: Path,
    casda_scores: list[float],
    cp_scores: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    bp = ax.boxplot(
        [casda_scores, cp_scores],
        labels=["CASDA\n(Seamless Blend)", "Copy-Paste\n(Direct Paste)"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    colors = ["#4C72B0", "#DD8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    casda_mean = float(np.mean(casda_scores))
    cp_mean = float(np.mean(cp_scores))
    ax.axhline(casda_mean, color=colors[0], linestyle=":", linewidth=1.2, alpha=0.9)
    ax.axhline(cp_mean, color=colors[1], linestyle=":", linewidth=1.2, alpha=0.9)

    # 레이블 겹침 방지: y 범위의 3% 미만 간격이면 오프셋 적용
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_gap = y_range * 0.03
    casda_label_y = casda_mean
    cp_label_y = cp_mean
    if abs(casda_mean - cp_mean) < min_gap:
        offset = min_gap * 0.6
        casda_label_y = casda_mean + offset
        cp_label_y = cp_mean - offset

    x_offset = 2.35
    ax.text(x_offset, casda_label_y, f" μ={casda_mean:.1f}",
            color=colors[0], va="center", fontsize=9)
    ax.text(x_offset, cp_label_y, f" μ={cp_mean:.1f}",
            color=colors[1], va="center", fontsize=9)

    ax.set_ylabel("Boundary Gradient Discontinuity (BGD)", fontsize=11)
    ax.set_title(
        "Compositing Boundary Quality\n(Lower BGD = More Natural Boundary)",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out = output_dir / "bgd_boxplot.png"
    _mkdir(out)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    casda_dir = Path(args.casda_dir)
    cp_dir = Path(args.copypaste_dir)
    output_dir = Path(args.output_dir)
    hint_dir = Path(args.hint_dir) if args.hint_dir else None

    for d, flag in [(casda_dir, "--casda-dir"), (cp_dir, "--copypaste-dir")]:
        if not d.exists():
            print(f"[ERROR] {flag} 경로 없음: {d}", file=sys.stderr)
            sys.exit(1)

    if args.sample_size is not None and args.sample_size <= 0:
        print("[ERROR] --sample-size는 1 이상이어야 합니다.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[BGD] kernel_size={args.kernel_size}  workers={args.workers}  seed={args.seed}")

    # ── Load tasks ──
    print("\n[1/4] metadata 로드...")
    casda_tasks = load_group(casda_dir, "CASDA")
    cp_tasks = load_group(cp_dir, "Copy-Paste")
    print(f"  CASDA      : {len(casda_tasks):,} entries")
    print(f"  Copy-Paste : {len(cp_tasks):,} entries")

    # ── Compute BGD ──
    print("\n[2/4] CASDA BGD 계산...")
    casda_raw, casda_skips = compute_group(
        casda_tasks, args.workers, args.kernel_size, hint_dir, "CASDA"
    )
    print(f"  유효: {len(casda_raw):,}")

    print("\n[3/4] Copy-Paste BGD 계산...")
    cp_raw, cp_skips = compute_group(
        cp_tasks, args.workers, args.kernel_size, hint_dir, "Copy-Paste"
    )
    print(f"  유효: {len(cp_raw):,}")

    if not casda_raw or not cp_raw:
        print("[ERROR] 유효 점수 없음. hint-dir 경로 또는 dataset 구조를 확인하세요.", file=sys.stderr)
        sys.exit(1)

    # ── N 균형 (점수 계산 완료 후 적용) ──
    target_n = args.sample_size if args.sample_size else min(len(casda_raw), len(cp_raw))
    casda_bal = subsample(casda_raw, target_n, args.seed)
    cp_bal = subsample(cp_raw, target_n, args.seed)
    print(
        f"\n  N 균형: CASDA {len(casda_raw):,}→{len(casda_bal):,}, "
        f"Copy-Paste {len(cp_raw):,}→{len(cp_bal):,}"
    )

    # ── Output ──
    print("\n[4/4] 결과 저장...")
    write_json(output_dir, casda_raw, cp_raw, casda_bal, cp_bal,
               casda_skips, cp_skips, args.kernel_size, args.seed)
    write_summary_txt(output_dir, casda_bal, cp_bal, args.kernel_size)
    write_boxplot(output_dir, casda_bal, cp_bal)

    print("\n[BGD] 완료")


if __name__ == "__main__":
    main()
