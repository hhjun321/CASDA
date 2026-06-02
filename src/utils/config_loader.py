"""
CASDA Config Loader

Stage 0(analyze_dataset.py)가 생성한 recommended_config.yaml을
Stage A~C 스크립트에서 로드하기 위한 공통 유틸리티.

우선순위: CLI 인자 > config 파일 > 코드 기본값
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# 유효 범위 스키마
# ──────────────────────────────────────────────────────────────────────────────

_RANGES: dict[str, tuple[float, float]] = {
    # DefectCharacterizer
    'high_linearity':       (0.0, 1.0),
    'high_aspect_ratio':    (1.0, 100.0),
    'low_aspect_ratio':     (1.0, 100.0),
    'high_solidity':        (0.0, 1.0),
    'low_solidity':         (0.0, 1.0),
    # BackgroundAnalyzer (내부 인자명 기준)
    'variance_threshold':   (0.0, 1e6),
    'edge_threshold':       (0.0, 1.0),
    'weak_edge':            (0.0, 1e6),   # YAML: total_strength
    'stripe_ratio_v':       (1.0, 10.0),
    'stripe_ratio_h':       (1.0, 10.0),
    'complex_freq':         (0.0, 1.0),   # YAML: high_freq_ratio
    # Pipeline parameters
    'min_suitability':      (0.0, 1.0),
    'min_quality_score':    (0.0, 1.0),
    'per_class_cap':        (1.0, 1e6),
    'rare_class_threshold': (0.0, 1e6),
}


def _validate_value(key: str, value: Any) -> None:
    """값 타입·범위 검증. 위반 시 sys.exit(1)."""
    if not isinstance(value, (int, float)):
        print(f"[config] 오류: '{key}' 값이 숫자가 아닙니다: {value!r}")
        sys.exit(1)
    lo, hi = _RANGES.get(key, (-1e18, 1e18))
    if not (lo <= float(value) <= hi):
        print(f"[config] 오류: '{key}' = {value} 는 허용 범위 [{lo}, {hi}] 밖입니다.")
        sys.exit(1)


def load_recommended_config(path: str | Path | None) -> dict:
    """
    recommended_config.yaml 로드.

    - path=None 또는 파일 미존재: 빈 dict 반환 (경고 1줄 출력)
    - path 지정 + 파일 미존재: 하드 에러 sys.exit(1)
    - YAML 파싱 오류: 하드 에러 sys.exit(1)
    """
    if path is None:
        return {}

    p = Path(path)
    if not p.exists():
        print(f"[config] 오류: config 파일을 찾을 수 없습니다: {p}")
        sys.exit(1)

    try:
        with open(p, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[config] YAML 파싱 오류: {p}\n  {e}")
        sys.exit(1)

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        print(f"[config] 오류: config 최상위가 dict여야 합니다: {p}")
        sys.exit(1)

    return cfg


def resolve(cli_val: Any, cfg_val: Any, default: Any) -> Any:
    """
    CLI > config > default 우선순위 적용.
    cli_val 이 None 이면 cfg_val 시도, 그것도 None 이면 default 사용.
    """
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return cfg_val
    return default


# ──────────────────────────────────────────────────────────────────────────────
# 섹션별 추출 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

# recommended_config.yaml의 YAML 키 → DefectCharacterizer.__init__ 인자명 매핑
_DEFECT_KEY_MAP: dict[str, str] = {
    'HIGH_LINEARITY':       'high_linearity',
    'HIGH_ASPECT_RATIO':    'high_aspect_ratio',
    'LOW_ASPECT_RATIO':     'low_aspect_ratio',
    'HIGH_SOLIDITY':        'high_solidity',
    'LOW_SOLIDITY':         'low_solidity',
}

# BackgroundAnalyzer YAML 키 → __init__ 인자명 매핑
# analyze_dataset.py 출력 키명(YAML) → BackgroundAnalyzer 내부 속성명
_BG_KEY_MAP: dict[str, str] = {
    'variance_threshold': 'variance_threshold',
    'edge_threshold':     'edge_threshold',
    'total_strength':     'weak_edge',       # YAML: total_strength → 내부: weak_edge
    'stripe_ratio_v':     'stripe_ratio_v',
    'stripe_ratio_h':     'stripe_ratio_h',
    'high_freq_ratio':    'complex_freq',    # YAML: high_freq_ratio → 내부: complex_freq
}

_PIPELINE_KEYS = frozenset({
    'min_suitability', 'per_class_cap', 'rare_class_threshold',
    'min_quality_score', 'num_images_per_class', 'compositions_per_roi',
})


def get_defect_thresholds(cfg: dict) -> dict:
    """
    `defect_characterizer_thresholds` 섹션 추출 + 범위 검증.
    YAML 대문자 키 → 소문자 인자명 변환.
    알 수 없는 키는 경고 출력 후 무시.
    """
    raw = cfg.get('defect_characterizer_thresholds') or {}
    result: dict = {}
    for yaml_key, arg_name in _DEFECT_KEY_MAP.items():
        if yaml_key in raw:
            v = raw[yaml_key]
            _validate_value(arg_name, v)
            result[arg_name] = float(v)
    unknown = set(raw) - set(_DEFECT_KEY_MAP)
    if unknown:
        print(f"[config] 경고: defect_characterizer_thresholds 알 수 없는 키: {unknown}")
    return result


def get_bg_thresholds(cfg: dict) -> dict:
    """
    `background_analyzer_thresholds` 섹션 추출 + 범위 검증.
    YAML 키 → BackgroundAnalyzer 내부 인자명 변환 (_BG_KEY_MAP).
    """
    raw = cfg.get('background_analyzer_thresholds') or {}
    result: dict = {}
    for yaml_key, arg_name in _BG_KEY_MAP.items():
        if yaml_key in raw:
            v = raw[yaml_key]
            _validate_value(arg_name, v)
            result[arg_name] = float(v)
    unknown = set(raw) - set(_BG_KEY_MAP)
    if unknown:
        print(f"[config] 경고: background_analyzer_thresholds 알 수 없는 키: {unknown}")
    return result


def get_pipeline_params(cfg: dict) -> dict:
    """
    `pipeline_parameters` 섹션 추출 + 범위 검증.
    """
    raw = cfg.get('pipeline_parameters') or {}
    result: dict = {}
    for k in _PIPELINE_KEYS:
        if k in raw:
            v = raw[k]
            if k in _RANGES:
                _validate_value(k, v)
            result[k] = v
    return result
