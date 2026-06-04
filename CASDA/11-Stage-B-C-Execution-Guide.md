# 11 — Stage A→C 실행 가이드 (ControlNet 재패키징 → 학습 → 생성 → 합성)

> **Claude 요약:** Stage A 완료 후 ControlNet 데이터 재패키징(prepare_controlnet_data.py) → 학습(train_controlnet.py) → 합성 이미지 생성(test_controlnet.py) → 배경 합성(compose_casda_images.py) 까지 Colab Pro에서 순서대로 실행하는 가이드. `recommended_config.yaml` 도출값(num_images_per_class, compositions_per_roi 등)을 반영한 최신 파라미터 포함.

---

## 사전 조건

| 항목 | 확인 |
|------|------|
| Stage A (`extract_rois.py`) 완료 — `$ROI_DIR/roi_metadata.csv` 존재 | 필요 |
| `$ANALYSIS_CONFIG` — `recommended_config.yaml` 존재 | 필요 |
| Colab Pro (A100 / L4 권장, 최소 T4) | GPU 필수 (Step 2~3) |
| `approve` 브랜치 클론 완료 | 필요 |

---

## 환경 준비

```python
import os, json, yaml

# 기본 경로 변수 (colab-execution.md 기준)
DRIVE    = os.environ['DRIVE']
SCRIPTS  = os.environ['SCRIPTS']

# 분석 결과
os.environ['ANALYSIS_DIR']    = f"{DRIVE}/analysis"
os.environ['ANALYSIS_CONFIG'] = f"{DRIVE}/analysis/recommended_config.yaml"

# Stage A 출력
os.environ['ROI_DIR']    = f"{DRIVE}/roi_patches_v5.1"

# Stage A→B 연결
os.environ['CN_DATASET']  = f"{DRIVE}/controlnet_dataset"
os.environ['CN_TRAINING'] = f"{DRIVE}/controlnet_training"
os.environ['CN_VALIDATION'] = f"{DRIVE}/controlnet_validation"
os.environ['BEST_MODEL']  = f"{DRIVE}/controlnet_training/best_model"

# Stage B→C 연결
os.environ['AUG_IMAGES']       = f"{DRIVE}/augmented_images_v5.5"
os.environ['AUG_DATASET']      = f"{DRIVE}/augmented_dataset_v5.6"
os.environ['CASDA_COMPOSED']   = f"{DRIVE}/augmented_dataset_v5.6/casda_composed"
os.environ['BG_CACHE']         = f"{DRIVE}/cache/bg_types.json"

# recommended_config.yaml 로드
with open(os.environ['ANALYSIS_CONFIG']) as f:
    cfg = yaml.safe_load(f)

pipeline = cfg['pipeline_parameters']
num_images_per_class = pipeline['num_images_per_class']
compositions_per_roi = pipeline['compositions_per_roi']

print("=== 데이터 기반 파라미터 ===")
print(f"num_images_per_class : {num_images_per_class}")
print(f"compositions_per_roi : {compositions_per_roi}")
print(f"per_class_cap        : {pipeline['per_class_cap']}")
print(f"rare_class_threshold : {pipeline['rare_class_threshold']}")
```

---

## Step 1: prepare_controlnet_data.py — ControlNet 학습 데이터 재패키징

> Stage A의 ROI 메타데이터를 ControlNet 학습용 JSONL + 힌트 이미지로 패키징한다. CPU 전용.

### 사전 확인

```python
import os
roi_meta = f"{os.environ['ROI_DIR']}/roi_metadata.csv"
assert os.path.exists(roi_meta), f"roi_metadata.csv 없음: {roi_meta}"

import pandas as pd
df = pd.read_csv(roi_meta)
print(f"ROI 메타데이터: {len(df):,}개")
print("클래스 분포:")
print(df['class_id'].value_counts().sort_index().to_string())
```

### 실행

```python
!python $SCRIPTS/prepare_controlnet_data.py \
    --roi_metadata         $ROI_DIR/roi_metadata.csv \
    --train_images         $TRAIN_IMAGES \
    --train_csv            $TRAIN_CSV \
    --output_dir           $CN_DATASET \
    --per_class_cap        11178 \
    --rare_class_threshold 1510 \
    --class_edge_override  "4:0.05,0.0" \
    --skip_validation
```

> `per_class_cap=11178`, `rare_class_threshold=1510` 은 `recommended_config.yaml` 도출값.

### 완료 확인

```python
import os, json

cn_dataset = os.environ['CN_DATASET']
jsonl_path = f"{cn_dataset}/train.jsonl"
meta_path  = f"{cn_dataset}/packaged_roi_metadata.csv"

assert os.path.exists(jsonl_path), "train.jsonl 없음"
assert os.path.exists(meta_path),  "packaged_roi_metadata.csv 없음"

# JSONL 샘플 수 확인
with open(jsonl_path) as f:
    n_samples = sum(1 for _ in f)

print(f"학습 샘플 수: {n_samples:,}")
print(f"hints 디렉토리: {cn_dataset}/hints/")

import glob
n_hints = len(glob.glob(f"{cn_dataset}/hints/*.png"))
print(f"힌트 이미지 수: {n_hints:,}")
```

---

## Step 2: train_controlnet.py — ControlNet 재학습

> SD v1.5 + sd-controlnet-canny 기반으로 ControlNet 파인튜닝. **GPU 필수**.

### GPU 확인

```python
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                        '--format=csv,noheader'], capture_output=True, text=True)
print(result.stdout.strip())
```

### 실행

```python
!python $SCRIPTS/train_controlnet.py \
    --data_dir                        $CN_DATASET \
    --output_dir                      $CN_TRAINING \
    --pretrained_model_name_or_path   runwayml/stable-diffusion-v1-5 \
    --controlnet_model_name_or_path   lllyasviel/sd-controlnet-canny \
    --resolution 512 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --mixed_precision fp16 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --lr_scheduler cosine \
    --lr_warmup_steps 50 \
    --controlnet_conditioning_scale 1.0 \
    --snr_gamma 5.0 \
    --early_stopping_patience 20 \
    --validation_steps 200 \
    --logging_steps 10 \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 3 \
    --save_fp16 \
    --skip_save_pipeline \
    --seed 42
```

### 학습 완료 확인

```python
import os, json

best_model = os.environ['BEST_MODEL']
log_path   = f"{os.environ['CN_TRAINING']}/training_log.json"

assert os.path.exists(best_model), f"best_model 디렉토리 없음: {best_model}"

if os.path.exists(log_path):
    with open(log_path) as f:
        log = json.load(f)
    print(f"최종 에폭: {log.get('final_epoch', '?')}")
    print(f"최적 검증 손실: {log.get('best_val_loss', '?'):.4f}")
    print(f"조기 종료: {log.get('early_stopped', False)}")
else:
    print("학습 로그 없음 — best_model 폴더 직접 확인")
    print(os.listdir(best_model))
```

---

## Step 3: test_controlnet.py — 합성 결함 이미지 생성

> 학습된 모델로 클래스별 합성 결함 이미지 생성. **GPU 필수**.

### num_images_per_class 확인

```python
# recommended_config.yaml 기반 값
# 클래스 2(희귀): 46장, 클래스 3(지배적): 1장
print("=== 생성 계획 ===")
for cls, n in sorted(num_images_per_class.items()):
    tag = " ← 희귀" if int(n) >= 30 else (" ← 지배적" if int(n) == 1 else "")
    print(f"  Class {cls}: {n}장{tag}")

# JSON 문자열로 변환 (CLI 인자용)
import json
num_images_json = json.dumps({str(k): int(v) for k, v in num_images_per_class.items()})
print(f"\n--num_images_per_class 인자: '{num_images_json}'")
```

### 실행

```python
import json as _json

_num_images_json = _json.dumps(
    {str(k): int(v) for k, v in num_images_per_class.items()}
)

!python $SCRIPTS/test_controlnet.py \
    --model_path    $BEST_MODEL \
    --jsonl_path    $CN_DATASET/train.jsonl \
    --output_dir    $AUG_IMAGES \
    --num_inference_steps 30 \
    --guidance_scale 7.5 \
    --controlnet_conditioning_scale 0.7 \
    --num_images_per_class "$_num_images_json" \
    --resolution 512
```

> **주의**: `--controlnet_conditioning_scale 0.7` (학습 1.0 → 생성 0.7, 네온 아티팩트 방지)

### 생성 결과 확인

```python
import os, json, glob

aug_images = os.environ['AUG_IMAGES']
generated_dir = f"{aug_images}/generated"
summary_path  = f"{aug_images}/generation_summary.json"

assert os.path.exists(generated_dir), f"generated 디렉토리 없음: {generated_dir}"

n_generated = len(glob.glob(f"{generated_dir}/*.png"))
print(f"생성된 이미지: {n_generated:,}장")

if os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
    print("\n클래스별 생성 수:")
    for cls, cnt in sorted(summary.get('per_class_counts', {}).items()):
        print(f"  Class {cls}: {cnt}장")
```

---

## Step 4: compose_casda_images.py — 배경 합성 (Poisson Blending)

> 생성된 512×512 ROI를 실제 1600×256 배경에 Poisson Blending으로 합성한다. CPU 전용.

### 사전 확인

```python
import os

aug_images = os.environ['AUG_IMAGES']
cn_dataset = os.environ['CN_DATASET']

assert os.path.exists(f"{aug_images}/generated"), "generated 디렉토리 없음 — Step 3 확인"
assert os.path.exists(f"{cn_dataset}/hints"),     "hints 디렉토리 없음 — Step 1 확인"
assert os.path.exists(f"{cn_dataset}/packaged_roi_metadata.csv"), "packaged_roi_metadata.csv 없음"
assert os.path.exists(f"{aug_images}/generation_summary.json"),   "generation_summary.json 없음"

print(f"compositions_per_roi: {compositions_per_roi}")
print("사전 확인 완료")
```

### 실행

```python
# BG_CACHE 디렉토리 생성 (없으면)
import os
os.makedirs(os.path.dirname(os.environ['BG_CACHE']), exist_ok=True)
os.makedirs(os.environ['CASDA_COMPOSED'], exist_ok=True)

!python $SCRIPTS/compose_casda_images.py \
    --generated-dir       $AUG_IMAGES/generated \
    --hint-dir            $CN_DATASET/hints \
    --metadata-csv        $CN_DATASET/packaged_roi_metadata.csv \
    --summary-json        $AUG_IMAGES/generation_summary.json \
    --clean-images-dir    $TRAIN_IMAGES \
    --train-csv           $TRAIN_CSV \
    --output-dir          $CASDA_COMPOSED \
    --workers 8 \
    --bg-cache            $BG_CACHE \
    --compositions-per-roi 6
```

> `--compositions-per-roi 6` 은 `recommended_config.yaml` 도출값.

### 합성 결과 확인

```python
import os, glob

casda = os.environ['CASDA_COMPOSED']
n_composed = len(glob.glob(f"{casda}/*.jpg")) + len(glob.glob(f"{casda}/*.png"))
print(f"합성 완료: {n_composed:,}장")

meta_path = f"{casda}/metadata.json"
if os.path.exists(meta_path):
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"메타데이터 항목: {len(meta):,}개")
    # 클래스별 분포
    from collections import Counter
    cls_dist = Counter(str(v.get('class_id','?')) for v in meta.values())
    print("클래스별 합성 수:")
    for cls, cnt in sorted(cls_dist.items()):
        print(f"  Class {cls}: {cnt:,}장")
```

---

## 다음 단계

| Step | 스크립트 | Stage | 목적 |
|------|----------|-------|------|
| 품질 점수 산출 | `score_casda_quality.py` | C | suitability_score 계산 |
| 품질 검증 | `validate_augmented_quality.py` | C | 0.7 미만 제외 |
| FID 측정 | `run_fid.py` | D | 합성 품질 정량 평가 |
| 벤치마크 | `run_benchmark.py` | D | 모델 학습·성능 비교 |

→ 상세: [[04-Pipeline-StageC]] | [[05-Pipeline-StageD]]

---

## 트러블슈팅

| 증상 | 원인 | 조치 |
|------|------|------|
| `train.jsonl` 없음 | Step 1 미완료 | `$CN_DATASET` 디렉토리 확인 후 Step 1 재실행 |
| CUDA OOM (Step 2) | 배치/해상도 문제 | `--train_batch_size 1`, `--gradient_checkpointing` 확인 |
| 생성 이미지 네온 아티팩트 | conditioning_scale 과도 | `--controlnet_conditioning_scale` 0.5~0.7로 낮춤 |
| `bg_types.json` 없음 (Step 4) | 캐시 디렉토리 미생성 | `os.makedirs($DRIVE/cache)` 후 재실행 |
| 합성 이미지 0장 | `generated/` 비어있음 | Step 3 재확인 |

---

## 관련 노트

- [[00-INDEX]] — 전체 경로 변수
- [[02-Pipeline-StageA]] — Stage A 스크립트 상세
- [[03-Pipeline-StageB]] — Stage B 파라미터 상세
- [[04-Pipeline-StageC]] — Stage C 품질 관리
- [[06-Scripts-Reference]] — 스크립트 전체 입출력 매핑
- [[10-Colab-Reanalysis-Guide]] — Stage 0 재분석 가이드
