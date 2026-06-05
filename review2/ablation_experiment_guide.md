# Review-2 Ablation & Baseline 실험 실행 가이드

Review-2 Comment 5 (baseline 비교 부족) + Comment 7 (ablation study 미비) 대응 실험.

## 컴포넌트 매핑 요약

| 컴포넌트 | Variant | 대응 Comment |
|---|---|---|
| Compatibility matrix | `ablation_no_compat` | C7 전용 |
| Three-channel hint image | `ablation_1ch_hint` | C7 전용 |
| Prompt construction | `ablation_generic_prompt` | C7 전용 |
| Quality gate | `ablation_no_pruning` (기존) | C7 전용 |
| **ControlNet conditioning** | **`baseline_vanilla_sd`** | **C5 + C7 겸임** |

> `ablation_no_pruning`은 기존 `casda_composed` 데이터를 재사용. 신규 생성 불필요.

## 스크립트별 실행 환경

| 스크립트 | 환경 | 비고 |
|---|---|---|
| `test_controlnet.py` | **GPU** (CUDA) | ControlNet + SD v1.5 추론 |
| `run_vanilla_sd.py` | **GPU** (CUDA) | SD v1.5 text-to-image 추론 |
| `compose_casda_images.py` | **CPU** | Poisson Blending (OpenCV/PIL), `--workers`로 병렬화 |
| `run_benchmark.py` | **GPU** (CUDA) | YOLO 모델 학습 |

---

## 환경 변수 (Colab 셀에서 먼저 실행)

```python
import os
DRIVE = os.environ['DRIVE']

os.environ['SCRIPTS']     = "/content/CASDA/scripts"
os.environ['CN_DATASET']  = f"{DRIVE}/controlnet_dataset_v5.5"
os.environ['BEST_MODEL']  = f"{DRIVE}/controlnet_training_v5.5/best_model"
os.environ['BG_CACHE']    = f"{DRIVE}/bg_cache.pkl"
os.environ['ABL_BASE']    = f"{DRIVE}/augmented_dataset_ablation"

# 아래는 기존 setup 셀에서 이미 os.environ에 등록됨 — 재확인용
# AUG_IMAGES, TRAIN_IMAGES, TRAIN_CSV
```

---

## Step 1: ablation_no_compat

기존 생성 결과(`$AUG_IMAGES/generated`)를 재사용. 재생성 불필요.

```python
# [CPU] compose_casda_images.py — Poisson Blending, GPU 불필요
!python $SCRIPTS/compose_casda_images.py \
    --generated-dir $AUG_IMAGES/generated \
    --hint-dir $CN_DATASET/hints \
    --metadata-csv $CN_DATASET/packaged_roi_metadata.csv \
    --summary-json $AUG_IMAGES/generation_summary.json \
    --clean-images-dir $TRAIN_IMAGES \
    --train-csv $TRAIN_CSV \
    --output-dir $ABL_BASE/casda_no_compat \
    --no-compatibility \
    --workers 8 \
    --bg-cache $BG_CACHE \
    --compositions-per-roi 5
```

> **주의**: `--no-compatibility` 지정 시 밝기 매칭(brightness tolerance)도 함께 제거됩니다.
> ablation 목적은 호환성 매트릭스 단독 제거이므로, 결과 해석 시 이 점을 논문에 명기합니다.

---

## Step 2: ablation_generic_prompt

```python
os.environ['GENERIC_AUG'] = f"{os.environ['DRIVE']}/augmented_images_generic_prompt"

# [GPU] test_controlnet.py — ControlNet + SD v1.5 추론, CUDA 필요
!python $SCRIPTS/test_controlnet.py \
    --model_path $BEST_MODEL \
    --jsonl_path $CN_DATASET/train.jsonl \
    --output_dir $GENERIC_AUG \
    --generic-prompt "Industrial steel surface defect on metal surface" \
    --num_inference_steps 30 \
    --guidance_scale 7.5 \
    --controlnet_conditioning_scale 0.7 \
    --num_images_per_class '{"1":2,"2":10,"3":1,"4":2}' \
    --seed 42

# [CPU] compose_casda_images.py — Poisson Blending, GPU 불필요
!python $SCRIPTS/compose_casda_images.py \
    --generated-dir $GENERIC_AUG/generated \
    --hint-dir $CN_DATASET/hints \
    --metadata-csv $CN_DATASET/packaged_roi_metadata.csv \
    --summary-json $GENERIC_AUG/generation_summary.json \
    --clean-images-dir $TRAIN_IMAGES \
    --train-csv $TRAIN_CSV \
    --output-dir $ABL_BASE/casda_generic_prompt \
    --workers 8 \
    --bg-cache $BG_CACHE \
    --compositions-per-roi 5
```

---

## Step 3: ablation_1ch_hint

```python
os.environ['MASK_AUG'] = f"{os.environ['DRIVE']}/augmented_images_1ch_hint"

# [GPU] test_controlnet.py — ControlNet + SD v1.5 추론, CUDA 필요
!python $SCRIPTS/test_controlnet.py \
    --model_path $BEST_MODEL \
    --jsonl_path $CN_DATASET/train.jsonl \
    --output_dir $MASK_AUG \
    --mask-only-hint \
    --num_inference_steps 30 \
    --guidance_scale 7.5 \
    --controlnet_conditioning_scale 0.7 \
    --num_images_per_class '{"1":2,"2":10,"3":1,"4":2}' \
    --seed 42

# [CPU] compose_casda_images.py — Poisson Blending, GPU 불필요
!python $SCRIPTS/compose_casda_images.py \
    --generated-dir $MASK_AUG/generated \
    --hint-dir $CN_DATASET/hints \
    --metadata-csv $CN_DATASET/packaged_roi_metadata.csv \
    --summary-json $MASK_AUG/generation_summary.json \
    --clean-images-dir $TRAIN_IMAGES \
    --train-csv $TRAIN_CSV \
    --output-dir $ABL_BASE/casda_1ch_hint \
    --workers 8 \
    --bg-cache $BG_CACHE \
    --compositions-per-roi 5
```

---

## Step 4: baseline_vanilla_sd (C5 + C7 겸임)

ControlNet 없이 SD v1.5 text-to-image만 사용.
- **C5 관점**: generative/diffusion-based baseline
- **C7 관점**: ControlNet conditioning 컴포넌트 제거 ablation

```python
os.environ['VANILLA_AUG'] = f"{os.environ['DRIVE']}/vanilla_sd_images"

# [GPU] run_vanilla_sd.py — SD v1.5 text-to-image 추론, CUDA 필요 (ControlNet 없음)
!python $SCRIPTS/run_vanilla_sd.py \
    --jsonl_path $CN_DATASET/train.jsonl \
    --output_dir $VANILLA_AUG \
    --num_inference_steps 30 \
    --guidance_scale 7.5 \
    --num_images_per_class '{"1":2,"2":10,"3":1,"4":2}' \
    --seed 42

# [CPU] compose_casda_images.py — Poisson Blending, GPU 불필요
# hint_dir는 기존 CN_DATASET/hints 참조 (마스크 추출용)
!python $SCRIPTS/compose_casda_images.py \
    --generated-dir $VANILLA_AUG/generated \
    --hint-dir $CN_DATASET/hints \
    --metadata-csv $CN_DATASET/packaged_roi_metadata.csv \
    --summary-json $VANILLA_AUG/generation_summary.json \
    --clean-images-dir $TRAIN_IMAGES \
    --train-csv $TRAIN_CSV \
    --output-dir $ABL_BASE/casda_vanilla_sd \
    --workers 8 \
    --bg-cache $BG_CACHE \
    --compositions-per-roi 5
```

> `run_vanilla_sd.py`는 hint를 conditioning에 사용하지 않지만,
> `generation_summary.json`에 `hint_path` 필드를 기록합니다.
> compose 단계가 이 필드로 마스크를 추출하므로 `--hint-dir`을 반드시 지정합니다.

---

## Step 5: 벤치마크 실행

```python
# [GPU] run_benchmark.py — YOLO 모델 학습, CUDA 필요
!python $SCRIPTS/run_benchmark.py \
    --config $CONFIG \
    --data-dir /content/dataset_local/train_images \
    --models yolo_mfd \
    --groups no_compat generic_prompt 1ch_hint vanilla_sd no_pruning \
    --casda-dir $ABL_BASE \
    --yolo-dir $YOLO_DATASETS \
    --output-dir $BENCHMARK_RESULTS \
    --no-fid \
    --reference-results $BENCHMARK_RESULTS/benchmark_results.json
```

> `ablation_no_pruning`은 `casda_composed`를 재사용하므로 `--casda-dir`을 `$AUG_DATASET`으로 별도 지정하거나
> `casda_dir_override` 없는 그룹은 기존 경로를 자동 참조합니다.

---

## ablation table 구성 (실험 완료 후)

| 제거 컴포넌트 | Variant | mAP@0.5 |
|---|---|---|
| (Full CASDA) | casda_composed_pruning | — |
| w/o Blending | ablation_no_blending | — |
| w/o Quality Gate | ablation_no_pruning | — |
| w/o Compat. Matrix | ablation_no_compat | — |
| w/o Context Prompt | ablation_generic_prompt | — |
| w/o 3-ch Hint | ablation_1ch_hint | — |
| Vanilla SD (C5+C7) | baseline_vanilla_sd | — |
