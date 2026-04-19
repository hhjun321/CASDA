# 03 — Stage B: ControlNet 학습 + 생성

> **Claude 요약:** Stage A에서 준비한 멀티채널 힌트로 ControlNet을 학습시키고(SD v1.5 기반), 4단계 검증 후 클래스별 합성 결함 이미지를 생성한다. GPU 필수.

## 목적

- SD v1.5 + sd-controlnet-canny 기반 ControlNet을 결함 이미지 합성에 파인튜닝
- 학습된 모델로 클래스별 합성 결함 이미지 생성
- 선택적 4단계 검증으로 모델 품질 확인

## 입력 / 출력

| 항목 | 경로 변수 | 설명 |
|------|-----------|------|
| **입력** | `CN_DATASET` | Stage A 출력 (힌트, train.jsonl) |
| **출력** | `CN_TRAINING` | 학습 체크포인트, 로그 |
| **출력** | `BEST_MODEL` | 최적 모델 (`${CN_TRAINING}/best_model`) |
| **출력** | `CN_VALIDATION` | 검증 결과 (선택) |
| **출력** | `AUG_IMAGES` | 생성된 합성 결함 이미지 |

## Step 3: train_controlnet.py — ControlNet 학습

```bash
!python ${SCRIPTS}/train_controlnet.py \
  --data_dir                        ${CN_DATASET} \
  --output_dir                      ${CN_TRAINING} \
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

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--resolution` | 512 | 학습 이미지 해상도 |
| `--train_batch_size` | 1 | 배치 크기 (T4 기준 1) |
| `--gradient_accumulation_steps` | 4 | 실질적 배치 크기 = 4 |
| `--mixed_precision` | fp16 | 메모리 절약 |
| `--num_train_epochs` | 20 | 최대 에폭 |
| `--learning_rate` | 1e-5 | 학습률 |
| `--lr_scheduler` | cosine | 스케줄러 |
| `--early_stopping_patience` | 20 | 조기 종료 patience |
| `--controlnet_conditioning_scale` | 1.0 | ControlNet 조건화 강도 |
| `--snr_gamma` | 5.0 | SNR 가중치 gamma |
| `--checkpoints_total_limit` | 3 | 저장할 최대 체크포인트 수 |

## Step 4: run_validation_phases.py — 모델 검증 (선택)

```bash
!python ${SCRIPTS}/run_validation_phases.py \
  --model_path          ${BEST_MODEL} \
  --jsonl_path          ${CN_DATASET}/train.jsonl \
  --image_root          ${CN_DATASET} \
  --roi_metadata_path   ${ROI_DIR}/roi_metadata.csv \
  --training_log_path   ${CN_TRAINING}/training_log.json \
  --output_base         ${CN_VALIDATION} \
  --phases 1 2 3 4 \
  --controlnet_conditioning_scale 0.7 \
  --resolution 512 \
  --workers 8
```

### 검증 4단계

| Phase | 내용 |
|-------|------|
| 1 | 기본 생성 품질 확인 |
| 2 | 힌트 조건화 충실도 |
| 3 | 클래스별 다양성 |
| 4 | 전체 파이프라인 통합 |

## Step 5: test_controlnet.py — 합성 이미지 생성

```bash
!python ${SCRIPTS}/test_controlnet.py \
  --model_path    ${BEST_MODEL} \
  --jsonl_path    ${CN_DATASET}/train.jsonl \
  --output_dir    ${AUG_IMAGES} \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --controlnet_conditioning_scale 0.7 \
  --num_images_per_class '{"1":2,"2":10,"3":1,"4":2}' \
  --resolution 512
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--num_inference_steps` | 30 | 디퓨전 추론 스텝 수 |
| `--guidance_scale` | 7.5 | CFG 스케일 |
| `--controlnet_conditioning_scale` | 0.7 | 학습(1.0)보다 낮게 설정 |
| `--num_images_per_class` | `{"1":2,"2":10,"3":1,"4":2}` | 클래스별 생성 이미지 수 (클래스 2 희귀하여 10개) |

### 출력

- `${AUG_IMAGES}/generated/` — 생성된 합성 결함 이미지 (512×512)
- `${AUG_IMAGES}/generation_summary.json` — 생성 요약

## 주의사항

- GPU 필수 (T4 이상)
- `--controlnet_conditioning_scale` 생성 시 0.7로 낮춤 (학습 1.0 → 생성 0.7)
- 클래스 2는 희귀 클래스로 다른 클래스보다 많이 생성
- `--skip_save_pipeline`으로 전체 파이프라인 저장 생략 (체크포인트만 저장)

## 관련 노트

[[00-INDEX]] | [[02-Pipeline-StageA]] | [[04-Pipeline-StageC]] | [[06-Scripts-Reference]]
