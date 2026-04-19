# 09 — 실험 기록

> **Claude 요약:** 완료된 실험 결과와 향후 실험 기록용 템플릿. 새 실험 시 템플릿 섹션을 복사해서 사용.

---

## 완료된 실험

### 전체 파이프라인 실험 — 2025

**목적:** CASDA 증강 프레임워크 전체 효과 검증

**설정:**
- 환경: Google Colab (T4 GPU)
- 데이터셋: Severstal Steel Defect Detection
- 모델: YOLO-MFD, EB-YOLOv8, DeepLabV3+
- 그룹: baseline_raw, baseline_trad, casda_composed, casda_composed_pruning, copypaste
- Ablation (YOLO-MFD): ablation_no_blending, ablation_no_pruning

**ControlNet 학습 설정:**
- Base: SD v1.5 + sd-controlnet-canny
- epochs: 20, batch: 1 × grad_accum 4, lr: 1e-5 (cosine)
- mixed_precision: fp16, seed: 42

**생성 설정:**
- num_inference_steps: 30, guidance_scale: 7.5
- controlnet_conditioning_scale: 0.7
- num_images_per_class: {"1":2, "2":10, "3":1, "4":2}

**결과:**
<!-- 실험 완료 후 여기에 핵심 지표 기록 -->
- FID (ROI): 
- FID (full-image): 
- YOLO-MFD mAP@0.5 (casda_composed_pruning vs baseline_raw): 
- EB-YOLOv8 mAP@0.5: 
- DeepLabV3+ Dice: 
- 최적 분할 비율: 

**관찰:**
<!-- 주요 발견 사항, 예상과 다른 결과, 다음 실험 아이디어 -->

**참조 파일:**
- `${BENCHMARK_RESULTS}/`
- `${FID_RESULTS}/`
- `${SPLIT_RESULTS}/`

---

## 새 실험 템플릿

> 아래 템플릿을 복사하여 새 실험 기록에 사용하세요.

```
### [실험명] — YYYY-MM-DD

**목적:**

**설정:**
- 환경:
- 변경 사항 (이전 실험 대비):
- 주요 파라미터:

**실행 명령 (변경된 부분만):**
\`\`\`bash

\`\`\`

**결과:**
- 지표 1:
- 지표 2:

**결론:**

**다음 실험 아이디어:**
```

---

## 실험 아이디어 목록

<!-- 향후 시도할 실험 아이디어를 여기에 메모 -->

- [ ] 

## 관련 노트

[[00-INDEX]] | [[01-Overview]] | [[06-Scripts-Reference]] | [[08-Dataset-Groups]]
