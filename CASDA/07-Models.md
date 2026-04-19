# 07 — 벤치마크 모델

> **Claude 요약:** CASDA 벤치마크에 사용된 3개 모델. YOLO-MFD와 EB-YOLOv8는 탐지(Detection), DeepLabV3+는 분할(Segmentation). 모두 YOLOv8s 또는 ResNet-101 기반.

## 모델 비교

| 모델 | 유형 | Base | 핵심 개선 | 소스 |
|------|------|------|-----------|------|
| **YOLO-MFD** | Detection | YOLOv8s | MEFE 모듈 — 멀티스케일 Sobel 엣지 + 채널 어텐션 | `src/models/yolo_mfd.py` |
| **EB-YOLOv8** | Detection | YOLOv8s | BiFPN — 양방향 특징 피라미드 + 정규화 가중치 융합 | `src/models/eb_yolov8.py` |
| **DeepLabV3+** | Segmentation | ResNet-101 | ASPP (atrous rates 6/12/18) 인코더-디코더 | `src/models/deeplabv3plus.py` |

## YOLO-MFD

**Multi-scale Feature-enhanced Defect detector**

- Base: YOLOv8s
- 핵심 모듈: **MEFE (Multi-scale Edge Feature Enhancement)**
  - 멀티스케일 Sobel 엣지 추출 (미세 결함 강조)
  - 채널 어텐션 메커니즘
  - 목적: 미세 결함(micro-defect) 탐지 성능 향상
- 평가 지표: mAP@0.5, per-class AP, precision-recall
- 소스: `src/models/yolo_mfd.py`

## EB-YOLOv8

**Edge-BiFPN YOLOv8**

- Base: YOLOv8s
- 핵심 모듈: **BiFPN (Bi-directional Feature Pyramid Network)**
  - 양방향 특징 피라미드 (top-down + bottom-up)
  - 빠른 정규화 가중치 융합 (fast normalized weighted fusion)
  - Depthwise separable convolution으로 효율화
- 평가 지표: mAP@0.5, per-class AP, precision-recall
- 소스: `src/models/eb_yolov8.py`

## DeepLabV3+

- Base: ResNet-101
- 핵심 모듈: **ASPP (Atrous Spatial Pyramid Pooling)**
  - Atrous rates: 6, 12, 18
  - Output stride: 16
  - 인코더-디코더 구조
- 평가 지표: Dice coefficient (전체 + 클래스별), IoU
- 역할: Stage D Step 12에서 최적 분할 비율 탐색의 기준 모델
- 소스: `src/models/deeplabv3plus.py`

## 벤치마크 실행 그룹

Step 13에서 3개 모델 모두, Step 14에서 YOLO-MFD만 실행:

```bash
# Step 13: 3개 모델 × 5개 그룹
--models yolo_mfd eb_yolov8 deeplabv3plus  (기본값)

# Step 14: YOLO-MFD만 × 2개 ablation 그룹
--models yolo_mfd
```

## 관련 노트

[[00-INDEX]] | [[05-Pipeline-StageD]] | [[08-Dataset-Groups]]
