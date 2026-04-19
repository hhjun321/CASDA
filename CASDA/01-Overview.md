# 01 — 프로젝트 개요

> **Claude 요약:** CASDA는 ControlNet 기반 철강 결함 이미지 합성 + Poisson Blending 후처리 + 품질 필터링으로 데이터를 증강하고, 3개 모델(YOLO-MFD, EB-YOLOv8, DeepLabV3+)로 효과를 검증하는 연구 파이프라인이다.

## Abstract

철강 제조 표면 결함 탐지는 심각한 클래스 불균형, 제한된 결함 샘플, 벤치마크 데이터셋의 배경 다양성 부족으로 인해 어렵다. CASDA(Context-Aware Steel Defect Augmentation)는 기하학적 결함 특성 분석과 ControlNet 기반 조건부 이미지 합성을 결합하여 물리적으로 타당한 결함 이미지를 생성하는 생성적 데이터 증강 프레임워크다.

- 데이터셋: [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection/data)
- Backbone: Stable Diffusion v1.5 + ControlNet (sd-controlnet-canny)
- 평가: 3개 모델 아키텍처 × 7개 데이터셋 그룹

## 핵심 특징 (Highlights)

1. **ControlNet 기반 결함 합성** — 멀티채널 힌트 (R=결함 형태, G=구조, B=텍스처) + 하이브리드 텍스트 프롬프트
2. **Poisson Blending 합성** — 생성된 512×512 ROI를 실제 1600×256 철강 이미지에 자연스럽게 합성
3. **품질 인식 프루닝** — 적합도 점수 기반 필터링 (색상 일관성, 아티팩트 탐지, 선명도)
4. **3개 벤치마크 모델** — YOLO-MFD, EB-YOLOv8, DeepLabV3+
5. **7개 데이터셋 그룹** — 각 파이프라인 구성요소 기여도 격리 ablation 포함
6. **통계적 가설 검정** — 증강 효과의 엄격한 검증
7. **FID 평가** — ROI 수준 및 전체 이미지 수준, 클래스별·서브타입별 세분화

## 전체 파이프라인

```
Stage A: 데이터 전처리 (CPU)
  ├── ROI 패치 추출 + 적합도 점수
  └── 멀티채널 ControlNet 힌트 + 하이브리드 텍스트 프롬프트 준비
          ↓
Stage B: ControlNet 학습 + 생성 (GPU)
  ├── ControlNet 학습 (SD v1.5 + sd-controlnet-canny)
  ├── 멀티페이즈 검증 (선택)
  └── 합성 결함 이미지 생성
          ↓
Stage C: 후처리 + 품질관리 (CPU)
  ├── Poisson Blending 합성 (깨끗한 배경에 합성)
  ├── 적합도 점수 (색상 / 아티팩트 / 선명도)
  ├── 품질 인식 프루닝 (계층별 top-k)
  └── CopyPaste 베이스라인 생성
          ↓
Stage D: 평가 (GPU)
  ├── FID 평가 (ROI 수준 + 전체 이미지)
  ├── 최적 분할 비율 탐색
  ├── 벤치마크: 3개 모델 × 7개 데이터셋 그룹
  └── 통계 가설 검정 + 결과 분석
```

## 리포지토리 구조

```
CASDA/
├── configs/
│   └── benchmark_experiment.yaml
├── scripts/          # 실행 가능한 파이프라인 스크립트 (14개 핵심)
├── src/
│   ├── models/       # YOLO-MFD, EB-YOLOv8, DeepLabV3+
│   ├── training/     # 데이터셋, 트레이너, 평가기
│   ├── preprocessing/
│   ├── analysis/
│   └── utils/
├── QUICKSTART.md
├── README.md
└── requirements.txt
```

## 관련 노트

[[00-INDEX]] | [[02-Pipeline-StageA]] | [[07-Models]] | [[08-Dataset-Groups]]
