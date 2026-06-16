# 13 — 미래 연구 가이드라인

> **문서 목적:** CASDA 논문 투고 이후 후속 연구를 위한 실험 로드맵 및 이론적 배경 정리.
> 요약본이 아닌, 실제로 실험을 설계하고 코드를 짤 때 참조할 수 있는 구체적 가이드.

---

## 목차

1. [현재 연구의 한계 인벤토리](#1-현재-연구의-한계-인벤토리)
2. [미래 실험 로드맵](#2-미래-실험-로드맵)
   - 2.1 통계적 신뢰도 강화
   - 2.2 비교 베이스라인 확장
   - 2.3 임계값 민감도 분석
   - 2.4 도메인 일반화 검증
   - 2.5 아키텍처 의존성 분석
   - 2.6 인간 지각 평가
   - 2.7 생성 모델 변형 실험
3. [이론적 배경 정리](#3-이론적-배경-정리)
   - 3.1 확산 모델 (Diffusion Models)
   - 3.2 ControlNet 조건부 생성
   - 3.3 Poisson Image Editing
   - 3.4 클래스 불균형과 데이터 증강
   - 3.5 산업용 결함 탐지
4. [연구 확장 방향](#4-연구-확장-방향)
5. [참조 논문 목록](#5-참조-논문-목록)

---

## 1. 현재 연구의 한계 인벤토리

리뷰어 피드백과 자체 분석에서 도출된 한계. 각 항목에 우선순위(H/M/L)와 투고 전 해소 여부를 표기.

| ID | 한계 | 우선순위 | 현재 상태 |
|----|------|---------|----------|
| L1 | 실험 seed 2개 → 통계 검정력 부족 | H | 미해소, future work로 언급 |
| L2 | 임계값 민감도 분석 없음 | H | 미해소, future work로 언급 |
| L3 | 비교 베이스라인: Raw + CopyPaste만 | H | 부분 해소 (Vanilla SD ablation 추가) |
| L4 | 인간 지각 평가 없음 | M | 미해소, future work로 언급 |
| L5 | 단일 데이터셋 (Severstal만) | M | 미해소 |
| L6 | 아키텍처 의존성 미설명 | M | 부분 해소 (헤징 언어 추가) |
| L7 | BGD 지표: 표본 500개 → 분포 편향 가능 | L | 미해소 |
| L8 | ControlNet 학습 하이퍼파라미터 선택 근거 약함 | L | 미해소 |

---

## 2. 미래 실험 로드맵

### 2.1 통계적 신뢰도 강화 (L1 해소)

**목표:** seed 2개의 분산 추정 불안정 문제 해결

**실험 설계:**

```
Seeds: {42, 123, 456, 789, 2024}  — 5개 이상 권장
각 seed별 전체 파이프라인 재실행 (Stage B 생성 제외, Stage D만)
→ 기존 생성 이미지 재사용 가능 (생성은 seed 42 고정)
```

**통계 검정 방법:**

| 검정 | 용도 | 적용 조건 |
|------|------|----------|
| Wilcoxon signed-rank test | CASDA vs CopyPaste 비교 | non-parametric, n≥5 |
| Bootstrap CI (n=10000) | 95% 신뢰구간 추정 | 모든 지표 |
| Cohen's d | 효과 크기 정량화 | 이미 BGD에 적용됨 — 탐지 지표로 확장 |
| Friedman test | 다중 방법 비교 (Raw, Trad, CopyPaste, CASDA) | 반복 측정 |

**코드 스니펫 (scipy):**
```python
from scipy import stats
import numpy as np

# Wilcoxon signed-rank test
casda_scores = np.array([seed42_map, seed123_map, seed456_map, seed789_map, seed2024_map])
copy_scores  = np.array([...])
stat, p_val = stats.wilcoxon(casda_scores, copy_scores, alternative='greater')

# Bootstrap CI
def bootstrap_ci(data, n_boot=10000, ci=0.95):
    boots = np.random.choice(data, (n_boot, len(data)), replace=True).mean(axis=1)
    lo = np.percentile(boots, (1-ci)/2 * 100)
    hi = np.percentile(boots, (1+ci/2) * 100)
    return lo, hi

# Cohen's d
def cohens_d(a, b):
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std
```

**보고 형식:**
```
CASDA: mAP@0.5 = 0.XXX ± 0.XXX (95% CI: [0.XXX, 0.XXX])
vs CopyPaste: Wilcoxon p = 0.XXX, Cohen's d = X.XX
```

---

### 2.2 비교 베이스라인 확장 (L3 해소)

**목표:** 생성적 증강 방법들과의 공정한 비교

#### 2.2.1 추가해야 할 베이스라인

| 방법 | 유형 | 구현 난이도 | 우선순위 |
|------|------|------------|---------|
| SD v1.5 Inpainting | Diffusion | 낮음 (Hugging Face) | H |
| AnoGAN / f-AnoGAN | GAN | 중간 | M |
| DefectGAN | GAN (결함 특화) | 중간 | H |
| DALL-E 2 / DALL-E 3 | Commercial Diffusion | 낮음 (API) | L |
| CycleGAN (도메인 전환) | GAN | 중간 | L |
| ControlNet (다른 conditioning) | Diffusion | 낮음 | H |

#### 2.2.2 SD Inpainting 베이스라인 구현 가이드

```python
from diffusers import StableDiffusionInpaintPipeline
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# 배경 이미지 + 결함 위치 마스크를 입력으로 결함 영역 생성
image = pipe(
    prompt="steel surface with linear scratch defect",
    image=background_roi,      # 512×512 clean background
    mask_image=defect_mask,    # binary mask of defect region
    guidance_scale=7.5,
    num_inference_steps=30,
).images[0]
```

**비교 포인트:**
- Inpainting: 결함 위치를 명시적으로 제어하지만 형태 조건화 없음
- CASDA: 3-channel hint로 형태·질감·구조 동시 조건화 → 핵심 차별점

#### 2.2.3 DefectGAN 비교

- 논문: "DefectGAN: Weakly-Supervised Defect Detection using Generative Adversarial Network" (WACV 2021)
- 특징: 결함 영역 마스크 + 클래스 레이블 조건화 GAN
- 한계: 학습 데이터 필요량 많음, 철강 특화 아님
- CASDA와 비교 가치: GAN vs Diffusion의 생성 품질 차이 정량화

---

### 2.3 임계값 민감도 분석 (L2 해소)

**목표:** 분류 임계값, 적합도 점수 가중치, 품질 게이트 임계값의 안정성 검증

#### 2.3.1 대상 임계값

| 임계값 | 현재값 | 역할 |
|--------|--------|------|
| 선형성 λ | 0.85 | Class 1 vs 기타 분류 |
| 솔리디티 σ | 0.90 | 형태 기반 분류 |
| 배경 분산 τ | 14.09 | 배경 적합도 |
| 엣지 밀도 ε | 0.607 | 배경 적합도 |
| 품질 게이트 θ | (suitability score cutoff) | 합성 이미지 필터링 |
| Blending kernel k | 5 (BGD 계산) | 경계 밴드 너비 |

#### 2.3.2 민감도 분석 설계

**방법 1: ±10%, ±20%, ±30% 범위 그리드 탐색**
```python
import numpy as np

lambda_range = np.linspace(0.85 * 0.7, 0.85 * 1.3, 7)  # ±30% grid
# 각 lambda 값에서 분류 결과 재계산 → 최종 mAP 변화 측정
# 결과: lambda vs mAP 곡선 → 민감도 평탄 구간 확인
```

**방법 2: Sobol 민감도 지수 (전역 민감도 분석)**
```python
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze

problem = {
    'num_vars': 5,
    'names': ['lambda', 'sigma', 'tau', 'epsilon', 'theta'],
    'bounds': [[0.70, 1.00], [0.80, 1.00], [10.0, 18.0], [0.50, 0.72], [0.3, 0.7]]
}
param_values = sobol.sample(problem, 512)  # 512 × 5 = 2560 조합
```

**보고 형식 (목표):**
- 1차 Sobol 지수: 각 임계값의 단독 기여도
- 총 Sobol 지수: 상호작용 효과 포함
- mAP가 ±5% 이내로 유지되는 안정 구간 명시

---

### 2.4 도메인 일반화 검증 (L5 해소)

**목표:** CASDA가 철강 결함 도메인을 넘어 일반화 가능한지 검증

#### 2.4.1 대상 데이터셋

| 데이터셋 | 도메인 | 클래스 수 | 특성 |
|---------|--------|---------|------|
| **NEU-DET** | 열연 강판 결함 | 6 | CASDA와 동일 도메인 (다른 결함 유형) |
| **DAGM 2007** | 직물/금속 결함 | 10 | 다른 재질, 클래스 불균형 |
| **MVTec AD** | 다영역 산업 결함 | 15개 카테고리 | 이상 탐지 형식 |
| **Magnetic Tile Defects** | 자석 타일 | 6 | 텍스처 다양성 |

#### 2.4.2 전이 실험 설계

**Cross-dataset 실험:**
```
1. Severstal에서 학습한 ControlNet → NEU-DET 생성에 적용
   → 도메인 전이 가능성 평가 (FID, BGD)

2. NEU-DET 전용 ControlNet 학습 → 탐지 성능 비교
   → 데이터셋별 최적화의 필요성 확인

3. 다중 데이터셋 혼합 학습 → 범용 결함 생성 모델 가능성
```

**평가 지표 추가:**
- Per-dataset mAP 변화 (Raw vs CASDA)
- ControlNet 학습 없이 zero-shot 전이 시 성능 저하율

---

### 2.5 아키텍처 의존성 분석 (L6 해소)

**현재 관찰:** CASDA가 EB-YOLOv8에서는 효과적이나 YOLO-MFD에서는 mAP 저하

**가설 목록:**

| 가설 | 내용 | 검증 방법 |
|------|------|----------|
| H-A | YOLO-MFD의 MEFE(엣지 강조 모듈)가 합성 결함의 Poisson blending 경계를 이미 smooth하게 처리 → 추가 gain 없음 | attention map 시각화 |
| H-B | BiFPN의 양방향 특징 융합이 합성 이미지의 multi-scale 패턴을 더 잘 활용 | feature activation 비교 |
| H-C | 학습 데이터 증가보다 모델 capacity의 효과 | 데이터 양 vs mAP 곡선 |
| H-D | 합성 이미지의 분포가 YOLO-MFD 학습 공간과 불일치 | t-SNE로 feature space 시각화 |

**권장 실험:**
```python
# 1. Grad-CAM으로 합성 이미지에서 모델이 주목하는 영역 시각화
from pytorch_grad_cam import GradCAM
# CASDA 합성 이미지 vs CopyPaste 이미지에서 각 모델의 activation map 비교

# 2. 학습 곡선 분석: 합성 이미지 비율(0%, 25%, 50%, 75%, 100%) vs mAP
# → 최적 혼합 비율이 모델마다 다른지 확인

# 3. Confusion matrix 분석: 클래스별 오탐/미탐 패턴 비교
```

---

### 2.6 인간 지각 평가 (L4 해소)

**목표:** BGD 등 자동 지표만으로 포착 못 하는 지각적 품질 평가

#### 2.6.1 MOS (Mean Opinion Score) 스터디 설계

**평가 대상:**
- CASDA 합성 이미지 100장
- CopyPaste 이미지 100장
- 실제 Severstal 이미지 100장 (기준선)

**평가자:**
- 최소 5명 (컴퓨터 비전 / 결함 탐지 전문가 우선)
- Amazon Mechanical Turk / Prolific 활용 가능 (비전문가 + 전문가 혼합)

**평가 척도:**

```
[리얼리즘 평가]
이 이미지가 실제 철강 표면의 결함처럼 보입니까?
1(전혀 아님) ─── 2 ─── 3(보통) ─── 4 ─── 5(매우 그럼)

[경계 자연스러움]
결함과 배경의 경계가 자연스럽습니까?
1(매우 부자연) ─── 3(보통) ─── 5(매우 자연)

[결함 유형 일치도]
이 결함이 어떤 유형처럼 보입니까? (Class 1/2/3/4/모름)
```

**분석:**
- Krippendorff's α로 평가자 간 신뢰도 측정 (목표: α > 0.6)
- CASDA vs CopyPaste MOS: paired t-test
- 자동 지표(BGD, FID)와 MOS 상관 분석 → 지표 타당성 검증

#### 2.6.2 두 대안 이미지 강제 선택 (2AFC)

```
"다음 두 이미지 중 더 실제 철강 결함처럼 보이는 것을 선택하시오."
[CASDA 이미지] vs [CopyPaste 이미지]

→ 승률로 직접 비교 가능
→ MOS보다 단순해서 평가자 편향 낮음
```

---

### 2.7 생성 모델 변형 실험

**목표:** CASDA의 핵심 가정(ControlNet + Poisson이 최적 조합)을 검증

#### 2.7.1 ControlNet 조건화 채널 변형

현재 CASDA 3-channel hint: R=결함 형태, G=구조(Canny), B=텍스처

| 변형 | 변경 내용 | 검증 질문 |
|------|----------|----------|
| 1-ch Canny | Canny 엣지만 | 멀티채널이 필요한가? |
| 2-ch (형태+Canny) | 텍스처 채널 제거 | 텍스처 채널 기여도? |
| Depth hint | Canny → MiDaS 깊이맵 | 깊이 조건화의 효과? |
| Segmentation hint | 마스크 이진 이미지 | 세그멘테이션 조건화? |

#### 2.7.2 더 강력한 생성 모델 백본

| 백본 | 현재 | 변경 | 기대 효과 |
|------|------|------|----------|
| SD v1.5 → SD v2.1 | ✓ 사용 중 | 업그레이드 | 768px 지원, 품질 향상 |
| SD v1.5 → SDXL | ✓ 사용 중 | 업그레이드 | 1024px, 더 현실적 텍스처 |
| ControlNet → ControlNet-XL | ✓ 사용 중 | 업그레이드 | SDXL 조건화 |
| ControlNet → IP-Adapter | - | 새로 추가 | 참조 이미지 기반 조건화 |

**구현 주의사항:**
```python
# SDXL + ControlNet-XL 예시
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet, torch_dtype=torch.float16
)
# 주의: 512px → 1024px 해상도 변경 → ROI 추출 파라미터 수정 필요
```

#### 2.7.3 Poisson Blending 대안

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| Poisson Blending (현재) | gradient-domain 합성 | 경계 부드러움 | 색상 드리프트 가능 |
| Alpha Blending | 선형 혼합 | 단순 | 경계 부자연 |
| Laplacian Pyramid Blending | 주파수 도메인 합성 | 텍스처 보존 | 구현 복잡 |
| Neural Blending (GP-GAN) | GAN 기반 합성 | 최고 품질 | 추가 학습 필요 |
| Harmonization (iHarmony4) | 색조 조화 | 색상 불일치 해소 | 무거운 모델 |

**권장 비교 실험:**
```
BGD 지표로 Poisson vs Laplacian vs GP-GAN 비교
→ 탐지 성능(mAP)과 경계 품질(BGD)의 trade-off 분석
```

---

## 3. 이론적 배경 정리

### 3.1 확산 모델 (Diffusion Models)

#### 3.1.1 DDPM (Denoising Diffusion Probabilistic Models)

**핵심 수식:**

Forward process (노이즈 추가):
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)

누적 결과:
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)

여기서 ᾱ_t = ∏_{s=1}^{t} (1-β_s)
```

Reverse process (이미지 복원):
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

학습 목표 (simple 목적함수):
L_simple = E_{t,x_0,ε} [||ε - ε_θ(√ᾱ_t · x_0 + √(1-ᾱ_t)·ε, t)||²]
```

**SD v1.5 특이점:**
- 픽셀 공간이 아닌 **잠재 공간(latent space)**에서 확산
- VAE 인코더로 이미지 → 잠재 표현 (8배 다운샘플)
- U-Net이 잠재 공간에서 노이즈 예측
- CFG (Classifier-Free Guidance): 조건부/무조건부 노이즈 예측 선형 보간

```
ε_θ(x_t, c) = ε_θ(x_t, ∅) + w · (ε_θ(x_t, c) - ε_θ(x_t, ∅))
# w = guidance_scale (CASDA: 7.5)
# c = 텍스트 프롬프트 임베딩
```

#### 3.1.2 CASDA에서의 확산 모델 역할

- 학습: 결함 ROI 이미지를 타깃으로 ControlNet fine-tuning
- 추론: 3-channel hint + 텍스트 프롬프트 → 새로운 결함 이미지 생성
- 핵심: 학습 데이터 분포의 다양화 (mode coverage 향상)

---

### 3.2 ControlNet 조건부 생성

#### 3.2.1 아키텍처

```
입력: x (노이즈 잠재), c_t (텍스트), c_f (힌트 이미지)

ControlNet 구조:
1. SD U-Net 인코더를 완전히 복사 (Trainable copy)
2. 힌트 이미지 c_f → zero convolution → SD 인코더 입력
3. ControlNet 출력 → zero convolution → SD U-Net 디코더에 가산

Zero convolution:
- 초기화: 가중치 = 0, 편향 = 0
- 목적: 학습 초기에 SD 원본 가중치 보존 (학습 안정성)
```

**학습 목표:**
```
L = E_{z_0, t, c_t, c_f, ε~N(0,1)} [||ε - ε_θ(z_t, t, c_t, c_f)||²]
```

#### 3.2.2 CASDA의 3-Channel Hint 설계 근거

```
힌트 이미지 H ∈ R^{512×512×3}:
- R채널: 결함 형태 마스크 (binary) → 위치·크기 조건화
- G채널: Canny 엣지맵 → 경계·구조 조건화
- B채널: 원본 ROI 텍스처 (grayscale) → 재질·질감 조건화

설계 의도: 단순 Canny-only conditioning보다 더 풍부한 제어 신호
검증: ablation에서 3-channel 제거 시 mAP -28.2% (표 14)
```

#### 3.2.3 Conditioning Scale 의미

```python
controlnet_conditioning_scale = 0.7  # CASDA 설정값

# 0.0 → ControlNet 무시 (순수 텍스트 생성)
# 1.0 → ControlNet 최대 (힌트 이미지에 강하게 구속)
# 0.7 → 형태 조건화 유지하면서 생성 자유도 확보
```

**민감도 실험 아이디어:**
```
conditioning_scale ∈ {0.3, 0.5, 0.7, 0.9, 1.0}에서 생성 후
→ BGD, FID, 탐지 mAP 변화 측정
```

---

### 3.3 Poisson Image Editing

#### 3.3.1 수학적 기초

**목표:** 소스 패치 f*를 타깃 이미지 f에 합성할 때 경계를 자연스럽게 만들기

Poisson 방정식 (경계값 문제):
```
min ∫∫_Ω |∇f - v|² dA
  subject to f|_∂Ω = f*|_∂Ω

여기서:
- Ω: 합성 영역 (결함 마스크)
- ∂Ω: 영역 경계
- v: 가이던스 벡터 필드 (소스 이미지의 그래디언트)
- f*: 타깃 이미지의 경계값
```

이산화 (픽셀 단위):
```
4f_p - Σ_{q∈N_p∩Ω} f_q = Σ_{q∈N_p∩∂Ω} f*_q + Σ_{q∈N_p} v_{pq}

v_{pq} = g_p - g_q  (소스 이미지 g의 그래디언트)

→ 희소 선형 시스템 Ax = b 풀기 (scipy.sparse 활용)
```

**왜 Poisson이 효과적인가:**
- 색상값 자체가 아닌 **그래디언트(변화율)** 보존
- 경계에서 타깃의 색조로 자연스럽게 전환
- BGD 지표에서 30.5% 개선의 물리적 원인

#### 3.3.2 구현 핵심

```python
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def poisson_blend(source, target, mask):
    """
    source: 합성할 결함 이미지 (생성된 ROI)
    target: 배경 이미지 (깨끗한 철강 표면)
    mask:   결함 영역 마스크 (binary)
    """
    h, w = mask.shape
    n_pixels = h * w
    
    # 픽셀 인덱스 매핑
    pixel_idx = np.arange(n_pixels).reshape(h, w)
    interior = mask > 0
    
    # 희소 행렬 구성 (이웃 픽셀 관계)
    A = lil_matrix((n_pixels, n_pixels))
    b = np.zeros(n_pixels)
    
    for y in range(h):
        for x in range(w):
            p = pixel_idx[y, x]
            if interior[y, x]:
                A[p, p] = 4
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        q = pixel_idx[ny, nx]
                        if interior[ny, nx]:
                            A[p, q] = -1
                        else:
                            b[p] += target[ny, nx]
                        # 그래디언트 가이던스 추가
                        b[p] += source[y, x] - source[ny, nx]
            else:
                A[p, p] = 1
                b[p] = target[y, x]
    
    x = spsolve(A.tocsr(), b)
    return np.clip(x.reshape(h, w), 0, 255).astype(np.uint8)
```

---

### 3.4 클래스 불균형과 데이터 증강

#### 3.4.1 Severstal 데이터셋 불균형 구조

```
Class 1: ~900개   (보통)
Class 2: ~247개   (희귀 — 핵심 문제)
Class 3: ~5,902개 (다수)
Class 4: ~1,059개 (보통)
```

**불균형이 탐지 모델에 미치는 영향:**
- Class 2 AP 저하: 학습 배치에서 희귀 클래스 등장 빈도 낮음
- Loss 불균형: 다수 클래스가 gradient를 지배
- Recall 저하: 희귀 결함 미탐지율 증가 → 산업 현장에서 치명적

#### 3.4.2 데이터 증강 전략 분류

| 전략 | 방법 | CASDA와의 관계 |
|------|------|---------------|
| **기하학적 증강** | flip, rotation, crop | `baseline_trad` 포함 |
| **색상 증강** | brightness, contrast, hue | `baseline_trad` 포함 |
| **Copy-Paste** | 실제 ROI 직접 붙이기 | `copypaste` 그룹 |
| **생성적 증강** | GAN/Diffusion으로 새 이미지 생성 | CASDA 핵심 |
| **혼합 증강** | Mixup, CutMix | 미검증 |
| **합성 데이터** | 렌더링/시뮬레이션 | 미검증 |

#### 3.4.3 Long-tail 분포에서 생성 증강의 이론적 정당성

**직관:** Class 2처럼 희귀한 클래스는 실제 데이터의 mode coverage가 부족
→ 생성 모델이 인접 manifold를 탐색하여 새로운 mode 커버

**수식적 관점:**
```
학습 데이터 분포: P_data(x, y)
생성 증강 목표: 희귀 클래스 y=2에서 P_aug(x|y=2) ≈ P_real(x|y=2)

CASDA의 조건화: P_CASDA(x|y, h) — h=힌트 이미지
→ 힌트로 인해 생성 분포가 실제 분포에 더 근접
```

**실험적 검증 방법:**
- t-SNE / UMAP으로 생성 이미지 vs 실제 이미지 feature space 시각화
- FID per-class 측정 (현재 CASDA에 구현됨)
- Class 2 AP 변화가 전체 mAP 변화보다 큰지 확인

---

### 3.5 산업용 결함 탐지

#### 3.5.1 결함 탐지의 특수성

| 특성 | 일반 객체 탐지와의 차이 |
|------|----------------------|
| 클래스 불균형 | 정상:결함 비율 100:1 이상 흔함 |
| 소규모 결함 | sub-pixel ~ 수 픽셀 크기 (기준: COCO의 평균 객체와 비교) |
| 도메인 특수성 | 철강 텍스처 학습 데이터 극히 제한 |
| 정밀도 요구 | 미탐지(FN)가 오탐지(FP)보다 치명적 |

#### 3.5.2 평가 지표 심화

**mAP@0.5 한계:**
- IoU 0.5 단일 임계값 → 경계 정밀도 무시
- 권장 추가 지표: mAP@0.5:0.95 (COCO 스타일), F1-score

**Precision-Recall trade-off:**
```
산업 검사 맥락:
- High Recall 우선: 결함 미탐지 = 불량품 출하 = 치명적
- Precision은 후처리로 보완 가능 (수작업 재검사)

→ 보고 지표에 Recall@Precision=0.9 추가 고려
```

#### 3.5.3 CASDA Quality Gate의 의미 재해석

현재 ablation 결과:
```
w/o Quality Gate: mAP@0.5 = 0.0941 (+7.8% vs CASDA Full)
                 Precision = 0.368 (대폭 감소)
                 Recall = 0.112 (증가)
```

**해석:** Quality Gate는 단순한 이미지 필터가 아니라
**Precision-Recall 조절 메커니즘** — 산업 현장의 "고정밀 검사" 요건에 맞춤

---

## 4. 연구 확장 방향

### 4.1 단기 후속 연구 (논문 1편 가능)

**제목 후보:** "Sensitivity Analysis and Multi-Seed Validation of CASDA: A Statistical Robustness Study"

**핵심 기여:**
1. 5+ seed 실험 → 통계적으로 유의한 결론
2. 임계값 민감도 분석 (Sobol 지수)
3. SD Inpainting + DefectGAN 베이스라인 추가 비교

**예상 실험 비용 (Colab Pro):**
- Stage D 재실행 × 5 seed: ~10시간 × 5 = 50 GPU-hour
- 추가 베이스라인 학습: ~20 GPU-hour

---

### 4.2 중기 후속 연구 (논문 1편 + 확장)

**제목 후보:** "Generalizing Context-Aware Defect Augmentation Across Industrial Domains"

**핵심 기여:**
1. NEU-DET, MVTec AD로 도메인 확장
2. ControlNet 대신 IP-Adapter 기반 참조 이미지 조건화
3. 인간 지각 평가 (2AFC 스터디)

---

### 4.3 장기 연구 방향 (새로운 프레임워크)

**방향 1: 능동 학습 (Active Learning) + 생성 증강**
- 모델이 가장 불확실한 결함 유형 → 생성 우선순위 결정
- 생성-탐지 반복 루프

**방향 2: 결함 인식 이상 탐지 (Anomaly Detection)**
- CASDA 합성 이미지를 이상 탐지 학습에 활용
- MVTec AD 벤치마크 적용

**방향 3: 비디오 결함 탐지**
- 정적 이미지 → 동영상 프레임 연속성 고려
- 시간축 ControlNet (AnimateDiff + ControlNet)

---

## 5. 참조 논문 목록

### 확산 모델 기초

| 논문 | 저자 | 핵심 기여 | 링크 키워드 |
|------|------|----------|------------|
| DDPM | Ho et al. (2020) | 확산 모델 학습 프레임워크 | "Denoising Diffusion Probabilistic Models" |
| DDIM | Song et al. (2020) | 빠른 샘플링 (non-Markovian) | "Denoising Diffusion Implicit Models" |
| LDM | Rombach et al. (2022) | 잠재 확산 모델 (SD 기반) | "High-Resolution Image Synthesis with Latent Diffusion Models" |
| ControlNet | Zhang et al. (2023) | 조건부 생성 제어 | "Adding Conditional Control to Text-to-Image Diffusion Models" |
| CFG | Ho & Salimans (2022) | 분류기 없는 가이던스 | "Classifier-Free Diffusion Guidance" |

### 데이터 증강

| 논문 | 저자 | 핵심 기여 | 링크 키워드 |
|------|------|----------|------------|
| Copy-Paste | Ghiasi et al. (2021) | 세그멘테이션 증강 | "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" |
| Mixup | Zhang et al. (2018) | 이미지 선형 보간 | "mixup: Beyond Empirical Risk Minimization" |
| CutMix | Yun et al. (2019) | 패치 교환 증강 | "CutMix: Training Strategy that Makes Use of Sample Mixing" |
| DefectGAN | Zhang et al. (2021) | 결함 특화 GAN | "DefectGAN: Weakly-Supervised Defect Detection using Generative Adversarial Network" |

### 이미지 합성

| 논문 | 저자 | 핵심 기여 | 링크 키워드 |
|------|------|----------|------------|
| Poisson Blending | Pérez et al. (2003) | Gradient-domain 합성 | "Poisson Image Editing" |
| GP-GAN | Wu et al. (2019) | GAN 기반 이미지 합성 | "GP-GAN: Towards Realistic High-Resolution Image Blending using Generative Adversarial Networks" |
| iHarmony4 | Cong et al. (2021) | 이미지 조화 | "DoveNet: Deep Image Harmonization via Domain Verification" |

### 산업 결함 탐지

| 논문 | 저자 | 핵심 기여 | 링크 키워드 |
|------|------|----------|------------|
| NEU-DET | He et al. (2020) | 열연 강판 결함 데이터셋 | "An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features" |
| MVTec AD | Bergmann et al. (2019) | 이상 탐지 벤치마크 | "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" |
| YOLO-MFD | (CASDA 내 정의) | 멀티스케일 엣지 강조 | `src/models/yolo_mfd.py` |

### 통계 방법

| 방법 | 참조 | 용도 |
|------|------|------|
| Sobol Sensitivity Analysis | Saltelli et al. (2010) | 전역 민감도 지수 | 
| Wilcoxon Signed-Rank Test | Wilcoxon (1945) | Non-parametric 쌍체 비교 |
| Bootstrap CI | Efron & Tibshirani (1993) | 신뢰구간 추정 |
| Cohen's d | Cohen (1988) | 효과 크기 정량화 |
| Krippendorff's α | Krippendorff (2011) | 평가자 간 신뢰도 |

---

## 관련 노트

[[00-INDEX]] | [[01-Overview]] | [[08-Dataset-Groups]] | [[09-Experiments]]
