# CASDA Stage A — Hybrid Prompt & Multi-Channel Hint Image Pipeline

> **용도:** 이 문서를 바탕으로 figure 를 생성한다.
> **대상 독자:** 논문 리뷰어 — "ControlNet 학습 데이터를 어떻게 설계했는가"를 한눈에 이해시키는 것이 목표.

---

## 1. Stage A 전체 흐름 요약

```
Severstal train_images (1600×256, JPG)
        │  train.csv (RLE 마스크 + ClassId)
        ▼
[Step 1] ROI 추출 (extract_rois.py)
        │  256×256 패치 + roi_metadata.csv
        ▼
[Step 2] ControlNet 데이터 패키징 (prepare_controlnet_data.py)
        ├─ 결함 형상 분석 (DefectCharacterizer)  ──────┐
        ├─ 배경 유형 분류 (BackgroundAnalyzer)          │
        ├─ ROI 적합도 평가 (ROISuitabilityEvaluator)    │
        │                                               ▼
        ├─→ 멀티채널 힌트 이미지 생성 (HintImageGenerator)
        └─→ 하이브리드 텍스트 프롬프트 생성 (PromptGenerator)
                │
                ▼
        controlnet_dataset/
        ├── hints/          ← 3채널 그레이스케일 힌트 이미지 (256×256 PNG)
        ├── train.jsonl     ← {target, hint, prompt, negative_prompt}
        └── packaged_roi_metadata.csv
```

---

## 2. ROI 추출 (Step 1)

| 파라미터 | 값 | 의미 |
|----------|----|------|
| `roi_size` | 256 px | 추출 패치 크기 (ControlNet 입력 해상도와 동일) |
| `grid_size` | 64 px | 슬라이딩 그리드 간격 |
| `min_suitability` | 0.5 | 이 이하 패치는 이후 파이프라인에서 사용 불가 |

**출력:** `roi_patches/` 디렉토리 + `roi_metadata.csv`

---

## 3. 결함 형상 분석 — DefectCharacterizer

> 소스: `src/analysis/defect_characterization.py`

RLE 마스크로부터 4개의 기하학적 지표를 계산. 이 지표는 힌트 채널 생성과 프롬프트 구성에 직접 사용된다.

| 지표 | 계산 방법 | 해석 |
|------|-----------|------|
| **Linearity** | 픽셀 좌표 공분산 행렬의 고유값 비: `(λ₁ - λ₂) / λ₁` | 1.0 → 선형 결함(스크래치), 0.0 → 원형 점 |
| **Solidity** | 결함 면적 / 볼록 껍질 면적 | 1.0 → 속이 꽉 찬 형상, 낮을수록 불규칙 |
| **Extent** | 결함 면적 / 바운딩 박스 면적 | 형상 충전율 |
| **Aspect Ratio** | 바운딩 박스 너비 / 높이 | >5.0 → 매우 가늘고 긴 형상 |

**결함 서브타입 분류 규칙:**

| 서브타입 | 조건 | 설명 |
|----------|------|------|
| `linear_scratch` | Linearity > 0.7 | 고선형 스크래치 |
| `elongated` | Aspect Ratio > 3.0 | 긴 결함 영역 |
| `compact_blob` | Solidity > 0.8 | 속이 찬 점형 결함 |
| `irregular` | 위 조건 불만족 | 불규칙 경계 결함 |
| `general` | 기본값 | 분류 불가 |

---

## 4. 배경 유형 분류 — BackgroundAnalyzer

> 소스: `src/analysis/background_characterization.py`, `scripts/compose_casda_images.py`

Severstal 강재 이미지(1600×256)의 중앙 256×256 패치를 분석해 배경 유형을 5종으로 분류.

| 배경 유형 | 판별 기준 | 강재 표면 설명 |
|-----------|-----------|----------------|
| `smooth` | 분산 < 200, 엣지 밀도 < 0.02 | 균일 무무늬 표면 |
| `vertical_stripe` | Sobel-X 에너지 비율 > 0.65 | 세로 줄무늬 |
| `horizontal_stripe` | Sobel-Y 에너지 비율 > 0.65 | 가로 줄무늬 |
| `textured` | 분산 > 500 또는 엣지 밀도 > 0.05 | 입자성 질감 |
| `complex_pattern` | 엣지 밀도 > 0.15 | 복합 패턴 |

---

## 5. ROI 적합도 평가 — ROISuitabilityEvaluator

> 소스: `src/analysis/roi_suitability.py`

결함 서브타입과 배경 유형의 **매칭 점수(0~1)**를 통해 ControlNet 학습에 적합한 ROI를 선별.

**매칭 규칙 (defect_subtype × background_type):**

| 결함 서브타입 | smooth | vertical_stripe | horizontal_stripe | textured | complex_pattern |
|---------------|--------|-----------------|-------------------|----------|-----------------|
| linear_scratch | 0.7 | **1.0** | **1.0** | 0.5 | 0.3 |
| elongated | 0.8 | **0.9** | **0.9** | 0.6 | 0.4 |
| compact_blob | **1.0** | 0.5 | 0.5 | 0.7 | 0.6 |
| irregular | 0.6 | 0.5 | 0.5 | 0.8 | **1.0** |

**직관:** 선형 결함은 줄무늬 배경에서, 점형 결함은 균일 배경에서 가장 자연스럽다.

---

## 6. 멀티채널 힌트 이미지 — HintImageGenerator

> 소스: `src/preprocessing/hint_generator.py`

ControlNet의 conditioning 입력으로 사용되는 256×256 힌트 이미지.
**핵심 설계 결정: 3채널 컬러 → 단일 그레이스케일로 변환 후 3채널 복제.**

### 6.1 3채널 계산

#### Red Channel — 결함 마스크 (형태 인코딩)

결함 서브타입에 따라 다른 방식으로 마스크를 강조:

| 조건 | 처리 방식 | 강조 대상 |
|------|-----------|-----------|
| Linearity > 0.7 (선형 결함) | Zhang-Suen 스켈레톤화 + Dilation | 중심선 → 255 |
| Solidity > 0.8 (점형 결함) | 마스크 채우기 | 내부 → 200 |
| 일반 | Canny 엣지 + 내부 채우기 | 경계 → 255, 내부 → 100 |

#### Green Channel — 배경 구조선 (엣지 방향성)

배경 유형에 따른 Sobel 필터 방향 선택:

| 배경 유형 | 사용 필터 | 목적 |
|-----------|-----------|------|
| `vertical_stripe` | Sobel-X (수평 기울기) | 세로선 강조 |
| `horizontal_stripe` | Sobel-Y (수직 기울기) | 가로선 강조 |
| `complex_pattern` | √(Sx² + Sy²) | 전방향 엣지 |
| `smooth`, `textured` | √(Sx² + Sy²) × 0.3 | 약한 구조선 |

강도 계수 = `0.5 + 0.5 × stability_score`  
→ stability가 높을수록 구조선이 선명하게 인코딩됨.

#### Blue Channel — 텍스처 밀도 (고주파 성분)

| 배경 유형 | 처리 | 범위 |
|-----------|------|------|
| `smooth` | 고정값 | 20 (낮음) |
| 기타 | 7×7 윈도우 로컬 분산 정규화 | 0~255 |
| `textured`, `complex_pattern` | × 1.2 증폭 | 최대 255 |

### 6.2 그레이스케일 변환 (핵심 설계 결정)

```python
gray = 0.5 × R  +  0.3 × G  +  0.2 × B
hint_image = stack([gray, gray, gray])   # 3채널 동일값 복제
```

**왜 그레이스케일인가?**  
RGB 컬러 힌트를 그대로 사용하면 ControlNet이 **색상 단축 학습(color-shortcut learning)**을 수행—힌트의 빨간 마스크 영역이 생성 이미지에 붉은색으로 그대로 전이되는 현상.  
단일 그레이스케일로 변환함으로써 **강도 정보(결함 위치·형태)만 남기고 색상 정보를 제거**한다.

**가중치 설계 의도:**

| 채널 | 가중치 | 이유 |
|------|--------|------|
| R (결함 마스크) | **0.5** | 결함 위치 특정이 가장 중요 |
| G (배경 구조선) | 0.3 | 배경 텍스처 연속성 보조 |
| B (텍스처 밀도) | 0.2 | 고주파 배경 특성 보조 |

### 6.3 힌트 파일 명명 규칙

```
{image_id}_class{class_id}_region{region_id}_hint.png
예: c924ce298.jpg_class4_region0_hint.png
```

---

## 7. 하이브리드 텍스트 프롬프트 — PromptGenerator

> 소스: `src/preprocessing/prompt_generator.py`

**구조:** `[결함 특성] + [배경 유형] + [표면 상태]`

### 7.1 3가지 프롬프트 스타일

#### Simple (기본 구조만)
```
"a compact blob defect on smooth metal surface, class 3"
```

#### Detailed (기본 + 텍스처 + 상태 형용사) — **실제 사용 스타일**
```
"a solid compact defect spot on smooth metal surface 
 with uniform texture (pristine condition), steel defect class 3"
```

#### Technical (정량적 지표 포함)
```
"Industrial steel defect: solid defect (class 3) 
 on smooth metal surface, no visible pattern, 
 background stability 0.82, match quality 0.91"
```

### 7.2 결함 특성 어휘 사전

| 서브타입 | base 설명 | detailed 설명 |
|----------|-----------|---------------|
| `linear_scratch` | a linear scratch defect | a high-linearity elongated scratch |
| `compact_blob` | a compact blob defect | a solid compact defect spot |
| `elongated` | an elongated defect | a moderately elongated defect region |
| `irregular` | an irregular defect | an irregular defect with complex boundaries |

### 7.3 표면 상태 수식어 (stability_score 기반)

| stability_score | 품질 등급 | 어휘 예시 |
|-----------------|-----------|-----------|
| ≥ 0.8 | high | pristine, well-maintained, clean |
| 0.5 ~ 0.8 | medium | standard, typical, normal |
| < 0.5 | low | worn, weathered, aged |

### 7.4 Negative Prompt (고정)
```
"blurry, low quality, artifacts, noise, 
 distorted, warped, unrealistic, 
 oversaturated, cartoon, painting, 
 text, watermark, logo"
```

---

## 8. train.jsonl 구조

ControlNet 학습 코드(`train_controlnet.py`)가 직접 읽는 형식:

```json
{
  "source":          "hints/c924ce298.jpg_class4_region0_hint.png",
  "target":          "roi_patches/c924ce298.jpg_class4_region0.png",
  "prompt":          "a solid compact defect spot on complex patterned metal surface with multi-directional texture, steel defect class 4",
  "hint":            "hints/c924ce298.jpg_class4_region0_hint.png",
  "negative_prompt": "blurry, low quality, artifacts, noise, ..."
}
```

| 필드 | 학습 내 역할 | 비고 |
|------|-------------|------|
| `target` | VAE 인코딩 → 노이즈 예측 대상 (학습 정답) | 결함 포함 ROI 패치 |
| `hint` | ControlNet conditioning 입력 (`conditioning_pixel_values`) | 멀티채널 그레이스케일 힌트 |
| `prompt` | 텍스트 인코더 입력 (크로스 어텐션) | 하이브리드 프롬프트 |
| `source` | 미사용 (레거시 호환용) | target과 동일값 |

---

## 9. 샘플링 및 필터링 전략

### 9.1 Edge Proximity 필터

결함이 ROI 경계에 너무 가까운 샘플 제거:

| 마진 유형 | 기본값 | 예외 |
|-----------|--------|------|
| 좌우(x) | 10% of ROI width | Class 4: 5% |
| 상하(y) | 5% of ROI height | Class 4: **0%** (이미지 전체 높이를 차지하는 것이 정상) |
| 희소 클래스 (≤200개) | 2% | 학습 데이터 최소 확보 보장 |

### 9.2 품질 필터

| 기준 | 임계값 | 이유 |
|------|--------|------|
| 결함 면적 | ≥ 100 px | 너무 작으면 힌트가 불명확 |
| stability_score | ≥ 0.3 | 마스크 품질 불안정 제외 |
| matching_score | ≥ 0.5 | hint-target 불일치 방지 |
| recommendation | suitable / acceptable | ROI 적합도 평가 통과 |

### 9.3 Class-Aware Capping (v5.2)

```
--per_class_cap 1200 --rare_class_threshold 200
```

- 희소 클래스 (≤ 200개): **전수 포함**
- 풍부한 클래스: 최대 1200개, 다양성 샘플링 (defect_subtype × background_type 조합 균형)
- 다양성 선택 알고리즘: 라운드로빈 방식으로 각 `(subtype, bg_type)` 그룹에서 suitability_score 상위를 순서대로 선택

---

## 10. Figure 생성을 위한 핵심 다이어그램 포인트

아래 4가지가 figure 에서 시각화할 핵심 내용이다.

### (A) 멀티채널 힌트 구성 다이어그램
```
원본 ROI 이미지 (256×256)
    ├── Red: 결함 마스크 (형태 인식)
    │       ├─ [linear] 스켈레톤 → 255
    │       ├─ [blob]   채우기 → 200
    │       └─ [일반]   엣지+채우기 혼합
    ├── Green: 배경 구조선 (방향 인식)
    │       └─ Sobel-X/Y/√ 선택 (배경 유형별)
    └── Blue: 텍스처 밀도 (고주파 인식)
            └─ 로컬 분산 정규화
    │
    ▼ 가중 합산 (0.5R + 0.3G + 0.2B) → 그레이스케일
    │
    ▼ [gray, gray, gray] → 힌트 이미지 저장
```

### (B) 하이브리드 프롬프트 구조
```
[결함 서브타입 특성]  +  [배경 유형]  +  [표면 상태]
"a solid compact         on smooth          with uniform texture
  defect spot            metal surface      (pristine condition),
                                            steel defect class 3"
```

### (C) defect-background 매칭 호환 행렬 (히트맵용)
5×4 매트릭스: rows = background_type, cols = defect_subtype, values = matching score

### (D) train.jsonl 데이터 흐름
```
ROI 패치 (target)  ──┐
힌트 이미지 (hint) ──┼──→ train.jsonl ──→ ControlNet 학습
텍스트 프롬프트    ──┘
```
