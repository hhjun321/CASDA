# Extracted Implementation Details

Sources:
- `D:\project\CASDA\configs\benchmark_experiment.yaml`
- `D:\project\CASDA\src\analysis\defect_characterization.py`
- `D:\project\CASDA\src\analysis\roi_suitability.py`
- `D:\project\CASDA\src\analysis\background_characterization.py`
- `D:\project\CASDA\src\preprocessing\background_library.py`
- `D:\project\CASDA\src\preprocessing\prompt_generator.py`

---

## Benchmark Model Training Settings

| Model | Task | Optimizer | LR | Epochs | Batch | Input Size | LR Scheduler | Backbone |
|-------|------|-----------|-----|--------|-------|------------|--------------|---------|
| YOLO-MFD | Detection | AdamW | 0.001 | 300 | 16 | 640×640 | Cosine | YOLOv8s + MEFE |
| EB-YOLOv8 | Detection | AdamW | 0.001 | 300 | 16 | 640×640 | Cosine | YOLOv8s + BiFPN |
| DeepLabV3+ | Segmentation | AdamW | 0.0001 | 300 | 8 | 256×512 | Poly (power=0.9) | ResNet-101 |

Additional settings (all models):
- Weight decay: 0.0005 (YOLO-MFD, EB-YOLOv8), 0.0001 (DeepLabV3+)
- Warmup epochs: 10 (detection), 5 (segmentation)
- Early stopping patience: 30 (detection), 40 (segmentation)
- AMP (mixed precision): enabled
- Device: CUDA (Google Colab T4 GPU)
- Random seed: 42
- Data split: 70/15/15 (train/val/test), stratified

---

## ControlNet Training Settings

> **Note:** ControlNet training settings are not stored in benchmark_experiment.yaml (which covers detection/segmentation models only). These settings are in Stage B training scripts. Base model is confirmed from 01-Overview.md:

- Pretrained model: Stable Diffusion v1.5 + sd-controlnet-canny
- Hardware: Google Colab T4 GPU
- [Additional settings — lr, epochs, batch, guidance scale, sampling steps — to be extracted from Stage B scripts]

---

## Threshold Values

| Parameter | Value | File | Line | Role |
|-----------|-------|------|------|------|
| Linearity threshold (HIGH_LINEARITY) | 0.85 | `src/analysis/defect_characterization.py` | 214 | Classifies linear_scratch defects |
| Aspect ratio threshold (HIGH_ASPECT_RATIO) | 5.0 | `src/analysis/defect_characterization.py` | 215 | Classifies linear morphology |
| Low solidity threshold (LOW_SOLIDITY) | 0.7 | `src/analysis/defect_characterization.py` | 218 | Compact blob vs. irregular distinction |
| Quality gate | 0.7 | `src/analysis/roi_suitability.py` | 148 | Minimum suitability score for inclusion |
| Suitability weight — matching | 0.5 | `src/analysis/roi_suitability.py` | 142 | Defect-background compatibility |
| Suitability weight — continuity | 0.3 | `src/analysis/roi_suitability.py` | 143 | Background uniformity |
| Suitability weight — stability | 0.2 | `src/analysis/roi_suitability.py` | 144 | Background texture stability |

### Threshold Justification (for paper/rebuttal)

Suitability score weights reflect domain-informed priorities:
- **Matching (0.5):** Defect-background chromatic/structural compatibility is the primary indicator of a physically plausible composite. Mismatched combinations are immediately perceptible to a human inspector.
- **Continuity (0.3):** Background discontinuity (seams, transitions) creates spurious gradients that confuse detection models.
- **Stability (0.2):** Background texture stability is least critical — moderate texture variation does not prevent accurate defect localization.

The quality gate Q ≥ 0.7 was selected based on the resulting dataset composition: Table 7 shows it retains 92.8% of ROIs (high-quality 61.3% + acceptable 31.5%) while excluding only the 7.2% with severe artifacts.

---

## Compatibility Matrix

Source: `src/preprocessing/background_library.py`, lines 50–80

**5 background types × 4 defect types** (compatibility scores 0.2–1.0):

| Defect Type | smooth | vertical_stripe | horizontal_stripe | textured | complex_pattern |
|-------------|--------|-----------------|-------------------|----------|-----------------|
| compact_blob | 1.0 | 0.8 | 0.8 | 0.5 | 0.2 |
| linear_scratch | 0.8 | 1.0 | 1.0 | 0.5 | 0.2 |
| scattered_defects | 1.0 | 0.8 | 0.8 | 0.5 | 0.2 |
| elongated_region | 0.8 | 1.0 | 1.0 | 0.5 | 0.2 |

**Construction method:** Expert-defined scores based on defect-background visibility and physical co-occurrence patterns in the Severstal dataset. Scores represent the probability/suitability of synthesizing a given defect type on a given background type.

---

## Background Categories

Source: `src/analysis/background_characterization.py`, lines 20–26

```python
class BackgroundType(Enum):
    SMOOTH = 'smooth'
    TEXTURED = 'textured'
    VERTICAL_STRIPE = 'vertical_stripe'
    HORIZONTAL_STRIPE = 'horizontal_stripe'
    COMPLEX_PATTERN = 'complex_pattern'
```

Descriptions (from `src/preprocessing/prompt_generator.py`, lines 50–76):

| Type | Surface | Texture | Pattern |
|------|---------|---------|---------|
| smooth | smooth metal surface | uniform texture | no visible pattern |
| textured | textured metal surface | grainy texture | subtle surface texture |
| vertical_stripe | vertical striped metal surface | directional texture | vertical line pattern |
| horizontal_stripe | horizontal striped metal surface | directional texture | horizontal line pattern |
| complex_pattern | complex patterned metal surface | multi-directional texture | complex surface pattern |

---

## Prompt Templates

Source: `src/preprocessing/prompt_generator.py`

Three prompt styles available (`generate_prompt()` routes based on style parameter):

**1. Simple** (`generate_simple_prompt()`, lines 116–143):
```
{defect_desc} on {bg_desc}, class {class_id}
```

**2. Detailed** (`generate_detailed_prompt()`, lines 145–188):
Combines defect description + background description + texture descriptor.

**3. Technical** (`generate_technical_prompt()`, lines 190–251):
```
Industrial steel defect: {char_str} defect (class {class_id}) on {bg_info['surface']},
{bg_info['pattern']}, background stability {stability_score:.2f}, match quality {suitability_score:.2f}
```

Example prompt (linear_scratch, vertical_stripe background, Class 1):
```
Industrial steel defect: linear_scratch defect (class 1) on vertical striped metal surface,
vertical line pattern, background stability 0.82, match quality 0.90
```

---

## Defect Subtypes

Source: `src/analysis/defect_characterization.py`, lines 192–230

5 categories:
1. `linear_scratch` — linearity > 0.85 AND aspect_ratio > 5.0
2. `elongated_region` — high aspect ratio, moderate linearity
3. `compact_blob` — high solidity (≥ 0.7), low aspect ratio
4. `irregular` — low solidity, non-linear
5. `general` — does not meet criteria for above four types
