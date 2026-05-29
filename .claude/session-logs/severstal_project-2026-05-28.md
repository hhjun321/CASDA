# severstal_project — 2026-05-28

## Session `2f8591c7` — 2026-05-28T08:49:03.838Z

### #1 · 17:49:45

> 👤 **User**
>
> D:\project\severstal_project\.claude\.memo\report_개선연구.md 내용을 분석한다

🤖 **Claude**

- 📖 Read `report_개선연구.md`

**CASDA 개선 연구 항목 분석**

6개 약점 — 우선순위별:

**즉시 블로킹 (논문 accept 위험)**
1. **통계적 유의성 부재** (§1): single-run 결과로 0.3~1pp 차이는 reviewer가 reject 근거로 씀. 3 seed × 9 variants = 27 run 필요. 가장 치명적.

2. **Baseline 부족** (§5): CopyPaste만으로는 SCI 기준 미달. DefectGAN / 최소 1개 diffusion baseline 필수. FID 공통 지표로 횡비교.

**중간 위험 (논문 약화)**
3. **품질 지표 확장** (§3): FID만으론 부족. KID (소규모 데이터셋 더 신뢰도 높음) + mask-to-defect IoU 정렬 검증이 CASDA 방법론 주장 강화에 직결.

4. **임계값 근거 불충분** (§4): linearity > 0.85, aspect_ratio > 5.0 등 empirical 기원 미기재. 리뷰어 "왜 이 값인가?" 질문에 답 없음.

**부분 미흡 (보완 가능)**
5. **합성 비율 최적화** (§2): 42.7% 근거 약함. fine-grained sweep + 다중 시드로 통계 결정 필요.

6. **극단 클래스 미검증** (§6): Class 2(247개)는 현재 커버. 50개 미만 시나리오 미테스트 — few-shot conditioning 연구로 보완.

---

**작업 순서 추천**:
- §1 (multi-seed) → §5 (SOTA baseline) → §3 (KID/IoU 지표) → §4 (임계값 문서화) → §2 (ratio sweep) → §6 (few-shot)

§1과 §5가 논문 통과 여부에 직결. 나머지는 contribution 강화.

---

### #2 · 17:51:52

> 👤 **User**
>
> 이 문서 내용을 좀더 자세하게 보완하자. 질의하자

🤖 **Claude**

문서 보완용 질문. 각 항목별로 현재 상태 파악 필요.

---

**§1 통계적 유의성**
- 현재 학습된 모델 checkpoint가 몇 개 있나? (seed 재실험 시 base가 되는 것)
- 27 run 전부 새로 돌릴 여건인가, 아니면 일부만 선택적으로?

**§2 합성 비율 최적화**
- 42.7% 결정 당시 탐색한 ratio 범위가 어떻게 됐나? (예: 20%~60% 10% 간격?)
- variance 추정 없다고 했는데, 당시 단일 시드로만 실험한 게 맞나?

**§3 품질 평가 지표**
- FID는 현재 어느 시점에서 계산? (train set 기준? test set?)
- LPIPS 도입 시 비교 쌍(pair)을 어떻게 구성할 계획인가? (생성 이미지 vs 원본 real 이미지?)
- mask-to-defect IoU — 현재 blending 후 mask 정보가 보존되어 있나?

**§4 임계값 결정 근거**
- linearity > 0.85 등의 값 — 원래 어떻게 정했는지 기억나나? (직관적 설정인가, 분포 보고 정한 건가?)
- 배경 5개 카테고리 경계 — 코드에 하드코딩된 수치인가, 외부 문서에 있나?

**§5 Baseline 비교**
- DefectGAN 등 SOTA 재현 직접 구현할 계획인가, 아니면 논문 reported number만 인용?
- 현재 평가 데이터셋 구성이 어떻게 되나? (test split 크기, class별 분포)

**§6 극단 클래스**
- Class 2 외에 다른 클래스 sample 수는? (전체 분포 확인 필요)
- few-shot conditioning — ControlNet fine-tuning 추가 실험 계획이 있나, 아니면 future work으로 남길 건가?

---

### #3 · 07:43:47

> 👤 **User**
>
> /devnote

🤖 **Claude**

파일 이미 읽음. Branch B 진행.

---

```
파일: .claude/.memo/report_개선연구.md

작업 유형: 연구 코드 개선 (실험 재현성 + 평가 지표 확장 + 분류 임계값 문서화)
사용할 스킬: feature-dev (복수 독립 태스크) 또는 항목별 micro-fix

작업 항목:
1. [§1] Multi-seed 실험 파이프라인 — 3개 random seed로 반복 실행, mean ± std 보고 코드 추가
2. [§2] Ratio sweep 자동화 — fine-grained 비율 탐색 + 다중 시드 반복 스크립트
3. [§3] 평가 지표 확장 — KID, LPIPS, mask-to-defect IoU 계산 코드 추가
4. [§4] 임계값 결정 근거 문서화 — linearity/aspect_ratio/배경 카테고리 임계값 도출 근거 코드 + 문서
5. [§5] SOTA Baseline 비교 — DefectGAN 등 1~2개 baseline 동일 조건 평가 파이프라인
6. [§6] 극단 클래스 시나리오 — Class 2 50개 미만 few-shot conditioning 실험 코드
```

---

진행 전 질의 먼저 (질의하자 요청대로):

**코드 현황 파악 필요**

1. **현재 학습 파이프라인** 진입점 파일이 어디인가? (`train.py`? `controlnet_training/` 하위?)
2. **평가 코드** 위치 — FID 계산 현재 어느 스크립트에 있나?
3. **§4 임계값** — `roi_metadata.csv` 생성 스크립트에 하드코딩되어 있나, 별도 config 파일인가?
4. **§5 Baseline** — DefectGAN 직접 구현 vs 논문 reported number 인용 중 어느 방향인가?
5. 6개 항목 전부 한 번에 할지, 우선순위 추려서 할지?