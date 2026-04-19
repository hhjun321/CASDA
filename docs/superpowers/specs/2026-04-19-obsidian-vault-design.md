# CASDA Obsidian Vault 설계 문서

**날짜:** 2026-04-19  
**목적:** CASDA 프로젝트의 Obsidian vault 문서 구조 설계  
**참조:** README.md, QUICKSTART.md

---

## 목표

1. **Claude 참조용** — 새 대화 시작 시 Claude가 `00-INDEX.md` 하나로 전체 컨텍스트 파악
2. **사용자 위키** — 파이프라인 구조, 스크립트 파라미터, 실험 결과 기록
3. **실험 관리** — 완료된 실험 결과 + 향후 실험 템플릿

---

## vault 위치

```
D:/project/CASDA/CASDA/   ← Obsidian vault root
```

---

## 파일 구조

```
CASDA/
├── 00-INDEX.md                  ← Claude 진입점 / 전체 컨텍스트 맵
├── 01-Overview.md               ← 프로젝트 목적, 아키텍처, 핵심 개념
├── 02-Pipeline-StageA.md        ← 데이터 전처리
├── 03-Pipeline-StageB.md        ← ControlNet 학습 + 생성
├── 04-Pipeline-StageC.md        ← 후처리 + 품질관리
├── 05-Pipeline-StageD.md        ← 평가
├── 06-Scripts-Reference.md      ← 전체 스크립트 입출력 매핑 표
├── 07-Models.md                 ← 3개 모델 상세
├── 08-Dataset-Groups.md         ← Severstal 데이터셋, 7개 실험 그룹
└── 09-Experiments.md            ← 실험 기록 (완료 + 새 실험 템플릿)
```

---

## 네이밍 원칙

- 숫자 접두사(`00-`, `01-` ...) → Obsidian 파일 탐색기 순서 고정
- 영어 파일명 → Claude Glob/Read 접근 시 인코딩 문제 없음
- 노트 내부: 제목·설명은 한국어, 코드·파라미터·경로는 영어

---

## 노트별 설계

### 00-INDEX.md — Claude 진입점

섹션:
- 프로젝트 한 줄 요약
- 실행 환경 (Colab, 경로 변수)
- 파이프라인 전체 흐름
- 노트 맵 (파일 | 내용 | Claude 활용 시점)
- 완료된 실험 요약 및 링크
- 전체 경로 변수 (QUICKSTART Global Variables 기반)

### 01-Overview.md — 프로젝트 개요

섹션:
- Abstract (README 기반)
- Highlights (7개 핵심 특징)
- 전체 파이프라인 다이어그램
- 리포지토리 구조 설명

### 02~05-Pipeline-Stage*.md — 파이프라인 노트 (공통 템플릿)

각 노트 공통 섹션:
```
# Stage X — [단계명]
> Claude 요약: 한 줄 목적 + 핵심 입출력

## 목적
## 입력 / 출력 (경로 변수 포함 표)
## 스크립트 실행 (CLI 명령 블록)
## 주요 파라미터 설명 (표)
## 주의사항 / 알려진 이슈
## 관련 노트 (Obsidian 링크)
```

### 06-Scripts-Reference.md — 스크립트 매핑 표

전체 14개 스크립트를 하나의 표로:
```
| 스크립트 | Stage | Step | 주요 입력 변수 | 주요 출력 변수 | CPU/GPU |
```

### 07-Models.md — 모델 상세

- YOLO-MFD: MEFE 모듈 설명
- EB-YOLOv8: BiFPN 설명
- DeepLabV3+: ASPP 설명
- 모델 비교 표 (Type / Base / Key Enhancement)

### 08-Dataset-Groups.md — 데이터셋 그룹

- Severstal 데이터셋 설명
- 7개 실험 그룹 표 (Group | Description)
- 평가 지표 (Detection / Segmentation / Synthesis / Statistical)

### 09-Experiments.md — 실험 기록

섹션:
- **완료된 실험**: 날짜 / 설정 / 결과 / 관찰 사항
- **새 실험 템플릿** (복사해서 사용):
  ```
  ### [실험명] — YYYY-MM-DD
  - 목적:
  - 설정:
  - 결과:
  - 결론:
  ```

---

## 언어 정책

| 항목 | 언어 |
|------|------|
| 파일명 | 영어 |
| 노트 제목, 섹션 헤더 | 한국어 |
| 설명, 요약 | 한국어 |
| CLI 명령, 경로, 파라미터명 | 영어 |
| 코드 블록 내용 | 영어 |
| 기술 용어 (ControlNet, FID 등) | 영어 유지 |

---

## 구현 범위

총 10개 파일 생성 (vault root 기준):
- `00-INDEX.md` ~ `09-Experiments.md`
- 기존 `환영합니다!.md` 삭제 불필요 (별도 존재해도 무방)
