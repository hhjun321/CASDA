# MDPI 논문 텍스트 추출 가이드

## 목적
MDPI 논문 URL에서 텍스트(Abstract, 각 Section)만 추출하여 `.md` 파일로 저장.
이미지, 표, Figure 제외. MathJax 수식 → 일반 텍스트 변환.

---

## 필요 도구
- **Playwright MCP** (Claude Code에 설치됨)
- **Python 3** + `beautifulsoup4` (`pip install beautifulsoup4`)

---

## 작업 흐름

### Step 1. 브라우저로 논문 URL 이동
```
mcp__playwright__browser_navigate(url)
```

### Step 2. JS로 섹션 HTML 추출 → `raw_sections.json` 저장
아래 JS를 `mcp__playwright__browser_evaluate`로 실행, `filename`을 `raw_sections.json` 경로로 지정:

```javascript
() => {
  const skipIds = new Set(['html-abstract', 'main-section', 'html-body']);
  const result = [];

  // Abstract: MDPI 논문에 따라 .art-abstract 또는 section#Abstract 사용
  const artAbs = document.querySelector('.art-abstract');
  if (artAbs) {
    const absClone = artAbs.cloneNode(true);
    const h = absClone.querySelector('strong, b, h1, h2, h3, h4');
    if (h && h.innerText.trim() === 'Abstract') h.remove();
    result.push({ depth: 1, heading: 'Abstract', html: absClone.innerHTML });
  }

  document.querySelectorAll('section[id]').forEach(sec => {
    if (skipIds.has(sec.id)) return;
    if (sec.id === 'Abstract') return; // .art-abstract로 이미 처리

    let depth = 1;
    let parent = sec.parentElement;
    while (parent) {
      if (parent.tagName === 'SECTION' && parent.id && !skipIds.has(parent.id)) depth++;
      parent = parent.parentElement;
    }

    const h = sec.querySelector(':scope > h1, :scope > h2, :scope > h3, :scope > h4');
    const heading = h ? h.innerText.trim() : sec.id;

    const contentParts = [];
    for (const child of sec.children) {
      if (child.tagName === 'SECTION') continue;
      if (/^H[1-4]$/.test(child.tagName)) continue;
      // 이미지, 표, Figure 제외
      if (['TABLE', 'FIGURE', 'IMG', 'SVG', 'CANVAS'].includes(child.tagName)) continue;
      if (child.classList.contains('html-fig') || child.classList.contains('html-table-wrap')) continue;
      contentParts.push(child.outerHTML);
    }

    result.push({ depth, heading, html: contentParts.join('\n') });
  });

  return JSON.stringify(result);
}
```

**저장 경로:** `D:\project\CASDA\review\english_approve\extracted\raw_sections.json`

### Step 3. Python 파싱 → `.md` 저장

`parse_html.py`의 `out` 경로를 논문 제목에 맞게 변경 후 실행:

```bash
python D:\project\CASDA\review\english_approve\extracted\parse_html.py
```

`parse_html.py` 주요 처리:
- `raw_sections.json` 읽기 (Playwright가 JSON string으로 저장 → 이중 `json.loads` 필요)
- `span.MathJax`의 `data-mathml` 속성에서 수식 → 텍스트 치환 (중복 방지)
- `script`, `style`, `table`, `figure`, `img`, `svg` 제거
- `div.html-p` (MDPI 본문 단락 클래스) 기준으로 텍스트 추출
- `#Section Name` 형식 마크다운 헤더 생성
- 3줄 이상 공백 → 2줄로 정리

---

## MDPI 구조 특이사항

| 구조 | 설명 |
|------|------|
| `section#html-abstract` | 전체 article 컨테이너 (skip) |
| `section#Abstract` | 일부 논문의 Abstract 섹션 |
| `.art-abstract` | 다른 논문의 Abstract div (sidebar 영역) |
| `div.html-p` | MDPI 본문 단락 (`<p>` 태그 아님) |
| `span.MathJax` | 수식 시각 렌더링 (data-mathml 속성에 실제 값) |

---

## 파일명 규칙
`제목_키워드_Short.md` 형식 (공백 → `_`, 특수문자 제거)

---

## 추출 완료 목록

| 파일명 | URL |
|--------|-----|
| `A_Comprehensive_Review_Position_Movement_Visual_Monitoring.md` | /2076-3417/16/9/4497 |
| `Addressing_Resolution_Scaling_YOLO_Brain_Tumor_Detection.md` | /2076-3417/16/9/4320 |
| `Backbone_Feature_Fusion_YOLOv8_Bacterial_Microcolony_Detection.md` | /2076-3417/16/9/4241 |
| `Integration_Computer_Vision_ML_Automated_pH_Prediction.md` | /2076-3417/16/9/4557 |
| `LDA_YOLO_Rotated_Object_Detection_Remote_Sensing.md` | /2076-3417/16/9/4168 |
| `Mislabel_Detection_ChestXray_CoAtNet_Embedding.md` | /2076-3417/16/9/4067 |
| `SmallData_Neural_Computing_RSM_Injection_Molding.md` | /2076-3417/16/9/4288 |
| `Unsupervised_Hierarchical_Marble_Taxonomy_ViT.md` | /2076-3417/16/9/4137 |
| `VMMedSAM_X_DualBranch_Medical_Image_Segmentation.md` | /2076-3417/16/9/4199 |
