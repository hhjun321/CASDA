import sys, re, json
sys.stdout.reconfigure(encoding='utf-8')
from bs4 import BeautifulSoup

src = r'D:\project\CASDA\review\english_approve\extracted\raw_sections.json'
with open(src, 'r', encoding='utf-8') as f:
    raw = f.read().strip()

# Playwright saves as JSON string (quoted) — parse outer then inner
inner_str = json.loads(raw)
sections = json.loads(inner_str)

def extract_math_text(data_mathml_encoded):
    """Extract plain text from HTML-encoded MathML in data-mathml attribute."""
    import html as html_mod
    decoded = html_mod.unescape(data_mathml_encoded)
    # Extract all text content from MathML tags (mn, mi, mo, mtext)
    text = re.sub(r'<[^>]+>', '', decoded).strip()
    text = re.sub(r'\s+', '', text)
    return text

def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Replace span.MathJax with number from data-mathml attribute
    for span in soup.find_all('span', class_='MathJax'):
        mathml_encoded = span.get('data-mathml', '')
        if mathml_encoded:
            txt = extract_math_text(mathml_encoded)
            span.replace_with(txt)
        else:
            span.decompose()

    # Remove MathJax_Preview spans
    for span in soup.find_all('span', class_='MathJax_Preview'):
        span.decompose()

    # Remove non-text elements: tables, figures, images, SVG
    for tag in soup.find_all(['script', 'style', 'table', 'figure', 'img', 'svg', 'canvas']):
        tag.decompose()

    parts = []

    # Get text from html-p divs
    for div in soup.find_all(class_='html-p'):
        txt = div.get_text(separator=' ', strip=True)
        txt = re.sub(r'  +', ' ', txt)
        if txt:
            parts.append(txt)

    # Fallback: li items
    if not parts:
        for li in soup.find_all('li'):
            txt = li.get_text(separator=' ', strip=True)
            txt = re.sub(r'  +', ' ', txt)
            if txt:
                parts.append('- ' + txt)

    return parts

md_lines = []
for sec in sections:
    hashes = '#' * sec['depth']
    md_lines.append('\n' + hashes + ' ' + sec['heading'])
    parts = extract_text(sec['html'])
    for p in parts:
        md_lines.append('')
        md_lines.append(p)

content = '\n'.join(md_lines).strip()
content = re.sub(r'\n{3,}', '\n\n', content)

out = r'D:\project\CASDA\review\english_approve\extracted\VMMedSAM_X_DualBranch_Medical_Image_Segmentation.md'
with open(out, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

idx = content.find('52.8')
print('--- 52.8 context ---')
print(content[max(0, idx-80):idx+150])
print(f'\nTotal chars: {len(content)}')
