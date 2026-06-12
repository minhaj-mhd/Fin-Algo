"""
WP2 — add YAML front-matter to every vault .md that lacks it. Classifies type/status/verdict/model
from path + first-H1 + content keywords, and stamps `updated` from the file's last git commit date.
Idempotent: files that already start with '---' are skipped.

Run from repo root: python scripts/memory/add_frontmatter.py
"""
import os, re, subprocess

VAULT = 'finalgo-memory-layer/finalgo'
VERDICT_RE = re.compile(r'\b(FILTER_GRADE|DEAD|KILLED|sub-cost)\b')

def first_h1(txt):
    for l in txt.splitlines():
        if l.startswith('# '):
            return l[2:].strip()
    return ''

def clean_title(h1, fallback):
    # strip leading emoji/symbols and surrounding backticks
    t = re.sub(r'^[^\w`A-Za-z0-9]+', '', h1).strip().strip('`').strip()
    return t or fallback

def classify(rel, txt, h1):
    low = rel.lower()
    name = rel.split('/')[-1].lower()
    head = txt[:4000]
    # type
    if rel.endswith('Welcome.md') or '_moc' in name:
        typ = 'moc'
    elif '/conversations/' in low or '/daily logs/' in low:
        typ = 'log'
    elif low.startswith('09 ') or '/legacy_archive/' in low:
        typ = 'archive'
    elif any(k in name for k in ['plan', 'roadmap', 'proposal', 'spec', 'preregist', 'architecture', 'framework']):
        typ = 'spec'
    elif any(k in name for k in ['report', 'results', 'audit', 'analysis', 'evaluation', 'verification',
                                 'findings', 'catalog', 'comparison', 'statistics', 'calibration']):
        typ = 'report'
    else:
        typ = 'reference'
    # status
    if low.startswith('09 ') or '/legacy_archive/' in low:
        status = 'archived'
    elif re.search(r'\bKILLED\b|❌|DEAD ON ARRIVAL|PERMANENTLY|LINE CLOSED', head):
        status = 'dead'
    elif re.search(r'\b(superseded|SUPERSEDED|deprecated|obsolete|retired)\b', head):
        status = 'superseded'
    elif '🔴' in head or 'Status**: 🔴' in head or 'Concluded' in head[:1500]:
        status = 'concluded'
    else:
        status = 'active'
    # model (02 — Models subfolder)
    model = ''
    m = re.match(r'02 — Models/([^/]+)/', rel)
    if m and not m.group(1).startswith('_'):
        model = m.group(1)
    # verdict
    verdict = ''
    mv = VERDICT_RE.search(head)
    if mv:
        verdict = mv.group(1)
    return typ, status, model, verdict

def git_date(path):
    r = subprocess.run(['git', 'log', '-1', '--format=%cs', '--', path], capture_output=True, text=True)
    return r.stdout.strip() or '2026-06-12'

def yamlq(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

added = skipped = 0
for root, _, files in os.walk(VAULT):
    if '.obsidian' in root:
        continue
    for f in files:
        if not f.endswith('.md'):
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, VAULT).replace('\\', '/')
        txt = open(p, encoding='utf-8', errors='replace').read()
        if txt.lstrip().startswith('---'):
            skipped += 1
            continue
        h1 = first_h1(txt)
        title = clean_title(h1, f[:-3])
        typ, status, model, verdict = classify(rel, txt, h1)
        fm = ['---', f'title: {yamlq(title)}', f'type: {typ}', f'status: {status}']
        if model:
            fm.append(f'model: {yamlq(model)}')
        if verdict:
            fm.append(f'verdict: {verdict}')
        fm.append(f'updated: {git_date(p)}')
        fm.append('tags: []')
        fm.append('---')
        fm.append('')
        open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(fm) + txt)
        added += 1

print(f'front-matter added={added} skipped(existing)={skipped}')
