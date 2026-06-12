"""
WP0 — read-only inventory of the Obsidian memory vault. Produces a CSV cataloguing every .md
file with size, first H1, front-matter presence, and a heuristic type/status/verdict guess to
drive the restructure (does NOT move or modify anything).

Usage: python scripts/memory/build_inventory.py
Output: finalgo-memory-layer/finalgo/06. Context & Logs/restructure_inventory.csv
"""
import os
import re
import csv
import sys

VAULT = 'finalgo-memory-layer/finalgo'
OUT = f'{VAULT}/06. Context & Logs/restructure_inventory.csv'

VERDICT_RE = re.compile(r'\b(FILTER_GRADE|DEAD|KILLED|sub-cost|net-negative|coin-flip)\b', re.I)
RUNID_RE = re.compile(r'\b\d{8}T\d{6}Z-[0-9a-f]{8}\b')
SUPERSEDED_RE = re.compile(r'\b(superseded|deprecated|obsolete|retired|archived|outdated)\b', re.I)
CONCLUDED_RE = re.compile(r'🔴|Concluded', re.I)

def classify(path, head):
    name = os.path.basename(path).lower()
    low = head.lower()
    # type
    if '/conversations/' in path.replace('\\', '/').lower():
        typ = 'log'
    elif '/daily logs/' in path.replace('\\', '/').lower():
        typ = 'log'
    elif '/05. archives/' in path.replace('\\', '/').lower() or '/legacy_archive/' in path.replace('\\', '/').lower():
        typ = 'archive'
    elif any(k in name for k in ['plan', 'roadmap', 'proposal', 'spec', 'preregist']):
        typ = 'spec'
    elif any(k in name for k in ['report', 'results', 'audit', 'analysis', 'evaluation', 'verification', 'findings', 'catalog']):
        typ = 'report'
    elif any(k in low for k in ['# welcome', 'navigation map', 'index']):
        typ = 'moc'
    else:
        typ = 'reference'
    # status
    if '/05. archives/' in path.replace('\\', '/').lower() or '/legacy_archive/' in path.replace('\\', '/').lower():
        status = 'archive'
    elif SUPERSEDED_RE.search(head):
        status = 'superseded'
    elif re.search(r'\bKILLED\b|❌|DEAD ON ARRIVAL', head):
        status = 'dead'
    elif CONCLUDED_RE.search(head):
        status = 'concluded'
    else:
        status = 'active'
    verdict = ''
    m = VERDICT_RE.search(head)
    if m:
        verdict = m.group(1).upper()
    runid = ''
    m = RUNID_RE.search(head)
    if m:
        runid = m.group(0)
    return typ, status, verdict, runid

rows = []
for root, dirs, files in os.walk(VAULT):
    if '.obsidian' in root:
        continue
    for f in sorted(files):
        if not f.endswith('.md'):
            continue
        p = os.path.join(root, f)
        try:
            with open(p, encoding='utf-8') as fh:
                text = fh.read()
        except Exception as e:
            rel = os.path.relpath(p, VAULT).replace('\\', '/')
            rows.append([rel, rel.split('/')[0], 0, 0, '(read error)', 'no', 'reference', 'active', '', ''])
            continue
        lines = text.count('\n') + 1
        has_fm = 'yes' if text.lstrip().startswith('---') else 'no'
        h1 = ''
        for ln in text.splitlines():
            if ln.startswith('# '):
                h1 = ln[2:].strip()
                break
        head = text[:4000]
        typ, status, verdict, runid = classify(p, head)
        rel = os.path.relpath(p, VAULT).replace('\\', '/')
        top = rel.split('/')[0]
        rows.append([rel, top, len(text.encode('utf-8')), lines, h1, has_fm, typ, status, verdict, runid])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['path', 'top_folder', 'bytes', 'lines', 'first_h1', 'has_frontmatter',
                'guess_type', 'guess_status', 'guess_verdict', 'run_id'])
    w.writerows(rows)

# summary to stdout
print(f"inventoried {len(rows)} files -> {OUT}")
from collections import Counter
print("\nby top folder:")
for k, v in sorted(Counter(r[1] for r in rows).items()):
    print(f"  {v:3d}  {k}")
print("\nby guess_status:", dict(Counter(r[7] for r in rows)))
print("with front-matter:", sum(1 for r in rows if r[5] == 'yes'), "/", len(rows))
print("with a verdict keyword:", sum(1 for r in rows if r[8]))
