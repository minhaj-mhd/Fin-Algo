"""
WP4 — rewrite Obsidian wikilinks after the restructure, using restructure_mapping.csv.
- Path-prefixed links ([[02. Model Suite/X]]) -> new full path.
- Basename-only links ([[X]]) -> rewritten ONLY if the basename changed (renamed files);
  unchanged basenames are left as-is (they still resolve, and ambiguous ones can't be pathed).
- Preserves |alias and #heading. Leaves external auto-memory links ([[project_*]] etc.) and
  file:/// code links untouched.

Run from repo root on the branch:  python scripts/memory/link_rewrite.py
"""
import os, csv, re

VAULT = 'finalgo-memory-layer/finalgo'
MAP = f'{VAULT}/06 — Logs/restructure_mapping.csv'   # CSV moved here in WP3

def read_text(p):
    for enc in ('utf-8', 'utf-8-sig', 'utf-16'):
        try:
            return open(p, encoding=enc).read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return open(p, encoding='latin-1').read()

def noext(p):
    return p[:-3] if p.endswith('.md') else p

rel_map = {}        # old_rel_noext -> new_rel_noext
base_changed = {}   # old_base_noext -> new_rel_noext  (unique old basenames only)
base_counts = {}
rows = list(csv.DictReader(open(MAP, encoding='utf-8')))
for r in rows:
    if r['new_path'] == '__DELETE__':
        continue
    o, n = noext(r['old_path']), noext(r['new_path'])
    rel_map[o] = n
    ob = o.split('/')[-1]
    base_counts[ob] = base_counts.get(ob, 0) + 1
for r in rows:
    if r['new_path'] == '__DELETE__':
        continue
    o, n = noext(r['old_path']), noext(r['new_path'])
    ob, nb = o.split('/')[-1], n.split('/')[-1]
    if ob != nb and base_counts[ob] == 1:
        base_changed[ob] = n
# special: legacy [[agent]] basename
base_changed.setdefault('agent', '00 — Start Here/AI Operating Protocol')

WL = re.compile(r'\[\[([^\]]+)\]\]')

def rewrite_target(inner):
    # inner = target[#heading][|alias]
    target = inner
    suffix = ''
    # split alias
    if '|' in target:
        target, alias = target.split('|', 1)
        suffix = '|' + alias
    head = ''
    if '#' in target:
        target, h = target.split('#', 1)
        head = '#' + h
    t = target.strip().rstrip('/')
    key = noext(t)
    new = None
    if key in rel_map:
        new = rel_map[key]
    elif '/' not in key and key in base_changed:
        new = base_changed[key]
    if new is None:
        return None
    return f'{new}{head}{suffix}'

changed_files = 0
changed_links = 0
for root, _, files in os.walk(VAULT):
    if '.obsidian' in root:
        continue
    for f in files:
        if not f.endswith('.md'):
            continue
        p = os.path.join(root, f)
        txt = read_text(p)
        n_local = [0]
        def repl(m):
            inner = m.group(1)
            new_inner = rewrite_target(inner)
            if new_inner is None or new_inner == inner:
                return m.group(0)
            n_local[0] += 1
            return f'[[{new_inner}]]'
        new_txt = WL.sub(repl, txt)
        if n_local[0]:
            open(p, 'w', encoding='utf-8').write(new_txt)
            changed_files += 1
            changed_links += n_local[0]

print(f"rewrote {changed_links} links across {changed_files} files")
print(f"rel_map entries={len(rel_map)}  base_changed entries={len(base_changed)}")
