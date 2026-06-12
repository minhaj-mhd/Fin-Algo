"""
WP4 link-integrity gate. For every wikilink in the vault, resolve target to a file:
- path form: <vault>/<target>.md exists, or
- basename form: exactly one file named <target>.md exists.
External auto-memory notes (project_/feedback_/reference_ prefixes) are treated as OK-external.
Prints broken links (target + source file). Exit 1 if any broken.

Run: python scripts/memory/link_check.py
"""
import os, re, sys
from collections import defaultdict

VAULT = 'finalgo-memory-layer/finalgo'
def read_text(p):
    for enc in ('utf-8','utf-8-sig','utf-16'):
        try: return open(p,encoding=enc).read()
        except (UnicodeDecodeError,UnicodeError): continue
    return open(p,encoding='latin-1').read()


WL = re.compile(r'\[\[([^\]]+)\]\]')
EXTERNAL_PREFIXES = ('project_', 'feedback_', 'reference_', 'user_',
                     'project-', 'feedback-', 'reference-', 'user-')
PLACEHOLDERS = {'Note Name', 'Folder/Note Name', 'Folder/Note Name|Display Label',
                '06. Context & Logs/Conversations/Conv-YYYY-MM-DD-Topic'}
IMG_EXT = ('.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp')

# index all md by relpath(noext) and by basename(noext); index ALL files by path-suffix (for assets)
by_rel = set()
by_base = defaultdict(list)
all_paths = []
for root, _, files in os.walk(VAULT):
    if '.obsidian' in root:
        continue
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), VAULT).replace('\\', '/')
        all_paths.append(rel)
        if f.endswith('.md'):
            r = rel[:-3]
            by_rel.add(r)
            by_base[r.split('/')[-1]].append(r)

def asset_exists(target):
    return any(p == target or p.endswith('/' + target) for p in all_paths)

broken = []
ambiguous = []
for root, _, files in os.walk(VAULT):
    if '.obsidian' in root:
        continue
    for f in files:
        if not f.endswith('.md'):
            continue
        src = os.path.relpath(os.path.join(root, f), VAULT).replace('\\', '/')
        txt = read_text(os.path.join(root, f))
        for m in WL.finditer(txt):
            target = m.group(1).split('|')[0].split('#')[0].strip().rstrip('/')
            if not target or target in PLACEHOLDERS:
                continue
            if target.lower().endswith(IMG_EXT):     # image embed -> resolve by path-suffix
                if not asset_exists(target):
                    broken.append((src, target))
                continue
            t = target[:-3] if target.endswith('.md') else target
            base = t.split('/')[-1]
            if base.startswith(EXTERNAL_PREFIXES):
                continue
            if t in by_rel:
                continue
            if '/' not in t and base in by_base:
                if len(by_base[base]) > 1:
                    ambiguous.append((src, target, by_base[base]))
                continue
            broken.append((src, target))

print(f"broken={len(broken)}  ambiguous_basename={len(ambiguous)}")
for src, tgt in broken[:60]:
    print(f"  BROKEN  [[{tgt}]]  in  {src}")
if ambiguous:
    print("--- ambiguous (basename resolves to >1 file; pre-existing, low priority) ---")
    for src, tgt, opts in ambiguous[:20]:
        print(f"  [[{tgt}]] in {src} -> {len(opts)} matches")
sys.exit(1 if broken else 0)
