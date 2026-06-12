"""
WP3 apply: execute the restructure_mapping.csv via `git mv` (history-preserving). Deletes files
mapped to __DELETE__. Idempotent-ish: skips rows where old==new or old is already gone.

Run from repo root on the memory-restructure branch:  python scripts/memory/apply_moves.py
"""
import os, csv, subprocess, sys

VAULT = 'finalgo-memory-layer/finalgo'
MAP = f'{VAULT}/06. Context & Logs/restructure_mapping.csv'

def git(*args):
    return subprocess.run(['git', *args], cwd='.', capture_output=True, text=True)

moved = deleted = skipped = 0
errors = []
with open(MAP, encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        old = f"{VAULT}/{r['old_path']}"
        if r['new_path'] == '__DELETE__':
            res = git('rm', '-q', old)
            if res.returncode == 0:
                deleted += 1
            else:
                errors.append(f"rm {old}: {res.stderr.strip()}")
            continue
        new = f"{VAULT}/{r['new_path']}"
        if old == new:
            skipped += 1
            continue
        if not os.path.exists(old):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(new), exist_ok=True)
        res = git('mv', old, new)
        if res.returncode == 0:
            moved += 1
        else:
            # fall back to plain move + add (handles untracked)
            try:
                os.replace(old, new)
                git('add', new); git('rm', '--cached', '-q', old)
                moved += 1
            except Exception as e:
                errors.append(f"mv {old} -> {new}: {res.stderr.strip()} / {e}")

print(f"moved={moved} deleted={deleted} skipped={skipped} errors={len(errors)}")
for e in errors[:30]:
    print("  ERR", e)
