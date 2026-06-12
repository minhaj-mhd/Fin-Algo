"""
WP3 second pass: move the non-.md files the md-only inventory missed (assets/*.png, legacy_archive
scripts, working CSVs) into the new taxonomy using the same folder-prefix rules. Deletes __pycache__.
Then removes now-empty old folders.

Run from repo root on the memory-restructure branch: python scripts/memory/apply_moves_nonmd.py
"""
import os, subprocess

VAULT = 'finalgo-memory-layer/finalgo'

PREFIX_RULES = [
    ('08. Model Analysis/1-Hour Vanguard Model/', '02 — Models/1H/'),
    ('08. Model Analysis/15-Minute Vanguard Model/', '02 — Models/15m/'),
    ('08. Model Analysis/30-Minute Vanguard Model/', '02 — Models/30m/'),
    ('08. Model Analysis/Gauntlet Reports/', '02 — Models/Gauntlet Reports/'),
    ('08. Model Analysis/Meta-Veto/', '02 — Models/Meta-Veto/'),
    ('08. Model Analysis/', '02 — Models/'),
    ('07. Cluster Research/', '04 — Research/Cluster Research/'),
    ('07. MCP Integrations/', '05 — Operations/MCP/'),
    ('07. Research & Backtests/', '04 — Research/'),
    ('05. Archives/', '09 — Archive/'),
    ('research_1030_strategy/', '04 — Research/1030 Strategy/'),
    ('03. Trading Strategies/', '03 — Strategies/'),
    ('04. Data & Code Map/', '01 — Architecture/Data & Code/'),
    ('06. Context & Logs/', '06 — Logs/'),
    ('01. Core Architecture/', '01 — Architecture/'),
    ('02. Model Suite/', '02 — Models/_Shared/'),
]
OLD_DIRS = ['01. Core Architecture', '02. Model Suite', '03. Trading Strategies', '04. Data & Code Map',
            '05. Archives', '06. Context & Logs', '07. Cluster Research', '07. MCP Integrations',
            '07. Research & Backtests', '08. Model Analysis', 'research_1030_strategy']

def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True)

moved = deleted = 0
errors = []
for d in OLD_DIRS:
    base = f'{VAULT}/{d}'
    if not os.path.isdir(base):
        continue
    for root, _, files in os.walk(base):
        for f in files:
            p = os.path.join(root, f).replace('\\', '/')
            rel = p[len(VAULT) + 1:]
            if '__pycache__' in rel or rel.endswith('.pyc'):
                git('rm', '-q', '--ignore-unmatch', p)
                if os.path.exists(p):
                    os.remove(p)
                deleted += 1
                continue
            new = None
            for op, npref in PREFIX_RULES:
                if rel.startswith(op):
                    new = f'{VAULT}/{npref}{rel[len(op):]}'
                    break
            if new is None:
                errors.append(f'unmapped: {rel}')
                continue
            os.makedirs(os.path.dirname(new), exist_ok=True)
            res = git('mv', p, new)
            if res.returncode == 0:
                moved += 1
            else:
                try:
                    os.replace(p, new); git('add', new)
                    moved += 1
                except Exception as e:
                    errors.append(f'{rel}: {res.stderr.strip()} / {e}')

# remove now-empty old dirs
removed_dirs = 0
for d in OLD_DIRS:
    base = f'{VAULT}/{d}'
    for root, dirs, files in os.walk(base, topdown=False):
        if not os.listdir(root):
            os.rmdir(root); removed_dirs += 1

print(f'non-md moved={moved} pycache_deleted={deleted} empty_dirs_removed={removed_dirs} errors={len(errors)}')
for e in errors[:30]:
    print('  ERR', e)
