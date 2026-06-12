"""WP6 — split the Active Board: move the completed-items log to a Completed Work Archive,
leave a lean board (Current Focus + Next Steps + Daily Logs). No content lost."""
import re

V = 'finalgo-memory-layer/finalgo'
P = f'{V}/06 — Logs/Active Board.md'
s = open(P, encoding='utf-8').read()
body = s.split('---', 2)[-1]
m = re.search(r'## Active Focus\n(.*?)\n## Next Steps', body, re.S)
completed = m.group(1).strip() if m else ''
rest = body[body.index('## Next Steps'):] if '## Next Steps' in body else ''

archive = f"""---
title: "Completed Work Archive"
type: log
status: concluded
updated: 2026-06-12
tags: [archive]
---
# 🗃️ Completed Work Archive

> Rolling log of completed Active-Board items, moved here to keep [[06 — Logs/Active Board|Active Board]]
> lean. Durable findings also live as reference docs and in the [[00 — Start Here/Dead-Ends Register|Dead-Ends Register]].

## Active Focus (completed)

{completed}
"""
open(f'{V}/06 — Logs/Completed Work Archive.md', 'w', encoding='utf-8', newline='\n').write(archive)

board = f"""---
title: "Active Board"
type: reference
status: active
updated: 2026-06-12
tags: [board]
---
# 🎯 Active Board

> Current focus + next steps only (keep ≤ ~10 live items). Completed items roll into
> [[06 — Logs/Completed Work Archive|Completed Work Archive]]; dead research lines live in
> [[00 — Start Here/Dead-Ends Register|Dead-Ends Register]].

## 🔵 Current Focus

* **Memory layer restructure (IN PROGRESS)** — vault reorganized into the `00–09` taxonomy with
  per-doc front-matter and a generated index (`scripts/memory/build_index.py`). See
  [[06 — Logs/Memory Layer Restructure Plan|Restructure Plan]].
* **15-min conviction-flip exit — open decision**: plumbing fixed, but the 15m model is low-signal;
  decide whether the flip exit earns its keep (all 15m overlays are sub-cost). See
  [[06 — Logs/Conversations/Conv-2026-06-12-Fix-15m-Conviction-Flip-Calc|fix log]].
* **Intraday edge — strategic**: four independent lines (CST, DualRes, sided-transformer, gate) all
  confirm the 1h price/volume ceiling is **information**, not model/loss. Next real lever is
  order-flow / microstructure data, not another model.

{rest}"""
open(P, 'w', encoding='utf-8', newline='\n').write(board)
print(f'archived {len(completed)} chars of completed items; board slimmed')
