---
title: "AI Operating Protocol (pointer)"
type: reference
status: active
updated: 2026-06-12
tags: [protocol]
---
# 🤖 AI Operating Protocol

> **The canonical protocol lives in the repo root: [`AGENTS.md`](../../../AGENTS.md).**
> It is the single source of truth (loaded automatically by Claude Code via `CLAUDE.md`).
> This file used to hold a full copy, which drifted out of sync — that copy is removed to
> prevent divergence. **Edit the protocol only in `AGENTS.md`.**

## Quick start for any agent landing here
1. Read **[`AGENTS.md`](../../../AGENTS.md)** (4-phase continuity protocol, metric discipline, engineering discipline).
2. Open **[[00 — Start Here/Welcome|Welcome]]** — the generated hub linking every section's Map of Content.
3. Check **[[06 — Logs/Active Board|Active Board]]** for the current focus and next steps.
4. Check **[[00 — Start Here/Dead-Ends Register|Dead-Ends Register]]** before reviving any research line.
5. Agents that filter programmatically: read `00 — Start Here/INDEX.json` (path/title/type/status/model/verdict per doc).

## Vault conventions (enforced)
- **Front-matter on every doc**: `type` (spec/report/reference/research/log/archive/moc), `status`
  (active/concluded/dead/superseded/archived/wip), optional `model`/`verdict`/`gauntlet_run_id`, `updated`.
- A `verdict` without a `gauntlet_run_id` (or an explicit `⚠️ UNVERIFIED`) is a protocol violation.
- **Welcome, `_MOC.md`, `INDEX.json`, `Dead-Ends Register.md` are GENERATED** by
  `scripts/memory/build_index.py` — never hand-edit them; re-run it after any memory change.
- Filenames: Title Case With Spaces, except dated daily logs (`YYYY-MM-DD`) and `Conv-YYYY-MM-DD-Topic` notes.
