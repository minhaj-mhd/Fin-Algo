---
title: Re-Fix SignalGenerator entry_top_k (regressed by modular refactor)
type: log
status: concluded
updated: 2026-06-15
---

# 💬 Conversation Context: Re-Fix SignalGenerator `entry_top_k`

## 📌 Metadata
- **Start Date**: 2026-06-15
- **Status**: 🔴 Concluded
- **Focus Area**: Vanguard Engine — live signal generation

## 🎯 Objectives
- [x] Fix `TypeError: generate_candidate_signals() got an unexpected keyword argument 'entry_top_k'` (recurrence).
- [x] Determine why a "COMPLETED 2026-06-12" fix regressed.
- [x] Correct the stale archive claim.

## 💻 Active Code Files Modified
- [signal_generation.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/signal_generation.py) — `generate_candidate_signals`

## 📝 Compacted Session Log
- **Symptom**: After the 15-min data fix unblocked the scan, the orchestrator (`orchestrator.py:1652`) crashed passing `entry_top_k=self.risk_manager.entry_top_k` to a method that didn't accept it.
- **Inherited-claim check (AGENTS.md rule 8)**: Completed Work Archive said this was fixed 2026-06-12. It wasn't, at HEAD — the 06-12 fix edited the **old monolithic** `signal_generator.py`; refactor `c6b4c40` rebuilt it as modular `signal_generation.py` and dropped the plumbing. `git log -S"entry_top_k" -- ...signal_generation.py` → never present. Stale "COMPLETED" = regression nobody caught.
- **Design intent (from config + risk_manager + commit 263054b)**: `ENTRY_TOP_K = 5` (config.py:56, "Percentile Gating — 1H Trades") → risk_manager.entry_top_k → "scale-free Top-K rank entry replacing hardcoded head(2)". Commit 263054b removed `min_conviction`/`min_raw_score` gating but left `head(2)` literals.
- **Fix**: Added `entry_top_k=2` param; replaced both `head(2)` (rank-net + raw-directional, per side) with `head(entry_top_k)`. Default 2 = legacy for any caller that omits it; live engine passes 5.
- **Verification**: Synthetic 8-name panel → `entry_top_k=2` yields 4 LONG candidates, `entry_top_k=5` yields 8. Signature now accepts the kwarg → crash resolved.
- **⚠️ Behavioral effect**: With `ENTRY_TOP_K=5` the AI funnel widens 4 → up to 10 candidates/side (pre-gate; downstream conviction/risk/position limits still apply). Tunable via config.py:56, no code change.
- **Corrected**: rewrote the stale archive bullet to record the regression + re-fix.

## 🔗 Core Memory Links & Backlinks
- [[01 — Architecture/Execution & Runtime/Shadow Tracker & Execution Loop]]
