---
title: "Conv 2026-07-04 — Would the 2×ATR Stop Have Saved the 1×ATR Stop-Outs?"
type: log
status: concluded
updated: 2026-07-04
---

# 💬 Conversation Context: 1×ATR Stop-Out Rebound Replay vs Current 2×ATR Stop

## 📌 Metadata
- **Conversation ID**: c7401c07-d0af-47f2-b93e-7551713020e6
- **Start Date**: 2026-07-04
- **Status**: 🔴 Concluded
- **Focus Area**: Vanguard live ops / stop-loss policy

## 🎯 Objectives
- [x] Pull every Vanguard trade stopped out under the old 1×ATR(15m) stop
- [x] Replay 15-min bars: how many rebounded before touching the current 2×ATR(15m) stop?

## 💻 Method (exploratory — no verdict authority)
- Trades: `status='STOP_LOSS'` in [database_manager.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/database_manager.py) DB `data/vanguard_trades.db` — 33 total; 31 in the ATR era (2026-05-25 → 2026-07-03), 2 pre-ATR (fixed 0.5%, both gap-throughs, excluded).
- Stop mapping (deterministic, incl. clamps): old `sl1 = max(0.25, min(1.20, atr))` (git HEAD of [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py)) → new `sl2 = max(0.50, min(2.00, 2·atr))`; `sl1=0.25→sl2=0.50`, `sl1=1.20→sl2=2.00`, else `sl2=2·sl1`.
- Bars: 15-min IST from `raw_upstox_cache_15min` (≤05-27), `raw_upstox_cache_15min_3y` (≤06-04), `oos_cache_15min_jun2026` (06-05→06-18, UTC), Upstox v3 public REST for 06-19→07-03 (no auth; needs browser User-Agent, ≤1-month span per request).
- Events after the 1×ATR touch bar: first of {2×ATR level touched (bar low/high), rebound to entry (bars strictly after the touch bar)}; horizon A = scheduled `exit_time` (cap 15:20), horizon B = EOD 15:20. Zero same-bar ambiguities. All 31 SL1 touches reproduced (BRIGADE 06-16 required rescaling entry to the cache's corp-action-adjusted basis).
- Script: scratchpad `sl_rebound_replay.py` (session-local, results CSV alongside).

## 📊 Result (31 ATR-era stop-outs)
- **Horizon A (scheduled exit): 21/31 (68%) never touched the 2×ATR level** — 10 fully rebounded to breakeven first, 11 drifted between the two stops; only 10/31 (32%) would have been stopped again.
- Horizon B (held to EOD): 19/31 survive, 16 rebound to breakeven, 12 hit 2×ATR.
- Loss per episode in risk units (risk-parity sizing makes a full stop = −1R in both regimes): actual 1×ATR exits −1.31R avg (slippage included) vs −0.47R avg under the 2×ATR policy at horizon A.
- Caveats: 15-min bar granularity (no intrabar ordering; conservative SL2-first tie-break never triggered), survivor P&L ignores BE-lock/TP/trailing (understates rescue), winners' ₹ halve under the new sizing — this replay covers only the stop-out episodes, not full-policy P&L.

## 🧠 Verdict (⚠️ UNVERIFIED — exploratory replay, no Gauntlet run)
The 1×ATR stop was firing inside hourly noise exactly as the 2026-07-03 widening decision assumed: two-thirds of its stop-outs never reached 2×ATR. Consistent with [[04 — Research/Stop-Loss Research|stop-width research]]: the stop is a disaster brake, not an edge lever.

## 🛠️ Follow-up implemented same session: candle layer split (prescreen + fill-time recheck)
Post-mortem fix #1 from [[06 — Logs/Conversations/Conv-2026-07-03-RELAXO-Jun29-SL-Postmortem|RELAXO post-mortem]], built and unit-tested:
- **Prescreen (3b)**: deterministic completed-bar guards (THRUST/FADE_BREAKOUT/FADE_ADVERSE) now run in the entry loop BEFORE Kronos (GPU) and Gemini (quota); guard math extracted to single-source `_fade_guard_verdict`. A prescreen reject records the identical candle-stage VETOED row (same bar passed via `prescreen_candle=`) and moves to the next candidate (semantic change: previously a post-audit candle veto ended the side's scan cycle).
- **Fill-time recheck**: `_fill_time_veto` at PENDING_LIMIT trigger (and the disabled expiry→market-fill path): vetoes the fill if the trailing 15 min of 1m candles already form a violent adverse thrust (same `THRUST_VETO_RANGE_PCT`/`THRUST_VETO_POS` constants on a rolling window — nothing newly fitted). New reject_reason `LIMIT_FILL_THRUST`, `reject_stage="candle"` (so it feeds Running Guard Value), counter `candle_fill_recheck`, config kill-switch `FILL_RECHECK_ENABLED` (default ON, requires `CANDLE_LAYER_ENABLED`). Deliberately NOT the raw live-1m reversal test at fill time — no magnitude gate; would veto nearly every legitimate fade fill.
- Fail-open on any data problem. Tests: scratchpad `test_split_guards.py` — 5/5 incident-parity checks (BALKRISIND/BRIGADE/VBL/SUNDARMFIN), RELAXO synthetic cascade vetoed at 3.11% trailing range, quiet fills + with-direction moves pass, broker-error fail-open; `tests/test_trade_state.py` 6/6 green. ⚠️ Fill-time recheck catches fills into an ALREADY-running cascade; a fill at the very first tick of an ignition (RELAXO's exact second) may still pass — tick-driven SL remains the residual fix candidate.
- Files: [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py) (`_fade_guard_verdict`, `_fill_time_veto`, prescreen block 3b, PENDING_LIMIT trigger), [config.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/config.py) (`FILL_RECHECK_ENABLED`).

## 🔗 Core Memory Links & Backlinks
- [[06 — Logs/Daily Logs/2026-06-28|Daily Log 2026-06-28]] (v21 era context)
- [[06 — Logs/Conversations/Conv-2026-07-03-RELAXO-Jun29-SL-Postmortem|RELAXO Jun-29 SL Post-Mortem]]
- [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py) `compute_15min_atr` (lines ~712-766, 2×ATR change still uncommitted)
