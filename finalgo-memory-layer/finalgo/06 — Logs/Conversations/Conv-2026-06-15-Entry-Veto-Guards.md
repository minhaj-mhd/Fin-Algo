---
title: Entry Veto Guards — Thrust, Bias-Contradiction, Non-Marketable Limit
type: log
status: concluded
updated: 2026-06-15
---

# 💬 Conversation Context: Entry Veto Guards (BALKRISIND / SBILIFE loss post-mortem)

## 📌 Metadata
- **Start Date**: 2026-06-15
- **Status**: 🔴 Concluded
- **Focus Area**: Vanguard execution / AI veto layers

## 🎯 Objectives
- [x] Post-mortem the two stop-loss losses in the 2026-06-15 demo session
- [x] Block BALKRISIND without touching the day's other trades
- [x] Catch SBILIFE (second loser) cleanly
- [x] Fix the pending-limit instant-fill bug

## 📝 Compacted Session Log
- **2026-06-15 demo day**: 11 executed trades, net **−₹452.75**. Two stop-losses drove it: BALKRISIND SHORT (−574.16) and SBILIFE SHORT (−314.22). All 11 executed entries went through the *look-back-failed → PENDING_LIMIT* path.
- **Root cause #1 — fading a breakout**: BALKRISIND's look-back 15m bar was a violent rip (range **3.59%**, close pos 0.87). None of S1 (geometric-wall only), S2 (news only), or the look-back guard (re-prices, doesn't cancel) is allowed to block "don't fade a runaway bar."
- **Root cause #2 — instant-fill bug**: for a SHORT the limit was placed at `high − 0.25·range`, which sits *below* market when the bar closed in its top quartile. The fill trigger `price >= entry_price` is then already true → fills at market instantly (no patience). SBILIFE filled at 1761.20, *above* its bar high (1760.1). 11/12 of today's entries were instantly marketable.
- **Root cause #3 — bias ignored**: S2 rated SBILIFE **BULLISH** ("Strong Buy", higher targets) yet passed the SHORT; RULE 1 only hard-vetoes on enumerated catalysts (rating *upgrade*, earnings beat…), not a standing bullish lean.

### Fixes shipped (all verified by ledger replay)
1. **Thrust guard** — `config.THRUST_VETO_RANGE_PCT=2.5`, `THRUST_VETO_POS=0.75`; cancels a look-back-failed entry when range>2.5% **and** close in extreme quartile against the trade. Blocks BALKRISIND only; passes all 9 other executed trades. Base-rate: a >2.5% 15m bar is ~1-in-372, so it fires on outliers only.
2. **Bias-contradiction hard veto** — S2 verdict now blocked when `structural_bias` opposes the side. Full-ledger replay: fires on only 2/78 trades (SBILIFE −314, LAURUSLABS +85), net +₹229. Principled but **n=2** — sanity rule, not a proven edge.
3. **Non-marketable limit clamp** — pending limit now retraces toward the bar extreme (with eps for pos 0/1) so it is strictly non-marketable; 0/12 instantly marketable after the fix. Trigger logic was already correct — only the placement was wrong.

- **Combined effect on 2026-06-15**: day **−452.75 → +435.63** (both stop-losses removed, every winner kept).

### Follow-up (same session): 1h/30m feeding-grid bug
- **Confirmed live train/serve bug.** `v10_native_1h` (+ 30m model) trained on a **09:15-anchored** hourly grid (`data/ranking_data_upstox_1h_v3_3y.csv`: bars at 09:15, 10:15, 11:15, 12:15, 13:15; "a 1h bar == its four 15m bars"). But the live engine resampled with `origin='start_day'` → **09:00-anchored** bars (09:00 holds only 3×15m, every bar shifted +15min). So live ranking ≠ the model trained/Gauntlet-validated (run `20260610T184210Z`).
- **Fix:** anchor both resamples to 09:15 via `origin = index[0].normalize() + 9h15m` ([orchestrator.py:811-820]). Verified: produces 09:15/10:15/… (1h, 4 bars each) and 09:15/09:45/… (30m), robust to a missing open bar. Also realigns session features (Hour, Is_Open_Hour, Time_To_Close).
- The existing 15m path was already correct (raw Upstox 15m, naturally 09:15).

## ⚠️ Open / not done
- **Partial-forming-bar**: mid-hour scans still score the still-forming 09:15-hour bucket. Remaining half of the feeding fix = "completed-hour only" + cache the candidate set across the four intra-hour 15m checks (15m top-10% trigger already exists at [orchestrator.py:1770]).
- `SANDBOX-ERROR` fills on HAL and SBILIFE today — fill-integrity check still pending.
- Thrust threshold (2.5%) tuned on one day; revisit as look-back-candle OHLC now logs on every entry (forward sample accruing). Consider ATR-normalising.
- Pre-existing `ENTRY_TOP_K 5→3` working-tree change left uncommitted (not part of this work).

## 💻 Active Code Files Modified
- [config.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/config.py)
- [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py)
- [ai_veto.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/ai_veto.py)
