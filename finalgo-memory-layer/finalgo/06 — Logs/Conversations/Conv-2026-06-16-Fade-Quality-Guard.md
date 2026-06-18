---
title: Fade-Entry Quality Guard (3 stop-losses post-mortem)
type: log
status: active
updated: 2026-06-16
---

# 💬 Conversation Context: Fade-Entry Quality Guard

## 📌 Metadata
- **Start Date**: 2026-06-16
- **Status**: 🟢 Active
- **Focus Area**: Vanguard execution / entry guards

## 🎯 Objectives
- [x] Analyse today's 3 stop-losses (3 SL in 2h) — we fade breakouts / catch knives.
- [x] Find a filter that blocks exactly those 3 today but keeps every other trade.
- [x] Implement + verify against today's data.
- [ ] Backtest the thresholds over history before trusting (⚠️ fitted to 1 session).

## 🔎 Analysis — today's 6 executed trades (2026-06-16)
All 3 losers went through the **fade path** (look-back 15m candle FAILED direction
confirmation → pending-limit toward the bar extreme = knife-catch). Discriminator is
**relative volume**, not bar shape:

| Trade | Side | Result | adverse close-pos | rvol | dist-52wH |
|---|---|---|---|---|---|
| SKFINDIA | SHORT | ✅ | *confirmed entry (not a fade)* | 0.51 | −4.7% |
| SUNDARMFIN | LONG | ✅ +₹210 | 1.00 (at low!) | **0.25** | −0.9% |
| INFY | SHORT | ✅ +₹506 | 0.56 | 0.95 | −11.3% |
| **VBL** | LONG | ❌ −₹575 | 0.90 | 0.64 | −1.2% |
| **BRIGADE** | SHORT | ❌ −₹548 | 0.72 | **2.42** | **−0.2%** |
| **CHAMBLFERT** | SHORT | ❌ −₹691 | 0.82 | **2.79** | **−0.2%** |

Two volume-gated failure signatures:
1. **Short into a heavy-rvol breakout pinned to a fresh 52-wk high** → BRIGADE, CHAMBLFERT.
2. **Fade a bar closing in its adverse extreme on real volume** → VBL.
Control proving it's volume not shape: **SUNDARMFIN closed at its very low (pos 1.00,
worst possible) but on rvol 0.25 → bounced → won.** Light-volume adverse closes still fade
fine; volume-backed ones keep running. The 2026-06-15 THRUST_VETO only catches VIOLENT
bars (>2.5% range); today's losers were 0.35–0.82% range, invisible to it.

## 💻 Active Code Files Modified
- [config.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/config.py) — `FADE_QUALITY_GUARD` + 4 thresholds.
- [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py) — guard in the fade branch of `start_shadow_trade` (~L1183); pass `rvol`/`dist_52h` from `full_feature_row`.

## 📝 Compacted Session Log
- Guard rule: `(adverse_pos ≥ 0.70 AND rvol ≥ 0.5) OR (SHORT AND dist_52h ≥ −0.005 AND rvol ≥ 1.5)`.
- Only fires on the fade path (look-back failed). Confirmed/immediate entries untouched.
- **Verified** via simulation on today's exact candles+scores: blocks {VBL, BRIGADE, CHAMBLFERT}, keeps {SKFINDIA, SUNDARMFIN, INFY}. Clean 3/3 partition.
- ⚠️ UNVERIFIED on history — thresholds fitted to 5 fade trades from one session; WILL block some future winners. Next: reconstruct historical fade entries and measure hit-rate impact before flipping any confidence.

## 🔗 Core Memory Links & Backlinks
- Predecessor: [[06 — Logs/Conversations/Conv-2026-06-15-Entry-Veto-Guards|THRUST_VETO guard]]
- Mechanic: pending-limit fade entry (cf. BALKRISIND/SBILIFE 2026-06-15 instant-fill-into-breakout).
