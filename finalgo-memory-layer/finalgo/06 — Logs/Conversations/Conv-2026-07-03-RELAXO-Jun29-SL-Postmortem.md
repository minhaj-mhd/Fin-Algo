---
title: "Conv-2026-07-03-RELAXO-Jun29-SL-Postmortem"
type: log
status: concluded
updated: 2026-07-03
tags: [vanguard, postmortem, fade-guard, kronos]
---
# 💬 Conversation Context: RELAXO Jun-29 SL Post-Mortem

## 📌 Metadata
- **Conversation ID**: 77c35461-9a63-48c8-9de5-26bc90e99d2f
- **Start Date**: 2026-07-03
- **Status**: 🔴 Concluded
- **Focus Area**: Vanguard live engine — fade-entry mechanics, SL enforcement, gate stack

## 🎯 Objectives
- [x] Explain the RELAXO.NS SHORT 2026-06-29 12:47 loss (−2.36% net vs 0.68% SL)
- [x] Determine whether the same signal would be blocked by today's (2026-07-03) gate stack

## 📝 Compacted Session Log
- **Trade** `TRADE-RELAXO.NS-SHORT-260629124756`: fade PENDING_LIMIT @396.09 (look-back bar failed direction),
  filled 396.25 at ≈12:48:24 (27 s after signal), STOP_LOSS exit @405.35, net −2.36% (₹−1,580),
  ≈3.4× the ₹≈460 budgeted risk at SL 0.68%.
- **Tape** (Upstox 15m, refetched): 12:45 bar exploded 395.70→408.40 close 403.45 on **272k vol (~8–10× rvol)**;
  13:00 bar high **414.95** (−4.7% adverse). The short was filled into the ignition second(s) of a
  volume-backed breakout; stock was +34% off late-May lows (295→396), tv_sentiment BUY, bulk deal
  Jun-25 @412.95 (4% above market) noted by S2 and passed as neutral.
- **Why the SL slipped 1.6%**: stop is a polled software stop
  ([trade_state.py:102](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/trade_state.py));
  `peak_adverse_pct == exit pnl` proves the first post-fill adverse observation was already −2.30%.
  Post-fill `continue` means next evaluation is a full tracker cycle later
  ([orchestrator.py:1601](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py)) —
  minutes-scale gap during the fastest move. No resting exchange SL. DB `exit_time` = planned expiry
  (fill+1h), never overwritten with the actual exit moment (observability gap).
- **Why no guard fired**: at signal time the look-back bar was quiet (range 0.11%, adverse_pos 0.56, low rvol) →
  THRUST_VETO (2.5%/0.75), FADE_ADVERSE (0.70/rvol 0.5) and FADE_BREAKOUT (52wH 526, price 25% below) all pass.
  Guards key on the bar BEFORE the signal; nothing re-validates at FILL time. (Contrast 06-23 RELAXO short:
  blocked FADE_ADVERSE because that bar showed it.)
- **Kronos replay** (exploratory, single candidate, context = 439 completed 15m bars ≤12:30, logged to scratchpad
  not the live jsonl): **p_up 0.167 → aligned 0.833 → KEEP** at both live thr (1−p_up≥0.70) and backtest keep-70
  (0.4333). Kronos *endorsed* the short right before the +4.7% rip — consistent with the known miscalibration
  ([[04 — Research/Kronos Zero-Shot Veto|Kronos veto research]] verdict: DEAD in backtest).
- **Today's-scenario verdict**: the identical signal **passes every deterministic gate today**. Only new layer
  since Jun-29 is Kronos (uncommitted working-tree change) and (a) the live process logged `mode: shadow` on
  2026-07-03 (env override or stale process — config default is now `enforce`; memory says user wanted enforce)
  and (b) even enforce would KEEP this short. Only the non-deterministic S1 Gemini audit (which historically
  vetoed RELAXO shorts for daily-uptrend/thin-volume) might stop it.
- **Fix candidates (NOT implemented — reported only)**: (1) re-run thrust/live-reversal check at PENDING_LIMIT
  fill time; (2) tighter post-fill polling for fresh fade fills (first ~5 min) or tick-driven SL via the WS feed;
  (3) record actual exit timestamp separately from planned expiry; (4) real/synthetic hard stop at broker.

- **Frequency follow-up (full executed ledger, 123 trades 2026-06-03→07-03)**: SL exits = 28 (23%). Median SL
  overshoot ≈ 0.03pp (polling stop is usually clean); >0.25pp overshoot = 2/28; RELAXO's 1.61pp is the worst by 4×
  (next: NESTLEIND 0.40pp) → the blow-through *magnitude* is a ~1-in-120-trades tail event. BUT the excess loss
  beyond budgeted stops (overshoot+exit costs) = ₹−3,555 ≈ **52% of the entire book's ₹−6,840 net loss** (RELAXO
  alone ≈ ⅓ of the excess). And the mechanism is systemic: fade-limit fills = 84/123 trades (68%), SL-rate 26% vs
  15% for confirmed entries, mean net −0.115%/trade vs +0.049%, total ₹−8,080 vs ₹+1,240; 9 of the 10 worst
  overshoots are fade fills. Small sample / sandbox fills / non-causal split — but consistent with the
  adverse-selection mechanics.

## 💻 Active Code Files Modified
- None (analysis only; Kronos replay script in session scratchpad).

## 🔗 Core Memory Links & Backlinks
- [[06 — Logs/Daily Logs/2026-06-28|Daily Log 2026-06-28]]
- [[06 — Logs/Active Board|Active Board]]
- [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py) (fade guard ~L1262–1332, tracker loop ~L1534–1800)
- [config.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/config.py) (FADE/THRUST/KRONOS params)
- [kronos_veto.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/kronos_veto.py)
