---
title: Conv-2026-07-02-Open-Window-Trade-Stacking
type: log
status: active
updated: 2026-07-02
---

# 💬 Conversation Context: Open-Window Trade Stacking (3 trades/day research)

## 📌 Metadata
- **Conversation ID**: 98487abe-006d-43eb-b994-48aa0ad91386
- **Start Date**: 2026-07-02
- **Status**: 🟢 Active
- **Focus Area**: Research — intraday trade stacking at 6bps cost

## 🎯 Objectives
- [ ] User goal: ≥3 good trades/day, avg net ≥0.3%/trade. Prior math says 0.3%×3 is beyond the info ceiling; research the best achievable stack instead.
- [ ] T1: Harden open gap-reversal short — cover@09:30 vs hold-to-close, k sweep, liquid subset (pre-registered next-step from [[project_intraday_overnight_reversal_edge]]).
- [ ] T2: Second-window trade — signal from realized 09:15→09:30 move, trade 09:30→X.
- [ ] T3: Close-window trade — last-hour (14:15→15:15) fade/follow of intraday-so-far return.
- [ ] T4: Time-of-day dispersion profile (model-free) — where does IC≈0.03 buy the most bps?

## 💻 Active Code Files Modified
- (new) [open_window_stack.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/open_window_stack.py)

## 📝 Compacted Session Log
- **Initial Analysis**: At 6bps: open short ≈+14.6bps/trade (t4.5, edge all in 09:15→09:30); daily_macro_v2 +28bps/3d; 1h rolling book ~breakeven. Target 3×30bps/day needs IC≈0.25 — not in OHLCV. Plan: stack open/close-window trades where dispersion is highest.
- All work EXPLORATORY (no Gauntlet, no verdict authority). Costs modeled at 6bps round-trip flat (matches daily_inverse_intraday.py convention). Bars confirmed LEFT-labeled; cache spans 2022-01→2026-06 (987 days, 110-name liq universe).
- **T1 (c2c short, exit sweep)**: cover@09:30 keeps +5.4 of +8.9 net@6 (k=10); neg-control shows hold-to-close leg is ~⅓ market-drift beta, 09:30 leg is pure selection.
- **T2 (09:30 second-window trade) DEAD** (both directions gross-negative). **T3 (14:15→15:15 close-window) DEAD** (gross +3.3 < cost). **T4**: cs-dispersion 09:15 fwd-1h = 102bps vs 42-54 midday → IC 0.03 buys ~5.4bps gross at open, ~2.3 midday — ranker clears 6bps nowhere; window concentration alone insufficient.
- **HEADLINE — T1b gap-fade**: signal = overnight gap (open/prevclose−1). SHORT top-k gap-ups at open: net@6 **+17 to +31bps/trade, t 6-14.5**; LONG bot-k gap-downs **+12 to +17, t 5-10**. Kill-battery ([gap_fade_artifact_checks.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/gap_fade_artifact_checks.py)): survives top-40-ADV-only (+11-21), |gap|≤3% cap (improves), every year 2022-26 positive (09:30-cover most stable), independent daily CSV 2016-26 (+36-70, t 10-21). **BUT C1: entering at 09:30 keeps only +2-4** → ~90% of edge = capturing the open print; C2 pessimistic HLC/3 fill keeps +6-10 single-sided.
- **Paired neutral book** (short top-5 gap-up + long bot-5 gap-down, |gap|≤3%, net@6): AUCTION-fill tier: **+16.5bps/day cover@09:30 (t11.3, annSharpe 5.7, h1 +17.2/h2 +15.7, worst −157)**; +21.9/day hold-to-close (Sharpe 4.6). CONSERV-fill tier: +0.6 to +4.3/day only. **Everything hinges on fill quality in 09:15→09:30** → next step is EXECUTION measurement (pre-open indicative price capture + paper auction orders in Vanguard), not more backtesting.
- Scripts: [open_window_stack.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/open_window_stack.py), [gap_fade_artifact_checks.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/gap_fade_artifact_checks.py); results in `data/research/open_window_stack/`.
- **GOAL 2 — GapFade-Open v1 strategy + realistic-fill backtest** ([gap_fade_strategy_backtest.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/gap_fade_strategy_backtest.py), 5-min cache 2023-01→2026-06, 107 names, 848 days, |gap|≤3%, short top-5 gap-ups + long bot-5 gap-downs, net@6):
  - **E0 auction-fill**: book **+14.1bps/day @09:30 cover (t9.1, Sh4.9, h1+12.7/h2+15.5)**, +19.4/day to close; yearly 2023 +10.3 / 2024 +17.6 / 2025 +14.7 / 2026 +13.0; neg-control −5.5. Cross-cache reproduction of the 15-min-cache result ✅.
  - **E1 delayed market orders: DEAD** — 5-min delay book −3.6/day @09:30; 10/15-min worse. The fade completes within ~0-5 min of the open.
  - **E2 vwap15 working: ≈0** (−1.2 @09:30; +4.1 close t1.9). **E4 limit-at-open: −10..−15/day @72% fills — adverse selection, wrong instrument.**
  - **E3 slippage budget: break-even ~15bps @09:30 / ~20bps later exits**; at 10bps slip still +4-6/day.
  - **VERDICT: only viable execution = NSE pre-open auction participation** (order 09:00-09:07:59 → filled AT the open print), slippage/impact budget ~10bps. Capacity bound = pre-open auction volume (thin; size must stay small vs auction book). 5-min bars cannot resolve a "fire at 09:15:05" market order (truth between E0 +14 and E1 −3.6) — only live paper fills or tick data can.
  - Next: Vanguard pre-open module — capture indicative price 09:06-09:07, fire paper auction basket, log fill-vs-open slippage 2-4 wks → then single pre-registered Gauntlet run if slip ≤ ~10bps.

## 🔗 Core Memory Links & Backlinks
- [[project_intraday_overnight_reversal_edge]] · [[project_v21_clean_rebuild]] · [[project_v20_rolling_1h_result]] · [[project_dualtf_entry_exit_research]]
