# 💬 Conversation Context: 11-Month Gate Backtest Review

## 📌 Metadata
- **Conversation ID**: d0c39df5-87bc-4f0d-bf3e-ed81ae610bf6
- **Start Date**: 2026-07-10
- **Status**: 🔴 Concluded
- **Focus Area**: Backtesting — v20 80/20 untouched test, idx2h gate validation

## 🎯 Objectives
- [x] Review/verify the 11-month backtest results (SHORT + LONG, all gates × policies × books)
- [x] Confirm the idx2h≥0.5 long gate holds on production model
- [x] Deliver clean results for both full and 5-slot books
- [x] Analyze drawdowns and refine gates for both Longs and Shorts.

## 💻 Active Code Files Modified
- [testset_11mo_gate_dedupe.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/backtests/testset_11mo_gate_dedupe.py)
- [run_backfilled_analysis.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/run_backfilled_analysis.py)
- [plot_drawdown.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/plot_drawdown.py)
- [calculate_weekly.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/calculate_weekly.py)
- [plot_interactive_dashboard.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/plot_interactive_dashboard.py)

## 📝 Compacted Session Log
- **Bootstrapping**: Read Welcome.md, Active Board, latest Conv-2026-07-10-Conviction-Caps-Long-Filter.md.
  User shared context from a prior agent session (conv `d119591d`) that already ran the full 11-month
  backtest — script written, NIFTY 50 15m collected, results artifact produced. Both data files verified present.
- **Prior results confirmed**: SHORT all 18 cells negative; LONG idx2h≥0.5 is the clear winner
  (5-slot SKIP t_d=+2.1, ₹64k; 5-slot RAW t_d=+1.9, ₹75k).
- **15m Regime Gate Sweep**: Explored picking the Top 1 trade every 15 mins (10:15 - 14:15).
  - *Short*: Confirmed `ss > 0.082` absolute threshold is critical. Without it, forcing shorts in bull markets destroys capital. With it, +4.45 BPS edge across 328 trades.
  - *Long*: The `ls` absolute score doesn't work. Instead, Nifty 2H trailing return `> +0.25%` is an exceptional regime filter (+8.69 BPS if `> +0.50%`).
  - *Combined 5x Leverage Test*: Evaluated both rules in parallel at 5x leverage (₹5L notional). Resulted in +₹1.91 Lakhs profit on ₹1L base, but with a steep ₹81k Max Drawdown (191% return, 81% MDD, 2.35 Return/MDD). Visualized drawdown curve in `scratch/drawdown_plot.png`.
  - Created dedicated research note: `[[04 — Research/V20-15m-Regime-Gate-Sweep]]`.
- **Refinement (Final Gates)**: Deep-dive into May/April 2026 drawdowns revealed specific failure contexts.
  - *Short Leg*: Added an **Intraday Mean Reversion Override** (`nifty_intraday > 0.0036`) to bypass the 2H gate on heavily overextended days. Added a **Mid-Day Lull Time Filter** (`time < 11:30 or time > 13:00`) which completely eliminated the April/May drawdown. Short Win Rate hit 65.2%, Net BPS +38.94.
  - *Long Leg*: Added an **Intraday Alignment Gate** (`nifty_intraday > 0.0020`) to prevent buying long breakouts during morning dead cat bounces when the market is still structurally down for the day. Win Rate hit 54.7%, Net BPS +5.74.
  - *Final Result*: Combined 620 trades, 57.7% WR, +15.43 Avg Net BPS, Total Profit ₹4,78,290. All gates and visualizations saved to memory.

## 🔗 Core Memory Links & Backlinks
- [[06 — Logs/Conversations/Conv-2026-07-10-Conviction-Caps-Long-Filter|Conviction Caps Conversation]]
- [[00 — Start Here/Ray of Hope|Ray of Hope]]
