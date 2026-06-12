# 💬 Conversation Context: v10 1-Hour Hold Periods Backtest

## 📌 Metadata
- **Conversation ID**: a9f43092-f192-4743-a2fd-4776f92d5b7e
- **Start Date**: 2026-06-12
- **Status**: 🔴 Concluded
- **Focus Area**: Research & Backtests

## 🎯 Objectives
- [ ] Backtest 1-hour v10 model with a 2-hour hold period.
- [ ] Backtest 1-hour v10 model with a 3-hour hold period.
- [ ] Ensure 15:15 (3:15 PM) is the last exit point of the day.

## 💻 Active Code Files Modified

## 📝 Compacted Session Log
- **Initial Analysis**: Bootstrapped context and setting up ad-hoc backtest script for variable holding periods on the v10 1H model.
- **Script Creation**: Created `scripts/analysis/v10_hold_periods.py` which loads `ranking_data_upstox_1h_v3_3y.csv` and derives `Ret_2h` and `Ret_3h` returns locally from subsequent `Next_Hour_Return` per ticker/day, enforcing EOD cap at 15:15 natively since next-day shift is blocked by date grouping. Enabled GPU inference.
- **Execution & Results**: Ran 9-fold walk-forward validation (2023-08 to 2026-06). **Verdict**: Extending the holding period does not rescue the v10 model. All variations remain deeply net-negative at 10bps cost constraint. 
  - **Long (Top-3)**: 1h (-6.4 bps) -> 2h (-5.8 bps) -> 3h (-6.4 bps). 0/9 folds positive.
  - **Short (Top-3)**: 1h (-7.4 bps) -> 2h (-6.2 bps) -> 3h (-5.6 bps). 0/9 folds positive.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[02 — Models/_Shared/Model Registry & File Structures]]
