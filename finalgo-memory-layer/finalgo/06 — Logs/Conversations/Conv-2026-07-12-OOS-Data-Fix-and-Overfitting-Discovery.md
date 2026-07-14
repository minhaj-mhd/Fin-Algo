---
title: "Conv-2026-07-12-OOS-Data-Fix-and-Overfitting-Discovery"
type: log
status: active
updated: 2026-07-12
---

# 💬 Conversation Context: OOS Data Fix & Overfitting Discovery

## 📌 Metadata
- **Conversation ID**: 38a08ecd-5d90-4032-ae18-2524e430c6da
- **Start Date**: 2026-07-11
- **Status**: 🟢 Active
- **Focus Area**: Research & Strategy Evaluation

## 🎯 Objectives
- [x] Download precise June 4 - July 10 OOS data to test the 1-Slot architecture.
- [x] Fix Nifty data dropout bug where timestamps converted to UTC incorrectly blocked trades.
- [x] Re-run true OOS test with 5x Geometric Compounding on a ₹40K Base.
- [x] Document the discrepancy between the historical edge and the true OOS failure.

## 💻 Active Code Files Modified
- [download_nifty_oos.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/backtests/download_nifty_oos.py)
- [download_oos_jul10.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/backtests/download_oos_jul10.py)
- [oos_jul10_backtest.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/backtests/oos_jul10_backtest.py)

## 📝 Compacted Session Log
- **Initial Analysis**: Downloaded strict OOS period (June 4 - July 10) directly from Upstox API and rebuilt the V20 1H panel.
- **Bug Discovery**: Initial run produced 0 trades in late June/July. Discovered that the original Nifty 15m cache was saved in IST but mislabeled as UTC (`+0000`), whereas the new Upstox API fetch returned true UTC. The timezone parsing mismatch caused the Nifty 2H gate to drop 100% of signals post-June 9.
- **Data Fix**: Wrote `download_nifty_oos.py` to correctly append new UTC data, and rewrote the timezone parsing logic in `oos_jul10_backtest.py` to seamlessly merge the historical mislabeled data with the fresh UTC data into valid IST.
- **True OOS Result**: Ran the backtest successfully. Engine generated 35 trades (16 Shorts, 19 Longs).
  - PnL: -23.65% Total Portfolio Return (-₹9,462).
  - Risk Profile: A massive -34.78% Max Drawdown, heavily driven by the Short side collapsing to a 43.8% win rate (compared to 72% historically).
- **Core Realization (Overfitting)**: The user observed that stacking 7 gates produced a massive DEV/OOS gap (DEV +26, HOLDOUT -39). The constraints (like the Mid-Day lull and Nifty 2H filter) were not extracting a true edge, they were merely memorizing the profitable slices of the DEV set. The underlying V20 structural model has no intrinsic edge.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[04 — Research/V20-15m-Regime-Gate-Sweep]]
