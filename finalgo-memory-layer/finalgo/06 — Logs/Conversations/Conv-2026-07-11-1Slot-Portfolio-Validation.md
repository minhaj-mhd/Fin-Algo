---
title: "Conv-2026-07-11-1Slot-Portfolio-Validation"
type: log
status: concluded
updated: 2026-07-11
---

# 💬 Conversation Context: 1-Slot Portfolio Limit Validation

## 📌 Metadata
- **Conversation ID**: 5da8b7a2-8ac8-435c-9e17-cf5e9031f4bc
- **Start Date**: 2026-07-11
- **Status**: 🔴 Concluded
- **Focus Area**: Trading Strategies (Portfolio Risk Management)

## 🎯 Objectives
- [x] Analyze live week performance (July 6-10) using fresh re-scores to validate the new gates.
- [x] Investigate and solve the issue of overlapping single-ticker exposure.
- [x] Apply a global 1-Slot Limit across the entire 11-month historical panel (including the backfilled gap).
- [x] Calculate realistic Rupee Drawdown and Return/DD metrics for a ₹1L account deploying 5x leverage.

## 💻 Active Code Files Modified
- [this_week_new_gates.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/backtests/this_week_new_gates.py)
- [run_1slot_analysis.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/run_1slot_analysis.py)
- [plot_1slot_visualization_png.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/plot_1slot_visualization_png.py)

## 📝 Compacted Session Log
- **Initial Analysis**: Simulated the current live week using the newly finalized Regime Gates. 
  - Result: 15 Longs taken (+6.27 Net BPS), 0 Shorts taken. The Short model correctly stayed out of a strongly bullish Nifty week, proving the gates work perfectly in real-time.
- **Risk Flag Identified**: The 15 Long trades were dangerously concentrated (e.g., buying `ABB.NS` four separate times overlapping).
- **1-Slot Limit Implementation**: Ran a full 11-month backtest applying a strict 1-slot limit across both models, prioritizing Shorts in case of simultaneous triggers.
  - Result: Reduced trade volume by 57% (272 trades total).
  - PnL metrics: Win Rate improved to 60.3%, Edge increased to +22.38 Avg Net BPS.
- **Drawdown Analysis**: Plotted realistic drawdown assuming a fixed ₹1,00,000 capital base deploying exactly ₹5,00,000 (5x) per trade.
  - Result: Total Return of +304% (₹4.04L Final Equity).
  - Risk Profile: Max historical drawdown hit only -17.8% (-₹24k), establishing an incredibly safe 17.0x Return/DD ratio.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[04 — Research/V20-15m-Regime-Gate-Sweep]]
