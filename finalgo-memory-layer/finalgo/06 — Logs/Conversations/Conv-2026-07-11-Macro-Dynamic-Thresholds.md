# 💬 Conversation Context: Macro Dynamic Thresholds Research

## 📌 Metadata
- **Conversation ID**: 942602bc-be26-4372-950e-372d89b01374
- **Start Date**: 2026-07-11
- **Status**: 🔴 Concluded
- **Focus Area**: Research & Strategy Optimization

## 🎯 Objectives
- [x] Research relationship between crude oil price changes and Indian market (Nifty).
- [x] Explore using global indices for dynamic thresholding in v20_rolling_1h.
- [x] Create an implementation plan for dynamically adjusting thresholds based on macro factors.
- [x] Create exploratory scripts and backtest performance.
- [x] Analyze Dynamic Probability (Model Conviction) and Intraday Time gating.
- [x] Integrate Local Nifty regimes into a Multivariate Global x Local matrix.

## 💻 Active Code Files Modified
- [analyze_macro_impact.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/analyze_macro_impact.py)
- [evaluate_dynamic_gates.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/evaluate_dynamic_gates.py)

## 📝 Compacted Session Log
- **Initial Analysis**: Bootstrapping session to investigate replacing hard thresholds in v20_rolling_1h with dynamic thresholds driven by crude oil and global macro indices.
- **Analysis Execution**: Analyzed the relationship and confirmed strong statistical correlation (e.g. Brent gap ups kill Nifty structure). 
- **Backtest Result**: Formulated dynamic equation (`dynamic_2h = 0.0025 + (brent * 0.05) - (sp500 * 0.10)`) and ran OOS verification. **Falsified**: Static Nifty structural threshold (6411 Total BPS) strictly outperformed the dynamic scaler (5820 Total BPS) because the Nifty intraday pricing already perfectly prices in overnight macro, making a delayed scaler redundant.
- **Probability & Time Gate Analysis**: Pivoted to analyze probability mapping and midday block dynamic opening. Data strictly proved that the Midday Block (11:30-13:00) is universally toxic across all regimes and must remain closed.
- **Dynamic Probability Result**: Tightening model conviction when macro factors are against us (e.g. `sp500 > 0` -> `prob = 0.110`) resulted in an elite **72.0% Win Rate** and **45.98 Avg Net BPS**, though it reduced total trade volume from 185 to 100.
- **Multivariate (Global x Local) Refinement**: Mapped SP500 regimes against Nifty 2H regimes. Found that if SP500 is UP but Nifty is WEAK (decoupled), the short edge remains massive (+63.39 BPS). Only tightened the threshold to `0.110` when `SP500 is UP AND Nifty is NOT Weak`. Result: **Hit the Pareto Frontier** with 154 Trades, 70.1% Win Rate, 39.66 Avg Net BPS, recovering total volume to 6107 BPS.
- **Long Model Falsification**: Attempted to apply the Multivariate Matrix to the Long Model probability thresholds. Results definitively proved that the Long model's raw probability predictions are intrinsically toxic/anti-selected; dropping the threshold dynamically during favorable regimes simply dragged in more toxic trades. Concluded that Multivariate dynamic scaling is a Short-Side Only alpha lever.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[06 — Logs/Active Board]]
