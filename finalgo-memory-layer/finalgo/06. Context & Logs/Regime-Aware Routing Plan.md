# 🚦 Regime-Aware Routing Plan (Conditional Gatekeeping)

This document tracks our active architectural plan to modify how the **Daily Macro Gatekeeper** interacts with our execution engine and backtester.

## 🔴 The Problem: The Opportunity Cost of the Gatekeeper
Currently, in **Vanguard V2.3**, every single signal is forced through the Daily Gatekeeper. If a stock is in a daily downtrend, the system categorically refuses to take a LONG trade on it. 
While this is excellent for keeping **Win Rates high** and avoiding bull traps on trend-following breakouts, it comes at a massive cost: we completely miss **Intraday Mean-Reversions**, **Exhaustion Bottoms**, and **Flash Crashes**. Our models are forced to sit on the sidelines during the most lucrative counter-trend bounces because the daily macro trend is opposing it.

## 🟢 The Solution: Regime-Aware Routing
We will introduce a conditional routing layer that assigns a `Regime_Type` to every single strategy in our pipeline.

1.  **TREND Strategies** (e.g., Opening Range Breakout, Momentum Continuation):
    *   **Rule**: Must match the Daily Gatekeeper. If the daily trend opposes the breakout, the trade is rejected.
2.  **REVERSAL Strategies** (e.g., Exhaustion Traps, VWAP Pinch fades):
    *   **Rule**: Bypasses the Daily Gatekeeper entirely. If a stock is massively overextended, we allow a counter-trend short to ride it back to the mean.

## 🛠️ Implementation & Empirical Backtesting Roadmap

Instead of subjectively guessing which strategy requires the gatekeeper, we will mathematically prove it using a **Dual-Pass Comparative Simulator**.

### Step 1: Build the Comparative Simulator
We will build a dedicated script `scripts/strategy_regime_backtest.py`. This simulator will evaluate every strategy (and our pure 1-Hour AI models) under two strict passes:
1.  **GATED Pass**: The Daily Gatekeeper is strictly enforced (simulating our current V2.3 architecture).
2.  **UNGATED Pass**: The Daily Gatekeeper is entirely ignored (simulating a pure tactical intraday engine).

### Step 2: The Empirical Verdict
The simulator has completed its run and outputted the side-by-side performance matrix. You can view the full mathematical breakdown in the **[[02. Model Suite/Empirical Regime Simulation Results|Empirical Regime Simulation Results]]** document.

The core finding is that **Scalping (S4, S5, S3)** absolutely requires the Gatekeeper (TREND tag), but **Morning Breakouts (S8)** perform better without it (REVERSAL tag).

### Step 3: Tag and Route the Live Engine
Once we have the mathematical verdict, we will update `scripts/strategy_filters.py` to assign the correct `regime_type` to every signal. Then, we will update `vanguard_signal_engine.py` to conditionally apply the Daily Gatekeeper only to `TREND` strategies, while applying a high-volatility "Stretch Filter" (like distance from EMA) to `REVERSAL` strategies.

*Status: Awaiting review of the Stretch Filter mechanics before codebase implementation.*
