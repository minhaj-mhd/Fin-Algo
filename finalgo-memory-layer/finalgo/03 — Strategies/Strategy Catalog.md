---
title: "Strategy Catalog & Execution Pipelines"
type: report
status: active
updated: 2026-06-22
tags: []
---
# 📈 Strategy Catalog & Execution Pipelines

The Vanguard engine generates live signals from a **single execution pipeline: Pipeline 1 (Pure AI Signals)**. While historical strategies specified standard backtested exit anchors, the active production engine (V2.3) overrides standard risk boundaries at entry with **Dynamic Volatility-Adjusted ATR Brackets** (Stop Loss and Take Profit calculated in real time).

> [!WARNING]
> **PIPELINE 2 RETIRED (2026-06-22).** The Structural Strategy pipeline (Strategies S1-S50) and the merge / ensemble-overlap stage have been **removed from the live engine**. `SignalGenerator.generate_candidate_signals` now emits **only Pipeline 1** signals; every signal carries `strategy_id=None`, `is_ensemble=False`, and a `source` of `AI_Net` or `AI_Raw`. `scripts/strategy_filters.py` is no longer wired into [`orchestrator.py`](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py) and is retained only as a historical/backtest reference. The strategy backtests in the table below are **historical only and no longer executed in production.**

---

## 📊 Live Signal Generation Pipeline

The engine runs a single AI scanning process during the active session:

```mermaid
graph TD
    subgraph Pipeline 1: Pure AI Signals (1-Hour Core Ranker)
        A1[Live Candlestick data] -->|Feature Generator| A2[1-Hour features scores_df]
        A3[Daily Macro XGBoost Gatekeeper] -->|Eligible Filters| A4{Eligible Pool}
        A2 --> A4
        A4 -->|v8_upstox_3y hourly ranker| A5[Dual-Model Prediction Scores]
        A5 -->|Conviction: long - short| A6[Net Conviction Ranks]
        A6 -->|Top-K by Long_Rank / Short_Rank| A7[AI Hybrid Net Candidates (AI_Net)]
        A6 -->|Exclude Hybrid Net, Top-K by raw_score| A8[AI Pure Directional Candidates (AI_Raw)]
        A7 --> A9[AI Signals List]
        A8 --> A9
    end

    A9 --> V[HIERARCHICAL DUAL-STAGE AI VETO <br> Triage S1 flash + news CRO audit S2]
```

### 1. Pipeline 1: Pure AI Signals (the only live pipeline)
*   **The Eligible Pool**: Filters the universe to symbols in the daily macro `long_eligible` or `short_eligible` sets, excluding tickers currently in 30-minute entry or veto cooldowns.
*   **AI Hybrid Net Candidates**: Selects the top `entry_top_k` tickers per side, sorted by relative conviction rank (`Long_Rank` / `Short_Rank` ascending). Trigger source is logged as `AI_Net`.
*   **AI Pure Directional Candidates**: Excludes the hybrid net candidates, then selects the top `entry_top_k` remaining tickers per side, sorted by raw direction scores (`long_score` / `short_score` descending). Trigger source is logged as `AI_Raw`.

### 2. ~~Pipeline 2: Structural Strategy Signals~~ — 🔴 RETIRED (2026-06-22)
*Removed from the live engine.* The former rule-filter pipeline (`scripts/strategy_filters.py`, Strategies S1-S50) and its daily-gatekeeper re-validation no longer run in production. The strategy definitions below are kept for historical reference only.

### 3. ~~Pipeline Merging & Ensemble Overlaps~~ — 🔴 RETIRED (2026-06-22)
*Removed from the live engine.* With Pipeline 2 gone there is nothing to merge against; the engine no longer produces `Ensemble_*` sources. All live signals are pure AI.

---

## 🏆 Historical Backtested Strategies Reference (Retired — not executed in production)

All figures below represent historic backtest baselines under a **strict 0.06% round-trip transaction and slippage fee** (calculated on a ₹10 Buy + ₹10 Sell order brokerage, plus Securities Transaction Tax [STT] on sales):

| ID | Strategy Name | Win Rate (WR) | Long WR | Short WR | Total Return | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **8** | **Opening Range Breakout (ORB)** | **64.3%** | 50.0% | 75.0% | **+2.31%** | 🏆 **Best Strategy** |
| **10** | **Quad-Timeframe Unanimous** | **54.2%** | **85.7%** | 41.2% | **+1.86%** | 🏆 **Most Robust** |
| **1** | **Daily Macro Gatekeeper** | 50.9% | 50.0% | 51.7% | **+2.20%** | ✅ Profitable |
| **2** | **Short-Side Specialist** | 58.8% | N/A | 58.8% | **+0.48%** | ✅ Profitable |

### 🥇 Strategy 8: Opening Range Breakout (ORB)
*   **Setup Range**: Tracks morning range boundaries (`OR_High` and `OR_Low`) between 9:15 AM - 9:45 AM.
*   **Triggers (After 10:00 AM)**:
    *   **LONG**: Close of a 15M bar > `OR_High * 1.001` **AND** 15M rank $\le$ 5 **AND** 1H rank $\le$ 10.
    *   **SHORT**: Close of a 15M bar < `OR_Low * 0.999` **AND** 15M short_rank $\le$ 5 **AND** 1H short_rank $\le$ 10.
*   **Holistic Backtest Context**: During the `10x_backtest` simulations, Strategy 8 was **strictly gated** by the Daily Macro Gatekeeper. The simulator ran the daily scan at 9:15 AM to determine the universe of eligible tickers. Even if a ticker had a perfect opening range breakout, it was categorically rejected if it opposed the Daily Macro trend direction. This synergy between the Daily Gatekeeper and the 15M/1H ranks is what elevated Strategy 8 to a highly profitable **64.3% Win Rate**.

### 🥈 Strategy 10: Quad-Timeframe Unanimous Consensus
*   **Triggers**:
    *   **LONG**: 15M rank $\le$ 3 **AND** Daily rank $\le$ top 30% **AND** 1H rank $\le$ 5 **AND** 30M rank $\le$ 5.
    *   **SHORT**: 15M short_rank $\le$ 3 **AND** Daily short_rank $\le$ top 30% **AND** 1H short_rank $\le$ 5 **AND** 30M short_rank $\le$ 5.
*   **Holistic Backtest Context**: This strategy represents our most rigorous **multi-model ensemble test**. It mathematically proves that layering our 4 specialized timeframe models yields extreme precision. By demanding that the Daily Gatekeeper, 1-Hour core ranker, 30-Minute model, and 15-Minute sniper all fire in perfect unidirectional alignment, the strategy acts as a holistic backtest of the entire model suite. The resulting **85.7% Win Rate on Longs** validated the synergistic architecture of the system.

---

## ⚡ Volatility-Adjusted Risk Overrides

> [!WARNING]
> **DYNAMIC ATR OVERRIDE**: While strategy specifications outline static exit parameters (like standard +1.00% TP or -0.50% SL), **the active V2.3 engine programmatically overrides these static settings at trade entry.**
> 
> All executed trades are managed by the **Shadow Tracker**, which calculates dynamic volatility-based risk brackets via resampled 15M ATR (Stop Loss clamped [0.3%, 1.5%]; Take Profit clamped [0.75%, 2.5%]). Strategy-specific exit rules (like Strategy 10's immediate conviction flip close) are executed in addition to these core ATR brackets.

---

## 👁️ Key Related Notes
*   See the indicator features list calculated hourly: [[02 — Models/_Shared/Multi-Timeframe Models|Multi-Timeframe Models]].
*   See how shadow trades and dynamic brackets are managed: [[01 — Architecture/Execution & Runtime/Shadow Tracker & Execution Loop|Shadow Tracker & Execution Loop]].
*   Review our database schema columns for strategy logging: [[01 — Architecture/Data & Code/Database Architecture|Database Architecture]].
