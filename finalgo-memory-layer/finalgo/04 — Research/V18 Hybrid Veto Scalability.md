---
title: "Research: V18 Hybrid Veto Scalability & Portfolio Simulation"
type: reference
status: active
updated: 2026-06-12
tags: []
---
# 🧠 Research: V18 Hybrid Veto Scalability & Portfolio Simulation

## 📌 Background
In early June 2026, the Vanguard Engine tested a new hybrid architecture: combining the `v10` LambdaMART Ranker with a `v18` Random Forest Classifier (pure direction `>0 bps`, minimum 52% probability threshold). 

The goal was to test if a Classifier could act as a strict "veto" layer against the Ranker's cross-sectional choices to artificially clear the massive 10 bps statutory fee hurdle.

## 📈 12-Month Out-of-Sample Results (July 2025 - June 2026)
We ran the strict production models (trained only up to mid-2025) on the final 12-month untouched test set.

We tested three volume expansions of the `v10` Ranker:
*   **Logic A1**: Only evaluate the absolute #1 ranked stock (Long and Short).
*   **Logic A3**: Evaluate the Top 3 ranked stocks.
*   **Logic A5**: Evaluate the Top 5 ranked stocks.

### ⚖️ The Scalability Spectrum (Baseline 20% Cash Allocation / No Leverage)
To measure true portfolio compounding impact, all trades were run chronologically using a strict 20% maximum cash allocation per trade. If concurrent trades fired, the capital was fractionalized equally.

| Logic Config | Total Trades | Net Win Rate | Net Edge (per trade) | Portfolio Return | Max Drawdown | Sharpe Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logic A1** (Top 1) | 862 | **56.50%** | **+9.6 bps** | +17.63% | **-2.98%** | **3.51** |
| **Logic A3** (Top 3) | 2,571 | 52.94% | +4.6 bps | +26.45% | -3.68% | 3.38 |
| **Logic A5** (Top 5) | **4,277** | 51.79% | +3.1 bps | **+29.47%** | -5.97% | 2.76 |

### 🚀 High-Leverage Stress Test (Logic A3)
To prove the structural integrity of the Sharpe Ratio, Logic A3 was run through a hyper-aggressive growth simulator: **30% cash allocation per trade with 5x Intraday Margin (Effective 150% Exposure per trade)**.
*   **Total Executed Trades:** 2,571
*   **True Cumulative Net Return (Compounded):** **+396.06%**
*   **True Max Drawdown:** **-25.18%**
*   **Annualized Sharpe Ratio:** **3.44**

## 🧠 Strategic Conclusions
1.  **The Pure Direction Target Works:** Dropping the target label from `>20 bps breakout` (in `v16`) down to `>0 bps pure direction` (in `v18`) completely solved the catastrophic short-side bleeding. The Veto system perfectly trims the downside risk while allowing winners to run.
2.  **The Efficient Frontier:** Expanding the net from Top 1 to Top 5 successfully increases total absolute profit (+29% vs +17%), but strictly at the cost of risk-adjusted efficiency. The lower-conviction ranks drag the win rate down, doubling the drawdown and dropping the Sharpe ratio. 
3.  **The Workhorse:** Logic A3 acts as the perfect structural workhorse, massively increasing returns from A1 while keeping the Sharpe Ratio firmly in the 3.3+ territory, making it perfectly suited for immense intraday leverage.
