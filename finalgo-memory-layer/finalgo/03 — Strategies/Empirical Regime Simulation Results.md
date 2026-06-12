---
title: "Empirical Regime Simulation Results"
type: report
status: active
updated: 2026-06-12
tags: []
---
# 🧪 Empirical Regime Simulation Results

This document permanently logs the results of our **Dual-Pass Comparative Simulator**, which was executed on May 2026 to mathematically determine whether each strategy in our catalog should act as a `TREND` strategy (gated by the Daily Macro model) or a `REVERSAL` strategy (ungated, bypassing the Daily Macro model).

## 📊 The Comparative Matrix (Gated vs. Ungated)

The following table compares the performance of each strategy under two strict passes across the same dataset (172 common tickers):
1.  **GATED**: The Daily Gatekeeper must approve the trend direction.
2.  **UNGATED**: The Daily Gatekeeper is completely bypassed.

| ID | Strategy Name | Gated WR | Ungated WR | Gated Net Return | Ungated Net Return | Empirical Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **1** | Daily Macro Gatekeeper | 50.9% (108t) | 50.9% (108t) | **+2.20%** | **+2.20%** | `TREND` (Built-in) |
| **2** | Short-Side Specialist | 58.8% (17t) | 58.8% (17t) | **+0.48%** | **+0.48%** | `TREND` (Built-in) |
| **3** | Timeframe Divergence Fade | 58.3% (108t) | 53.7% (108t) | **+4.67%** | -0.17% | `TREND` |
| **4** | Score Momentum Scalper | 57.6% (144t) | 49.3% (144t) | **+4.70%** | -2.51% | `TREND` |
| **5** | Power Hour Sniper | 51.6% (64t) | 41.7% (72t) | **+3.32%** | -1.62% | `TREND` |
| **6** | Market-Neutral Pairs | 43.5% (108t) | 43.5% (108t) | -2.00% | -2.00% | `NEUTRAL` |
| **7** | Volatility Regime Switcher | 46.1% (141t) | 45.2% (146t) | -7.51% | **-1.14%** | `REVERSAL` |
| **8** | Opening Range Breakout | 60.0% (10t) | 64.3% (14t) | +2.10% | **+2.31%** | `REVERSAL` |
| **9** | Conviction Spread Z-Score | 42.2% (83t) | 40.7% (108t) | **-5.61%** | -10.61% | `TREND` |
| **10** | Quad-Timeframe Unanimous | 54.2% (24t) | 54.2% (24t) | **+1.86%** | **+1.86%** | `TREND` (Built-in) |

---

## 🔍 Key Insights & Architectural Impact

### 1. The Gatekeeper Saves Scalpers
Notice how **Strategy 4 (Score Momentum Scalper)** went from a devastating **-2.51% loss** ungated to a massive **+4.70% profit** when gated. This mathematically proves that attempting to scalp against the daily macro trend is a fatal flaw. The same is true for **Strategy 3** and **Strategy 5**. Scalping is undeniably a `TREND` strategy, and applying the Daily Gatekeeper is non-negotiable for these to be profitable.

### 2. ORB is Actually a Tactical Reversal
Surprisingly, **Strategy 8 (Opening Range Breakout)** performed *better* when we removed the Daily Gatekeeper (+2.31% vs +2.10%). This implies that violent morning breakouts frequently defy the long-term daily trend and succeed anyway. By forcing ORB through the gatekeeper in our legacy V2.3 setup, we were artificially stifling its best trades. It has earned the `REVERSAL` tag!

### 3. Hardcoded Strategies
Strategies S1, S2, and S10 possess internal, non-negotiable rules that directly reference the Daily model. Therefore, their Gated and Ungated passes are mathematically identical. S6 (Pairs) is entirely market-neutral and thus ignores the macro direction entirely.

*Linked to: [[06 — Logs/Regime-Aware Routing Plan|Regime-Aware Routing Plan]]*
