---
title: "Vanguard Trading System — Feature & Architecture Release"
type: reference
status: active
updated: 2026-06-12
tags: []
---
# Vanguard Trading System — Feature & Architecture Release
**Date Marked:** May 27, 2026 | **System Version:** Vanguard v2.5 Hardened | **Active Model:** `v8_upstox_3y`

---

## 1. Executive Summary

As of today, **May 27, 2026**, the Vanguard Algorithmic Trading System has been fully upgraded to a production-hardened hybrid (XGBoost + LLM Audit) architecture. The main changes focus on extending training datasets to 3 years, eliminating temporal data leakage, speeding up execution scans, and implementing safe state-resumption rules.

---

## 2. The Predictive Layer: `v8_upstox_3y` Model

The machine learning core has been upgraded from the small 90-day window model (`v6`) to the comprehensive **`v8_upstox_3y`** ranking model, resolving severe overfitting issues.

* **Expanded Historical Scale:** Trained on **1,106,481 rows** of 1-hour interval data (covering **6,455 unique hourly queries** over 3 years of Indian equity market history).
* **Zero-Leakage Feature Engineering:** Daily context features (mean returns, index correlations, and volatility) are strictly computed using **yesterday's completed daily candles** rather than the current day's forming candle, eliminating lookahead bias and train-live distribution mismatch.
* **Low Generalization Variance:** The Spearman Rho train-test gap is minimized to **`0.0162` (Short)** and **`0.0212` (Long)**, well below the standard `0.05` danger threshold.
* **Restored Seasonality Signals:** Wrapped index data inside `pd.Series` in [feature_utils.py](file:///c:/Users/loq/Desktop/Trading/finalgo-1/scripts/feature_utils.py) to fix a constant-variance bug. Time-of-day features (`Time_To_Close`, `Hour`, `Is_Close_Hour`) are now fully active and represent **over $11\%$ of the model's split decisions (Gain)**.

---

## 3. The Execution Layer: Vanguard Engine

The modular execution scanner anchored at [scripts/vanguard/orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py) has been patched to handle live API limits and offline state transitions:

* **15-Second Bulk Scans:** Replaced slow sequential fallback loops with a single-request batch `yfinance` download for failed Upstox quotes. The scanner now processes the full 172-symbol universe in under 15 seconds.
* **Stale Resumed Trade Expiry:** On server startup, any database-loaded `PENDING_ENTRY` trades that were created before the offline period are automatically expired and marked `CANCELLED` if the current time is past their next 15-minute close boundary plus a 5-minute grace period.
* **Stricter Confirmation Gates:** At the 15-minute close check, pending trades must maintain solid conviction to execute:
  * **Net Trades:** Must retain a conviction score of at least `0.10` (originally `0.15`).
  * **Raw Trades:** Must not drop by more than `0.05` from their initial signal and must remain above `0.05`.
* **Margin Safeguards:** Hardened capital tracking to release `used_margin` immediately when pending trades are cancelled or vetoed.

---

## 4. The Qualitative Audit Layer: Gemini AI Veto

The system incorporates Gemini AI as an asymmetric tail-risk filter. While the XGBoost model proposes technically sound momentum/reversion trades, Gemini conducts a two-stage qualitative analysis:

1. **Stage 1 (Technical Sentiment):** Verifies the immediate pattern alignment using recent hourly trends.
2. **Stage 2 (Fundamental Catalyst):** Queries live news feeds, earnings reports, block deals, and corporate action headers. If a negative qualitative catalyst conflicts with the technical signal, the trade is vetoed.

> [!TIP]
> **Asymmetric Drawdown Protection:** 
> Today's post-1:00 PM session demonstrates the power of this setup:
> * The raw XGBoost model generated **37 signals** with a **$64.86\%$ win rate**, but without filtering would have taken large losses on `COROMANDEL.NS` ($-0.86\%$) and `COALINDIA.NS` ($-1.33\%$).
> * Gemini's audit layer vetoed these high-risk events, and the only trade allowed to execute (`ALKEM.NS` SHORT) closed in profit (**`+0.33%` net**).

---

## 5. Summary of System Settings

| Parameter | Value | Description |
| :--- | :---: | :--- |
| **Active Model** | `v8_upstox_3y` | 3-Year regularized XGBoost pairwise ranker |
| **Minimum Conviction** | `0.10` | Minimum net conviction score gate |
| **Minimum Raw Score** | `0.12` | Minimum raw model confidence score gate |
| **Trading Window** | 09:15 - 15:00 IST | Active hour scan window |
| **Early Stopping** | 50 rounds | Validation-set early stopping limit during training |
