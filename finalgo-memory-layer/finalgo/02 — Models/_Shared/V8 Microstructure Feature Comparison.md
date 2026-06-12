# Vanguard v8 Model Comparison: Baseline vs Microstructure Features

**Date:** May 28, 2026  
**Active Version:** Vanguard v8 (Retrained)  
**Dataset:** 3-Year Upstox Hourly Data (1,107,513 rows, 6,461 query groups)

---

## 1. Overview
This report compares the performance of the baseline **v8_upstox_3y** model (trained on May 27, 2026) against the updated version (trained on May 28, 2026). The updated version incorporates 11 new, leak-free microstructure features (e.g., Internal Bar Strength, Buy Pressure) and completely resolves the lookahead data leakage previously associated with `Gap_Pct`.

---

## 2. Model Performance Comparison

The evaluations were conducted on a strict out-of-time temporal test set (the last 20% of query periods, representing 222,354 rows across 1,293 queries).

### Rank Correlation (Spearman Rho)
Spearman Rho measures the alignment between the model's scores and the actual cross-sectional returns for all stocks in a given hour.

| Metric | Old v8 Baseline | New Version (Microstructure) | Absolute Change | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Long Test Spearman Rho** | 0.0389 | **0.0461** | **+0.0072** | **+18.5%** |
| **Short Test Spearman Rho** | 0.0399 | **0.0490** | **+0.0091** | **+22.8%** |
| **Long Train-Test Gap** | N/A | 0.0140 | — | Stable (No Overfitting) |
| **Short Train-Test Gap** | N/A | 0.0139 | — | Stable (No Overfitting) |

---

### Win Rate (Precision @ K)
In this system, a selection is counted as a "hit" (win) if its return beats the hour's median return (for Longs) or falls below the median return (for Shorts).

| Metric | Old v8 Baseline | New Version (Microstructure) | Absolute Change | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Long Win Rate @ 1 (P@1)** | 57.71% | **59.16%** | **+1.45%** | **+2.5%** |
| **Long Win Rate @ 3 (P@3)** | 54.25% | **55.82%** | **+1.57%** | **+2.9%** |
| **Long Win Rate @ 5 (P@5)** | 53.65% | **55.12%** | **+1.47%** | **+2.7%** |
| **Short Win Rate @ 3 (P@3)** | 56.67% | **57.28%** | **+0.61%** | **+1.1%** |

---

### Portfolio Return Metrics
Average hourly returns for the top-3 ranked selections compared to a random stock pick on the test set.

| Metric | Old v8 Baseline | New Version (Microstructure) | Change | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Top-3 Long Return (hourly)** | +0.0668% | **+0.0810%** | **+0.0142%** | **+21.3%** |
| **Top-3 Short Return (hourly)** | *N/A (Not logged)* | **+0.1299%** | — | — |
| **Random Pick Return (hourly)** | +0.0025% | **+0.0025%** | 0.0000% | 0.0% |

---

## 3. Key Observations

### A. The Gap_Pct Validation
* **Historical Leaked IC:** `-0.19916` (t-stat: 61.95)
* **True Corrected IC:** `-0.00783` (t-stat: 1.14)
* **Diagnosis:** The baseline's high predicted performance for `Gap_Pct` was caused by lookahead data leakage in the analysis script (accessing the current day's closing price in hours 2-6). Once corrected to use the previous day's close, the predictive power disappeared. Consequently, `Gap_Pct` was excluded.

### B. Microstructure Features
To replace `Gap_Pct`, we introduced 11 leak-free high-frequency microstructure features:
1. **IBS (Internal Bar Strength):** Measures where the hourly close falls within the High-Low range. It became the **#1 feature in the Long model** (importance gain: 8.85) and the **#2 feature in the Short model** (importance gain: 4.55).
2. **Buy Pressure:** IBS scaled by relative volume. Captures exhaustion spikes.
3. **Shadow Ratios:** Candelstick upper/lower shadows mapping structural supply and demand.

### C. Core Takeaway
The addition of the microstructure features provided a massive, non-leaky rank correlation lift (+18.5% Long, +22.8% Short) without increasing model overfitting. The win rate for the Top-1 stock is now extremely close to the `60.0%` mark, which represents a very solid operational edge for Vanguard's live trading execution.
