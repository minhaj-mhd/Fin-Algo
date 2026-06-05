# Training Data & Regime Requirements

**Date:** June 4, 2026  
**Subject:** The fundamental requirements for XGBoost training data volume, exposing the "row count fallacy" and defining the 3-year standard for all timeframes.

---

## 1. The Row Count Fallacy

When training high-frequency intraday models, it is easy to assume that because lower timeframes (e.g., 15-min) generate exponentially more rows of data, they require fewer calendar months of history to train a robust model. **This is completely false.**

Consider the data generated in a single trading day for 50 stocks:
* **15-min model:** ~26 candles × 50 stocks = **1,300 rows**
* **30-min model:** ~13 candles × 50 stocks = **650 rows**
* **1-hour model:** ~6 candles × 50 stocks = **300 rows**

The 15-minute model receives 4× more rows per day than the 1-hour model. However, all 1,300 of those rows originate from **the exact same market day** — meaning they share the exact same macro regime, sentiment, volatility environment, and institutional flow. 

Having 26 snapshots of a bullish day does not teach the XGBoost algorithm what a bearish day looks like. **Row count does not equal learning; calendar diversity does.**

---

## 2. Empirical Evidence of Regime Overfitting

The difference in performance between the Vanguard models proves this thesis:

| Model | Timeframe | Calendar Data | Row Count | OOS Result |
|---|---|---|---|---|
| **`v8_upstox_3y`** | 1-Hour | **3 Years** | ~540,000 | ✅ Institutional Grade (68-76% WR) |
| **`v1_30min`** | 30-Minute | **1 Year** | ~541,000 | ❌ Weak, Regime-Fragile |
| **`v1_15min`** | 15-Minute | **1 Year** | ~1,030,000 | ❌ Untested, Structurally Flawed |

The 30-minute model had the exact same raw row count (~540K) as the 1-hour model, but produced drastically worse results. The only difference was that the 1-hour model's data spanned 3 years of market regimes, while the 30-minute model memorized the specific quirks of a single 12-month window (June 2025 - April 2026). When evaluated on the May 2026 OOS dataset, the 30-minute model's edge collapsed during a single week of regime shift.

**Crucial Insight:** Lower timeframes contain *more* intraday noise (whipsaws, stop hunts). More noise actually requires *more* historical calendar data to cut through, not less. 

---

## 3. The 3-Year Standard Mandate

To prevent regime overfitting, **all Vanguard models across all timeframes (15-min, 30-min, 1-hour, Daily) MUST be trained on roughly 3 years of calendar data.**

### Why 3 Years is the Sweet Spot for Indian Equities:
* **< 2 Years (Dangerous):** The model only sees 1 major macro regime (e.g., an extended bull run). It memorizes that regime and fails immediately when conditions change.
* **2 Years (Minimum Viable):** Covers ~2 earnings cycles and likely 1 regime shift. Still fragile.
* **3 Years (Ideal):** Covers 3 full annual cycles, multiple regime transitions (e.g., post-COVID bull run, 2023-2024 corrections), all 4 earnings seasons multiple times, and enough FII/DII flow reversals to learn true structural alpha rather than temporary anomalies.
* **4-5 Years (Diminishing Returns):** Indian market microstructure has changed significantly (retail options explosion, algo trading growth, new margin rules). Data from >4 years ago may teach patterns that no longer exist, actively hurting the model.
* **> 5 Years (Harmful):** Pre-COVID market dynamics are fundamentally obsolete for high-frequency intraday momentum.

### Implementation Rule
No model trained on less than 2.5 years of calendar data shall be deployed to live trading or subjected to rigorous OOS edge evaluation, regardless of its row count.

---

## Backlinks
- [[Model Registry & File Structures]] — For tracking model versions and datasets.
- [[Multi-Timeframe Models]] — The overarching architecture for cross-timeframe execution.
- [[08. Model Analysis/30-Minute Vanguard Model/Weekly Consistency & Regimes]] — Empirical proof of the 1-year model failing during regime shifts.
