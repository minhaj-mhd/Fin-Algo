---
title: "Vanguard Feature Engineering & Normalization Strategy"
type: reference
status: active
updated: 2026-06-12
tags: []
---
# Vanguard Feature Engineering & Normalization Strategy

This document outlines the specialized feature engineering pipeline used to prepare market data for the Vanguard XGBoost Ranking Model. 

Because we are training a **cross-sectional ranking model** (comparing multiple stocks against each other at the same point in time), **every single feature must be mathematically comparable across different assets**. If a feature relies on absolute stock prices or raw volume, the model will overfit by learning the absolute price levels rather than the underlying technical setups.

## 1. Feature Categorization

We have 48 features categorized into four distinct groups based on how they must be normalized.

### Category 1: Absolute Price Indicators (Percentage Distance)
These indicators output raw price values. They are converted into a percentage distance relative to the `Close` price: `(Indicator - Close) / Close` or `(Close - Indicator) / Close`.
*   **Moving Averages:** `SMA_6`, `SMA_12`, `EMA_12`, `EMA_24`, `HMA_12` -> `Dist_SMA_6`, etc.
*   **Price Bands:** `BB_Upper`, `BB_Lower`, `Donchian_Upper`, `Donchian_Lower`, `Keltner_Upper`, `Keltner_Lower` -> `Dist_BB_Upper`, etc.
*   **Band Widths:** `BB_Width`, `Donchian_Width`, `Keltner_Width` -> Scaled by dividing by `Close`.
*   **Elder Ray:** `Elder_Bull`, `Elder_Bear` -> Scaled by dividing by `Close`.
*   **DPO (Detrended Price Oscillator):** -> Scaled by dividing by `Close`.

### Category 2: Absolute Price Differences (Replaced)
These indicators subtract two raw prices.
*   **MACD:** Replaced entirely with **PPO** (Percentage Price Oscillator) because MACD is an absolute difference while PPO is normalized.
*   **MOM (Momentum):** Replaced with Percentage Momentum (which is effectively Rate of Change).
*   **Price_Accel:** Changed from absolute acceleration to percentage acceleration.

### Category 3: Absolute Volume (Rate of Change)
*   **OBV (On Balance Volume):** Because raw cumulative volume scales with the size of the company, OBV is converted to a distance from its own moving average `(OBV - SMA(OBV)) / SMA(OBV)`.

### Category 4: Inherently Bounded (Unchanged)
These indicators are already bounded or standardized.
*   **Oscillators (0 to 100 or -100 to 0):** `RSI_14`, `WPR_14`, `Ultimate_Osc`, `Stoch_K`, `Stoch_D`
*   **Normalized Ratios/Percentages:** `ROC_12`, `TRIX_15`, `PercentB`, `Volume_Change`, `PVO`
*   **Statistical/Standardized:** `CCI_20`, `CMF_20`, `Vortex_Plus`, `Vortex_Minus`, `Volume_Zscore`, `Price_Zscore`, `Rolling_Skew`, `Rolling_Kurt`

## 2. Cross-Sectional Z-Scoring

After all features are normalized as percentages or bounds, the final step is **Cross-Sectional Z-Scoring**.

1.  The data is grouped by the specific hour (`Query_ID`).
2.  A Z-score is applied to every feature across all 50 stocks for that specific hour.
3.  **Formula:** `(Feature_Value - Mean_of_all_stocks) / Std_Dev_of_all_stocks`

**Why?** This provides the model with *market context*. A stock with a +2% return is a weak signal if the entire market is up +5%, but a very strong signal if the market is down -2%. The cross-sectional Z-score explicitly feeds this relative outperformance directly into the model.

## 3. Training Modifications

To prevent overfitting, the XGBoost training pipeline strictly adheres to these rules:
*   **No Global Scaling:** We do not use a global `StandardScaler` across the entire time-series dataset, as this destroys the cross-sectional relativities.
*   **Regularization:** The model uses shallow trees (`max_depth=3`), L2 Regularization (`lambda`), L1 Regularization (`alpha`), and `min_child_weight` penalties to force it to ignore noise and focus only on robust, repeating patterns.
