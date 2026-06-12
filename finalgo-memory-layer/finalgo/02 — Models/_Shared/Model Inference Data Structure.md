---
title: "Vanguard Model — Inference-Time Data Structure"
type: reference
status: active
updated: 2026-06-12
tags: []
---
# Vanguard Model — Inference-Time Data Structure

> **Model Registry Default:** `v8_upstox_3y`
> **Algorithm:** XGBoost `rank:pairwise`
> **Feature count:** **86 features per ticker per scan cycle**
> **Source file:** `scripts/feature_utils.py` → `scripts/vanguard/model_inference.py`

---

## 1. Raw Inputs (What We Fetch)

Before any feature is computed, the engine fetches the following raw market data for **every ticker** in the universe.

| Field    | Source                 | Period / Interval         | Notes                                    |
|----------|------------------------|---------------------------|------------------------------------------|
| `Open`   | Upstox Historical API  | 90-day window, 1H/Day candles| 60-day fallback to yfinance |
| `High`   | Upstox Historical API  | 90-day window, 1H/Day candles| |
| `Low`    | Upstox Historical API  | 90-day window, 1H/Day candles| |
| `Close`  | Upstox Historical API  | 90-day window, 1H/Day candles| |
| `Volume` | Upstox Historical API  | 90-day window, 1H/Day candles| |

> **Minimum bars required:** Sufficient history to cover 50-period moving averages and 24-bar lags (roughly 75+ bars).

---

## 2. Feature Engineering Pipeline (3 Stages)

The raw OHLCV data passes through three sequential transformation stages before reaching the model.

```
Raw OHLCV (per ticker)
       │
       ▼
 ┌─────────────────────────────────────┐
 │  Stage 1: Per-Ticker Technical      │
 │  Indicators & Lags                  │
 │  → 82 core indicator values         │
 └─────────────────────────────────────┘
       │
       ▼
 ┌─────────────────────────────────────┐
 │  Stage 2: Cross-Sectional Market    │
 │  Context (computed across all       │
 │  tickers in the same scan batch)    │
 │  → 4 market-relative features       │
 └─────────────────────────────────────┘
       │
       ▼
 ┌─────────────────────────────────────┐
 │  Stage 3: Cross-Sectional Z-Score   │
 │  (normalises technical features     │
 │  relative to peers in this batch)   │
 │  *Excluded features bypassed        │
 └─────────────────────────────────────┘
       │
       ▼
 Final 86-Feature Vector  →  XGBoost DMatrix  →  long_score / short_score
```

---

## 3. The 86-Feature Vector (Model Input)

Features are presented in the exact order they appear in `models/v8_upstox_3y/metadata.json`.

### 3.1 Price, Range & Return (4 features)
1. `Return`: 1-bar percentage return
2. `Log_Return`: log return
3. `HL_Range`: Normalised candle range (proxy for ATR%)
4. `OC_Range`: Candle body direction and size

### 3.2 Trend, Distance & Moving Averages (6 features)
5. `Dist_SMA_6`
6. `Dist_SMA_12`
7. `Dist_SMA_50`
8. `Dist_EMA_12`
9. `Dist_EMA_24`
10. `Dist_HMA_12`

### 3.3 Momentum & Oscillators (11 features)
11. `RSI_14`
12. `ROC_12`
13. `MOM_12_pct`
14. `CCI_20`
15. `WPR_14`
16. `TRIX_15`
17. `PPO`
18. `PPO_Signal`
19. `PPO_Hist`
20. `Dist_DPO_20`
21. `Ultimate_Osc`

### 3.4 Volatility & Bands (12 features)
22. `PercentB` (Bollinger position)
23. `Dist_BB_Upper`
24. `Dist_BB_Lower`
25. `BB_Width`
26. `Dist_Donchian_Upper`
27. `Dist_Donchian_Lower`
28. `Donchian_Width`
29. `Dist_Keltner_Upper`
30. `Dist_Keltner_Lower`
31. `Keltner_Width`

### 3.5 Volume & Flow (5 features)
32. `OBV_Dist`
33. `CMF_20`
34. `Volume_Change`
35. `Volume_Zscore`
36. `PVO`

### 3.6 Stochastic & Directional (6 features)
37. `Stoch_K`
38. `Stoch_D`
39. `Elder_Bull`
40. `Elder_Bear`
41. `Vortex_Plus`
42. `Vortex_Minus`

### 3.7 Statistical, Liquidity & Temporal Base (10 features)
43. `Price_Zscore`
44. `Rolling_Skew`
45. `Rolling_Kurt`
46. `Price_Accel`
47. `Hour` (Excluded from Z-Score)
48. `DayOfWeek` (Excluded from Z-Score)
49. `Dollar_Volume`
50. `RVOL`
51. `Dist_52W_High`
52. `Dist_52W_Low`

### 3.8 Temporal Lags & Streaks (14 features)
53-56. `Return_lag1`, `RSI_lag1`, `Volume_Zscore_lag1`, `OC_Range_lag1`
57-60. `Return_lag2`, `RSI_lag2`, `Volume_Zscore_lag2`, `OC_Range_lag2`
61-64. `Return_lag3`, `RSI_lag3`, `Volume_Zscore_lag3`, `OC_Range_lag3`
65. `Up_Streak` (Excluded from Z-Score)
66. `Down_Streak` (Excluded from Z-Score)

### 3.9 Microstructural & Intraday Action (16 features)
67. `Intraday_Return`
68. `VWAP_Dist`
69. `Is_Open_Hour` (Excluded from Z-Score)
70. `Is_Close_Hour` (Excluded from Z-Score)
71. `Time_To_Close` (Excluded from Z-Score)
72. `IBS`
73. `IBS_3`
74. `Buy_Pressure`
75. `Direction_Consistency_3`
76. `Direction_Consistency_5`
77. `RSI_Momentum`
78. `Return_Accel`
79. `Lower_Shadow`
80. `Upper_Shadow`
81. `Alpha_3H`
82. `Alpha_6H`

### 3.10 Cross-Sectional Market Context (4 features)
These 4 features are **computed across the entire batch of tickers** for the current scan. They encode the macro regime at the exact moment of scanning. **These are intentionally excluded from Z-scoring.**

83. `Market_Mean_Return`
84. `Relative_Return`
85. `Market_Mean_Volatility`
86. `Relative_Volatility`

---

## 4. Z-Scoring at Inference (Stage 3 Detail)

After computing all 86 features, Stage 3 applies **cross-sectional Z-scoring** across the batch in `scripts/vanguard/model_inference.py`.

Specific features are excluded from Z-scoring to preserve their inherent absolute values (e.g., categoricals, bounds, already-normalised contexts):

```python
exclude_from_z = [
    "ticker", "DateTime", "Close", "Open", "High", "Low", "Volume",
    "Market_Mean_Return", "Relative_Return", "Market_Mean_Volatility", "Relative_Volatility",
    "Hour", "DayOfWeek", "Sector",
    "Is_Open_Hour", "Is_Close_Hour", "Time_To_Close", "Up_Streak", "Down_Streak",
    "RSI_14_Raw", "Stoch_K_Raw", "PercentB_Raw", ...
]
```

This exactly mirrors the training preparation, ensuring zero train/inference distribution mismatch. If `scaler.pkl` is present and active, standard scaling is applied; if `uses_scaler` is false, it proceeds scale-invariant.

---

## 5. Model Output & Conviction Calculation

```
86-Feature Vector (per ticker)
         │
         ├──▶  xgb_long_model.json   ──▶  long_score  (rank-pairwise float)
         └──▶  xgb_short_model.json  ──▶  short_score (rank-pairwise float)
                                               │
                                               ▼
                          Long_Conviction  = long_score  - short_score
                          Short_Conviction = short_score - long_score
```

| Output            | Meaning |
|---|---|
| `long_score`      | Raw ranking score for best LONG candidate this hour |
| `short_score`     | Raw ranking score for best SHORT candidate this hour |
| `Long_Conviction` | Net LONG bias — positive = model agrees with long trade |
| `Short_Conviction`| Net SHORT bias — positive = model agrees with short trade |

The signal engine then gates on minimum conviction before passing the candidate to the AI audit layer.
