# Vanguard Model — Inference-Time Data Structure

> **Models:** `xgb_long_model.json` & `xgb_short_model.json`  
> **Algorithm:** XGBoost `rank:pairwise`  
> **Feature count:** **54 features per ticker per scan cycle**  
> **Source file:** `scripts/feature_utils.py` → `scripts/vanguard_signal_engine.py → calculate_conviction_scores()`

---

## 1. Raw Inputs (What We Fetch)

Before any feature is computed, the engine fetches the following raw market data for **every ticker** in the universe.

| Field    | Source                 | Period / Interval         | Notes                                    |
|----------|------------------------|---------------------------|------------------------------------------|
| `Open`   | Upstox Historical API  | 60-day window, day candles| Falls back to yfinance on Upstox failure |
| `High`   | Upstox Historical API  | 60-day window, day candles| Falls back to yfinance on Upstox failure |
| `Low`    | Upstox Historical API  | 60-day window, day candles| Falls back to yfinance on Upstox failure |
| `Close`  | Upstox Historical API  | 60-day window, day candles| Falls back to yfinance on Upstox failure |
| `Volume` | Upstox Historical API  | 60-day window, day candles| Falls back to yfinance on Upstox failure |

> **Minimum bars required:** 25 candles. Tickers with fewer bars are dropped silently from the scan.

---

## 2. Feature Engineering Pipeline (3 Stages)

The raw OHLCV data passes through three sequential transformation stages before reaching the model.

```
Raw OHLCV (per ticker)
       │
       ▼
 ┌─────────────────────────────────────┐
 │  Stage 1: Per-Ticker Technical      │
 │  Indicators (compute_features())    │
 │  → 50 raw indicator values          │
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
 │  Market context & temporal features │
 │  are EXCLUDED from Z-scoring        │
 └─────────────────────────────────────┘
       │
       ▼
 Final 54-Feature Vector  →  XGBoost DMatrix  →  long_score / short_score
```

---

## 3. The 54-Feature Vector (Model Input)

Features are presented in the exact order they appear in `models/model_metadata.json`.

### 3.1 Price & Return (4 features)

| # | Feature Name | Formula / Description |
|---|---|---|
| 1 | `Return` | `Close.pct_change()` — 1-bar percentage return |
| 2 | `Log_Return` | `log(Close / Close.shift(1))` — log return |
| 3 | `HL_Range` | `(High - Low) / Close` — normalised candle range (proxy for ATR%) |
| 4 | `OC_Range` | `(Close - Open) / Open` — candle body direction and size |

---

### 3.2 Trend & Momentum (16 features)

| # | Feature Name | Formula / Description |
|---|---|---|
| 5  | `Dist_SMA_6`   | `(Close - SMA(6)) / Close` — distance from 6-period simple MA |
| 6  | `Dist_SMA_12`  | `(Close - SMA(12)) / Close` — distance from 12-period simple MA |
| 7  | `Dist_EMA_12`  | `(Close - EMA(12)) / Close` — distance from 12-period exponential MA |
| 8  | `Dist_EMA_24`  | `(Close - EMA(24)) / Close` — distance from 24-period exponential MA |
| 9  | `Dist_HMA_12`  | `(Close - HMA(12)) / Close` — distance from 12-period Hull MA |
| 10 | `RSI_14`       | 14-period Relative Strength Index (range 0–100) |
| 11 | `ROC_12`       | `Close.pct_change(12)` — 12-bar Rate of Change |
| 12 | `MOM_12_pct`   | `Close.pct_change(12)` — percentage momentum over 12 bars |
| 13 | `CCI_20`       | Commodity Channel Index (20-period) |
| 14 | `WPR_14`       | Williams %R (14-period, range −100 to 0) |
| 15 | `TRIX_15`      | Triple-smoothed EMA % change (15-period) |
| 16 | `PPO`          | Percentage Price Oscillator `(EMA12 - EMA26) / EMA26 * 100` |
| 17 | `PPO_Signal`   | EMA(9) of PPO |
| 18 | `PPO_Hist`     | `PPO - PPO_Signal` |
| 19 | `Dist_DPO_20`  | Detrended Price Oscillator (20-period) / Close |
| 20 | `Ultimate_Osc` | Ultimate Oscillator (7/14/28 periods, range 0–100) |

---

### 3.3 Volatility & Bands (12 features)

#### Bollinger Bands (20-period, 2σ)

| # | Feature Name | Formula / Description |
|---|---|---|
| 21 | `PercentB`         | `(Close - BB_Lower) / (BB_Upper - BB_Lower)` — position within bands |
| 22 | `Dist_BB_Upper`    | `(BB_Upper - Close) / Close` — distance to upper band |
| 23 | `Dist_BB_Lower`    | `(Close - BB_Lower) / Close` — distance to lower band |
| 24 | `BB_Width`         | `(BB_Upper - BB_Lower) / Close` — band width (volatility proxy) |

#### Donchian Channel (20-period)

| # | Feature Name | Formula / Description |
|---|---|---|
| 25 | `Dist_Donchian_Upper` | `(20-bar High - Close) / Close` — distance to swing high |
| 26 | `Dist_Donchian_Lower` | `(Close - 20-bar Low) / Close` — distance to swing low |
| 27 | `Donchian_Width`      | `(20-bar High - 20-bar Low) / Close` — channel width |

#### Keltner Channel (20-period)

| # | Feature Name | Formula / Description |
|---|---|---|
| 28 | `Dist_Keltner_Upper` | `(Keltner Upper - Close) / Close` |
| 29 | `Dist_Keltner_Lower` | `(Close - Keltner Lower) / Close` |
| 30 | `Keltner_Width`      | `Keltner Channel Width / Close` |

---

### 3.4 Volume & Flow (5 features)

| # | Feature Name     | Formula / Description |
|---|---|---|
| 31 | `OBV_Dist`      | `(OBV - SMA(OBV, 20)) / abs(SMA(OBV, 20))` — OBV divergence from trend |
| 32 | `CMF_20`        | Chaikin Money Flow (20-period, range −1 to +1) |
| 33 | `Volume_Change` | `Volume.pct_change()` — 1-bar volume change |
| 34 | `Volume_Zscore` | `(Volume - rolling_mean(24)) / rolling_std(24)` — volume vs 24-bar norm |
| 35 | `PVO`           | Percentage Volume Oscillator `(EMA12(Vol) - EMA26(Vol)) / EMA26(Vol) * 100` |

---

### 3.5 Oscillators (4 features)

#### Stochastic (14/3 period)

| # | Feature Name | Formula / Description |
|---|---|---|
| 36 | `Stoch_K` | Fast stochastic (range 0–100) |
| 37 | `Stoch_D` | Slow stochastic — SMA(3) of K |

#### Elder Ray (13-period EMA)

| # | Feature Name   | Formula / Description |
|---|---|---|
| 38 | `Elder_Bull` | `(High - EMA(13)) / Close` — bull power (how far High exceeds trend) |
| 39 | `Elder_Bear` | `(Low - EMA(13)) / Close` — bear power (how far Low is below trend) |

---

### 3.6 Directional — Vortex (2 features)

| # | Feature Name     | Formula / Description |
|---|---|---|
| 40 | `Vortex_Plus`  | VI+ — upward trend strength (14-period) |
| 41 | `Vortex_Minus` | VI− — downward trend strength (14-period) |

---

### 3.7 Statistical / Temporal (6 features)

| # | Feature Name     | Formula / Description |
|---|---|---|
| 42 | `Price_Zscore`   | `(Close - rolling_mean(24)) / rolling_std(24)` — price position vs 24-bar history |
| 43 | `Rolling_Skew`   | 24-bar rolling skewness of Close |
| 44 | `Rolling_Kurt`   | 24-bar rolling kurtosis of Close |
| 45 | `Price_Accel`    | `Close.pct_change().diff()` — second derivative of price (momentum acceleration) |
| 46 | `Hour`           | Hour of day (0–23) — intraday seasonality |
| 47 | `DayOfWeek`      | Day of week (0=Mon … 4=Fri) — day-of-week effect |

---

### 3.8 Liquidity & Ranking Anchors (3 features)

| # | Feature Name     | Formula / Description |
|---|---|---|
| 48 | `Dollar_Volume`  | `Close × Volume` — absolute liquidity in INR |
| 49 | `RVOL`           | `Volume / rolling_mean(Volume, 20)` — relative volume vs 20-bar average |
| 50 | `Dist_52W_High`  | `(Close - 52-week High) / 52-week High` — proximity to yearly peak (always ≤ 0) |

---

### 3.9 Cross-Sectional Market Context (4 features)

These are **computed across the entire batch of tickers** for the current scan, not on a single ticker's history. They encode the macro regime at the exact moment of scanning.

| # | Feature Name             | Formula / Description |
|---|---|---|
| 51 | `Market_Mean_Return`     | Average of `Return` across all tickers this scan — market's current momentum |
| 52 | `Relative_Return`        | `Return - Market_Mean_Return` — ticker outperformance vs peers |
| 53 | `Market_Mean_Volatility` | Average of `HL_Range` across all tickers — market's fear level |
| 54 | `Relative_Volatility`    | `HL_Range / Market_Mean_Volatility` — how volatile this ticker is vs peers |

> **These 4 features are intentionally excluded from Z-scoring.** They are already relative/normalised by construction — Z-scoring them would destroy their signal.

---

## 4. Z-Scoring at Inference (Stage 3 Detail)

After computing all 54 features, Stage 3 applies **cross-sectional Z-scoring** across the batch for features `#1–#50`, excluding the 4 market-context features (`#51–#54`) and the 2 temporal features (`Hour`, `DayOfWeek`):

```python
# From vanguard_signal_engine.py → calculate_conviction_scores()
exclude_from_z = [
    "ticker", "DateTime", "Close", "Open", "High", "Low", "Volume",
    "Market_Mean_Return", "Relative_Return",
    "Market_Mean_Volatility", "Relative_Volatility",
    "Hour", "DayOfWeek",
]
features_to_zscore = [c for c in self.feature_cols if c not in exclude_from_z]

for col in features_to_zscore:
    mean = scores_df[col].mean()   # mean across all N tickers this batch
    std  = scores_df[col].std()
    scores_df[col] = (scores_df[col] - mean) / (std + 1e-8)
```

This exactly mirrors `prepare_ranking_data.py` (training), ensuring zero train/inference distribution mismatch.

---

## 5. Model Output & Conviction Calculation

```
54-Feature Vector (per ticker)
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

The signal engine then gates on **`min_conviction = 0.15`** before passing the candidate to the Gemini AI audit layer.

---

## 6. What the Model Does NOT See

| Excluded | Reason |
|---|---|
| Absolute price (`Open`, `High`, `Low`, `Close`) | Must be price-agnostic to generalise across ₹50–₹5000 stocks |
| `Volume` (raw) | Only derived volume features (Z-score, RVOL, DollarVol) are used |
| `Dist_SMA_50` | Present in `feature_utils.py` but **absent from the saved feature list** — excluded during `prepare_ranking_data.py` |
| `Next_Hour_Return` | Training label only — never available at inference |
| News / fundamentals | Purely technical model; Gemini AI handles fundamental veto |
| TradingView Sentiment (`tv_ta`) | Post-model signal gate, not a model input feature |
| Live/intraday tick data | Model uses daily candles from Upstox history, not real-time ticks |
