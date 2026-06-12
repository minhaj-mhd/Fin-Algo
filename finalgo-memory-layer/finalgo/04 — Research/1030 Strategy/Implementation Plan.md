---
title: "10:30 AM Momentum Prediction System — Implementation Plan (V2)"
type: spec
status: active
updated: 2026-06-12
tags: []
---
# 10:30 AM Momentum Prediction System — Implementation Plan (V2)

> [!IMPORTANT]
> **V2 Revision**: This plan supersedes V1 after a critical review that identified overfitting risks, tautological features, and an undersized dataset. All changes are documented below.

Build a two-layer prediction system that runs at 10:30 AM IST each trading day:
- **Layer A (Market Filter)**: Predicts overall Nifty direction for the rest of the day — decides whether to trade, and whether to lean long or short.
- **Layer B (Stock Selector)**: Cross-sectionally ranks all stocks by expected rest-of-day return — picks the best candidates.

Both layers use XGBoost with walk-forward validation.

---

## V2 Changes (Critical Fixes)

### 1. Data Source: 250 → ~1100 Trading Days
**Problem**: V1 used the 15-min cache (~250 days). Training XGBoost on 250 rows with 15 features is a pilot study, not a model. Walk-forward gave only 3 test months.

**Fix**: Switched to the **30-min cache** (`data/raw_upstox_cache/`) as primary data source. This covers **Jan 2022 → present (~4.5 years, ~1100 trading days)**. The morning window becomes 3 bars (09:15, 09:45, 10:15 IST) instead of 5 from the 15-min cache — coarser but with **4x the data**.

### 2. Removed Advance-Decline Ratio (Near-Tautological)
**Problem**: Feature #12 (`Advance_Decline_Ratio`) counted stocks up vs down at 10:15. This is computed from the same ~170 stocks whose returns we're trying to predict. It's not technically look-ahead, but it's pricing in the very thing Layer A is supposed to predict. Near-tautological.

**Fix**: Removed from Layer A features entirely. Also removed `BankNifty_Nifty_Spread` (redundant with global index returns) and `DayOfWeek` (noise at this sample size).

### 3. Removed Slow-Moving Daily Indicators from Layer B
**Problem**: `Dist_52W_High`, `Dist_52W_Low`, `BB_Position_20`, `RSI_14_Daily` are slow-moving and encode the stock's longer-term narrative, not its intraday morning state. For a 10:30→15:15 trade they're noise at best, and they eat up model capacity.

**Fix**: Removed all four. Also removed `Return_Streak` (lagging), shadow features (marginal on 3 bars), morning drawdown/rally (need finer resolution), and `Relative_Morning_Volume` (derivative of a derivative). Layer B is now 20 features, down from 30.

### 4. Walk-Forward: 4 Folds → ~15 Folds
**Problem**: 8-1-1 month splits on 13 months = 4 folds with tiny test sets. Confidence intervals were decorative.

**Fix**: 6-2-2 month rolling splits, stepping by 2 months. With ~44 months of data this gives **~15 non-overlapping test folds**, each with ~40 test days.

### 5. Stronger Regularization
Added `reg_alpha=1.0`, `reg_lambda=5.0` to both models. With ~1100 effective dates, overfitting is still a risk — aggressive L1/L2 helps.

---

## Folder Structure

```
finalgo/
├── scripts/
│   └── strategy_1030/              # All code for this strategy
│       ├── __init__.py
│       ├── config.py               # Constants, paths, feature lists
│       ├── data_collection.py      # Fetch global indices + build datasets
│       ├── feature_engineering.py  # Z-scoring + daily indicator computation
│       ├── train.py                # Walk-forward training (both layers)
│       └── backtest.py             # Two-layer backtest simulation
│
├── data/
│   └── strategy_1030/              # All data artifacts
│       ├── global_indices/         # Cached yfinance downloads
│       ├── global_indices_merged.csv
│       ├── dataset_market.csv      # Layer A dataset (~1100 rows, 1 per day)
│       └── dataset_stocks.csv      # Layer B dataset (~1100 × 170 rows)
│
└── models/
    └── strategy_1030/
        ├── market_filter/          # Layer A models (per-fold + final)
        └── stock_selector/         # Layer B models (per-fold + final)
```

---

## Data Sources

| Source | Path | Resolution | Date Range | Rows/Ticker |
|---|---|---|---|---|
| **30-min cache (PRIMARY)** | `data/raw_upstox_cache/` | 30-min bars (UTC timestamps) | Jan 2022 → present | ~14,000 |
| Daily cache | `data/raw_upstox_daily_cache/` | Daily OHLCV | ~5 years | ~1,240 |
| yfinance | Downloaded at runtime | Daily closes | 5 years | ~1,250 |

The 30-min cache timestamps are in UTC. The morning mapping is:

| UTC | IST | Role |
|---|---|---|
| 03:45 | 09:15 | Bar 1 (opening candle) |
| 04:15 | 09:45 | Bar 2 |
| 04:45 | 10:15 | Bar 3 (last morning bar / entry price) |
| 09:45 | 15:15 | Bar 13 (exit price) |

---

## Feature Specifications

### Layer A — Market Filter (10 features)

| # | Feature | Source |
|---|---|---|
| 1 | `SP500_Overnight_Ret` | yfinance (shifted 1d) |
| 2 | `Nasdaq_Overnight_Ret` | yfinance (shifted 1d) |
| 3 | `Nikkei_Overnight_Ret` | yfinance (shifted 1d) |
| 4 | `HangSeng_Overnight_Ret` | yfinance (shifted 1d) |
| 5 | `VIX_Level` | yfinance (shifted 1d) |
| 6 | `VIX_Change` | yfinance (shifted 1d) |
| 7 | `VIX_Zscore_20d` | yfinance (shifted 1d) |
| 8 | `Nifty_Gap` | Mean stock gap at open |
| 9 | `Nifty_Morning_Ret` | Mean stock return 09:15→10:15 |
| 10 | `Prev_Day_Nifty_Ret` | yfinance (shifted 1d) |

**Removed from V1**: `Advance_Decline_Ratio` (tautological), `BankNifty_Nifty_Spread` (redundant), `Nifty_Morning_Range` (correlated with morning ret), `Nifty_Gap_Fill` (derivative of gap), `DayOfWeek` (noise).

**Target**: `Nifty_ROD_Return` = mean stock rest-of-day return (proxy for Nifty 50 index).

### Layer B — Stock Selector (20 features)

| # | Feature | Type |
|---|---|---|
| 1 | `Opening_Gap` | Intraday |
| 2 | `Gap_Fill_Status` | Intraday |
| 3 | `Morning_Return` | Intraday |
| 4 | `Morning_Range` | Intraday |
| 5 | `ORB_Position` | Intraday |
| 6 | `Morning_Body_Direction` | Intraday |
| 7 | `Morning_Volume_Total` | Intraday |
| 8 | `Morning_Volume_Ratio` | Intraday |
| 9 | `Volume_Acceleration` | Intraday |
| 10 | `VWAP_Deviation` | Intraday |
| 11 | `First_Candle_Return` | Intraday |
| 12 | `First_Candle_Range` | Intraday |
| 13 | `Post_Open_Trend` | Intraday |
| 14 | `IBS_Morning` | Intraday |
| 15 | `Prev_Day_Return` | Daily (1d lag) |
| 16 | `Prev_Day_Range` | Daily (1d lag) |
| 17 | `Prev_Day_Volume_Ratio` | Daily (1d lag) |
| 18 | `Prev_Day_IBS` | Daily (1d lag) |
| 19 | `Dist_SMA_20` | Daily (1d lag) |
| 20 | `Relative_Morning_Return` | Cross-sectional |

**Removed from V1**: `Dist_52W_High`, `Dist_52W_Low`, `BB_Position_20`, `RSI_14_Daily` (slow-moving, irrelevant for 5-hour window), `Morning_Upper/Lower_Shadow_Avg` (marginal on 3 bars), `Morning_Max_Drawdown/Rally` (need finer resolution), `Return_Streak` (lagging), `Relative_Morning_Volume` (derivative of derivative).

**Target**: `Target` = `(close_15:15 / close_10:15) - 1` per stock.

All Layer B features are **cross-sectionally Z-scored per day** before training. Effective sample size is ~1100 independent dates, not ~187,000 rows.

---

## Training

### [train.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_1030/train.py)

**Walk-Forward Schema**: Rolling 6-month train, 2-month val, 2-month test. Steps by 2 months → ~15 folds with ~40 test days each.

**Layer A — Market Direction Model**
- XGBoost regressor (`reg:squarederror`)
- 10 features, ~1100 rows
- Regularization: `reg_alpha=1.0, reg_lambda=5.0, max_depth=3`
- Reports: aggregate OOS directional accuracy + Pearson ρ across all folds

**Layer B — Stock Ranking Model**
- XGBoost `rank:pairwise` (separate long and short models)
- 20 features, Z-scored cross-sectionally
- Regularization: `reg_alpha=1.0, reg_lambda=5.0, max_depth=4`
- Reports: aggregate OOS Spearman ρ (long and short) across all folds

---

## Backtest

### [backtest.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/strategy_1030/backtest.py)

Uses the per-fold models on their respective test periods (true OOS simulation):

```
For each test-fold trading day:
  1. Layer A predicts Nifty rest-of-day return
  2. Apply direction filter:
     predicted > +threshold  →  LONG day
     predicted < −threshold  →  SHORT day
     |predicted| < threshold →  SKIP day
  3. Layer B ranks all stocks in the allowed direction
  4. Pick Top-3 stocks
  5. Entry at 10:15 bar close + half-spread slippage
  6. Exit at 15:15 bar close + half-spread slippage
  7. Log trade
```

**Reports**:
- Threshold sensitivity sweep (0%, 0.1%, 0.2%, 0.3%, 0.5%)
- Total trades, win rate, net return, Sharpe, max drawdown, profit factor
- Baseline comparison: Layer A only vs Layer B only vs Combined
- Monthly PnL breakdown

---

## Execution Order

```
Phase 1:  python -m scripts.strategy_1030.data_collection
            → Fetches global indices, builds both datasets from 30-min cache

Phase 2:  python -m scripts.strategy_1030.train
            → Walk-forward training, prints OOS metrics

Phase 3:  python -m scripts.strategy_1030.backtest
            → Runs combined simulation, prints performance report
```

---

## Verification Plan

### Automated
- `data_collection.py` validates dataset_market has ~1100 rows (not 250)
- `data_collection.py` validates dataset_stocks has ~1100 × 170 rows
- `train.py` prints per-fold AND aggregate OOS metrics
- `backtest.py` compares combined vs baselines

### Manual
- Verify zero look-ahead bias: all features use data ≤ 10:15 AM
- Confirm label: `Target = Close_15:15 / Close_10:15 - 1`
- Verify global index features are shifted by 1 day
- Spot-check morning bar extraction matches raw 30-min cache UTC timestamps
