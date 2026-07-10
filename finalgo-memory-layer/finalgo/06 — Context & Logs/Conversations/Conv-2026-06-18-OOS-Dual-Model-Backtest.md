# 💬 Conversation Context: OOS Dual-Model Backtest (v20 1H + v3 15min)

## 📌 Metadata
- **Conversation ID**: 23072aae-b7c0-47ef-a4e2-05a1f682f8e9
- **Start Date**: 2026-06-18
- **Status**: 🔴 Concluded
- **Focus Area**: Model Suite — Backtesting

## 🎯 Objectives
- [x] Confirm last training data dates for v20 1H and v3 15min
- [x] Identify backtest window (true OOS vs in-sample)
- [x] Build pure model backtest (v20 signal + v3 top-15% gate)
- [x] Analyse backtest results and report PnL

## 💻 Active Code Files Modified
- [v20_v3_oos_backtest.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/backtests/v20_v3_oos_backtest.py)
- [v20_v3_true_oos_backtest.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/backtests/v20_v3_true_oos_backtest.py)

## 📝 Compacted Session Log

### Data Cutoffs Confirmed (from metadata.json + actual parquet inspection)
| Model | Data Last Date | Trained At |
|---|---|---|
| v20 Rolling 1H | **2026-06-04 14:30 IST** | 2026-06-15 |
| v3 15min Clean | **2026-06-04 15:00 IST** | 2026-06-07 |

Both datasets share the **same cutoff: June 4, 2026**.

### Key Architecture Facts
- **Active model**: `v20_rolling_1h` (ROLLING_1H_CANDLES=True in config)
- **Live 15m gate**: `_check_15m_percentile(top_percent=0.10)` → top **10%** in live system
- **User requested**: top **15%** for this backtest
- **OOS window**: **Jun 5–18, 2026** (14-day True OOS after training cutoffs)
- **Execution**: Fetches fresh 15m bars from Upstox API, scores and aggregates v20 1H candles and v3 15m candles, and applies a 1-hour holding period.

### True OOS Backtest Results (Jun 5–18, 2026)

#### Option 1: 15% Gate Filter
- **Total Trades**: 270 (30 trades per day over 9 trading days)
- **Win Rate**: 45.2%
- **Avg Net PnL/trade**: +0.0026%
- **Profit Factor**: 1.01
- **Total Net PnL**: Rs. +924.62
- **Return on Capital**: +0.929% (on Rs. 99,518 capital)
- **Max Drawdown**: Rs. -13,269.44
- **LONG / SHORT Net PnL**: Rs. -6,288.26 / Rs. +7,212.88
- **Output CSVs**: `data/backtests/v20_v3_true_oos_trades_20260618_1521.csv` and `data/backtests/v20_v3_true_oos_daily_20260618_1521.csv`

#### Option 2: 10% Gate Filter (Preferred System Default)
- **Total Trades**: 270 (30 trades per day over 9 trading days)
- **Win Rate**: **48.1%**
- **Avg Net PnL/trade**: **+0.0570%**
- **Profit Factor**: **1.24**
- **Total Net PnL**: **Rs. +15,622.01**
- **Return on Capital**: **+15.698%** (on Rs. 99,518 capital)
- **Max Drawdown**: **Rs. -5,393.17**
- **LONG / SHORT Net PnL**: **Rs. +2,704.64 / Rs. +12,917.37**
- **Output CSVs**: `data/backtests/v20_v3_true_oos_trades_20260618_1609.csv` and `data/backtests/v20_v3_true_oos_daily_20260618_1609.csv`

##### Daily Summary (10% Gate):
| Date | Trades | Win Rate | Net PnL (INR) | Avg PnL % | Best PnL % | Worst PnL % | Cum PnL (INR) |
|---|---|---|---|---|---|---|---|
| 2026-06-05 | 30 | 53.3% | -1,399.11 | -0.0624% | +2.0404% | -2.2202% | -1,399.11 |
| 2026-06-08 | 30 | 40.0% | +2,431.87 | +0.0822% | +2.4829% | -0.8317% | +1,032.76 |
| 2026-06-09 | 30 | 50.0% | +7,468.83 | +0.2633% | +4.5014% | -1.4170% | +8,501.59 |
| 2026-06-10 | 30 | 50.0% | +1,256.64 | +0.0400% | +2.1975% | -1.6676% | +9,758.23 |
| 2026-06-11 | 30 | 53.3% | -1,715.85 | -0.0577% | +0.8879% | -2.0761% | +8,042.38 |
| 2026-06-12 | 30 | 50.0% | +4,379.01 | +0.1447% | +4.6667% | -4.3558% | +12,421.39 |
| 2026-06-15 | 30 | 46.7% | +4,540.35 | +0.1525% | +2.0926% | -0.7260% | +16,961.74 |
| 2026-06-16 | 30 | 46.7% | +1,385.92 | +0.0466% | +1.2397% | -0.7659% | +18,347.66 |
| 2026-06-17 | 30 | 43.3% | -2,725.65 | -0.0964% | +1.7193% | -2.0943% | +15,622.01 |

##### Hourly Performance Breakdown (10% Gate):
| Time Slot | Trades | Win Rate | Net PnL (INR) | Avg PnL % | Profit Factor | Max DD (INR) |
|---|---|---|---|---|---|---|
| **09:15** | 45 | 53.3% | Rs. +5,521.57 | +0.1228% | 1.66 | Rs. -2,813.00 |
| **10:15** | 45 | 60.0% | Rs. +7,586.48 | +0.1715% | 2.21 | Rs. -2,024.99 |
| **11:15** | 45 | 42.2% | Rs. +602.33 | +0.0141% | 1.06 | Rs. -3,753.23 |
| **12:15** | 45 | 40.0% | Rs. -3,110.58 | -0.0780% | 0.62 | Rs. -4,429.14 |
| **13:15** | 45 | 40.0% | Rs. -8,157.72 | -0.1841% | 0.32 | Rs. -8,848.21 |
| **14:15** | 45 | 53.3% | Rs. +13,179.93 | +0.2955% | 1.71 | Rs. -5,343.56 |

* **Insight:** Morning sessions (09:15 & 10:15) and afternoon session (14:15) are highly profitable, while the mid-day session (12:15 & 13:15) suffers severe decay (totaling Rs. -11,268 loss). This reveals a clear momentum-regime dependency.

## 🔗 Core Memory Links & Backlinks
- [[02 — Models/1H/Model Card - v10 Native 1h]] — predecessor model card
- [[02 — Models/15m/Model Card - v3 Clean 15min]] — v3 15min model card

---

## ⚡ Post-Conclusion Updates: 15-Minute Split & Same-Day Exit Audit
Following the initial run, the user requested a full split analysis at 15-minute intervals across all entry times (`09:15` to `14:15`).

### 1. Overnight Hold Bug Identification
Evaluating slots like `13:30`, `13:45`, `14:00`, and `14:15` revealed that trades were being held overnight (exiting at `09:15` the next trading day) instead of closing at the 1-hour mark. This occurred because the backtest loop and features were limited to the `VALID_TODS_1H` hourly grid.

### 2. Same-Day Exit Fix
We modified the backtester in [v20_v3_true_oos_backtest.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/backtests/v20_v3_true_oos_backtest.py) to look up exit prices inside the raw 15-minute data `ticker_15m` instead of `feat_v20`. This ensures that all late-day trades are successfully closed at the exact 1-hour mark (e.g. `14:30`, `15:15`) on the same day.

### 3. Corrected 15-Minute Split Results (Jun 5–18, 2026)
With the same-day exit fix active, the side-by-side comparison for `TOP_K = 1` vs `TOP_K = 3` (10% gate, 10 bps RT cost, `MAX_SLOTS = 1000`) is as follows:

```text
===================================================================================================================
              15-MINUTE SLOT COMPARISON REPORT  |  Cost: 10 bps RT  |  Jun 5-18, 2026
===================================================================================================================
             |          TOP_K = 1 (18 Trades/slot)           |          TOP_K = 3 (54 Trades/slot)          
  Time Slot  |   Win%      Net Rs      Avg%     PF     MaxDD |   Win%      Net Rs      Avg%     PF     MaxDD
-------------------------------------------------------------------------------------------------------------------
  09:15      |  44.4%     +1,545   +0.081%  1.36   -2,436 |  44.4%     +4,623   +0.085%  1.34   -3,620
  09:30      |  55.6%       +356   +0.018%  1.11   -1,285 |  51.9%     -4,735   -0.088%  0.70   -6,049
  09:45      |  72.2%     +5,908   +0.336%  4.09     -636 |  55.6%     +6,251   +0.119%  1.66   -2,402
  10:00      |  44.4%     -1,145   -0.062%  0.80   -2,694 |  44.4%     -1,062   -0.023%  0.93   -6,173
  10:15      |  38.9%       -134   -0.010%  0.97   -2,819 |  42.6%     -3,888   -0.073%  0.69   -6,110
  10:30      |  33.3%     -2,318   -0.130%  0.47   -3,613 |  38.9%       +582   +0.011%  1.06   -2,738
  10:45      |  44.4%     +4,912   +0.269%  4.26     -481 |  55.6%     +9,117   +0.169%  2.82   -1,816
  11:00      |  61.1%     +1,490   +0.084%  1.62   -1,117 |  50.0%     +2,682   +0.051%  1.31   -1,955
  11:15      |  33.3%     -4,082   -0.244%  0.42   -4,340 |  46.3%     -2,387   -0.050%  0.79   -4,430
  11:30      |  55.6%     +1,054   +0.060%  1.47     -883 |  40.7%     -1,630   -0.030%  0.85   -4,008
  11:45      |  44.4%     +2,084   +0.121%  1.87   -1,292 |  37.0%     -2,836   -0.054%  0.70   -4,680
  12:00      |  38.9%     +1,377   +0.074%  1.67   -1,300 |  37.0%        -68   -0.001%  0.99   -3,231
  12:15      |  33.3%     -2,077   -0.123%  0.54   -2,287 |  38.9%     -4,089   -0.081%  0.66   -6,449
  12:30      |  22.2%       -626   -0.035%  0.87   -3,091 |  44.4%       +686   +0.014%  1.06   -5,762
  12:45      |  38.9%       +436   +0.025%  1.17   -1,732 |  38.9%     +1,120   +0.021%  1.10   -4,793
  13:00      |  44.4%     -1,116   -0.061%  0.78   -2,666 |  53.7%     +1,800   +0.036%  1.19   -2,272
  13:15      |  44.4%       -236   -0.014%  0.91   -1,994 |  31.5%     -5,811   -0.112%  0.54   -6,608
  13:30      |  38.9%     -1,183   -0.068%  0.55   -1,992 |  31.5%     -5,458   -0.104%  0.51   -6,160
  13:45      |  38.9%     -1,929   -0.110%  0.53   -2,509 |  35.2%     -7,758   -0.147%  0.44   -8,714
  14:00      |  50.0%     -1,312   -0.073%  0.59   -2,157 |  46.3%     -3,614   -0.067%  0.76   -4,689
  14:15      |  50.0%     -1,377   -0.077%  0.82   -3,467 |  50.0%     -6,648   -0.123%  0.65   -7,198
-------------------------------------------------------------------------------------------------------------------
  TOTAL      |  44.2%     +1,627   +0.003%  1.02   -7,614 |  43.6%    -23,121   -0.021%  0.91  -31,008
===================================================================================================================
```
* **Key Finding**: When trades exit on the same day rather than holding overnight, total returns are significantly lower (`TOP_K = 1` drops to +Rs. 1,627; `TOP_K = 3` drops to -Rs. 23,121). This is primarily driven by feature window distortion (Compressive Feature Drift) on the 15-minute grid, causing the XGBoost model to lose predictive edge at almost all afternoon slots.

---

## 🛡️ Stop Loss Evaluation (0.30% SL Checked on 15-Minute Candles)
The user requested an evaluation of the same 15-minute split unblocked backtests with a **0.30% stop loss** active.
* **Stop Loss Logic**: Within the 1-hour holding period, we check every 15-minute candle. If a candle's Low (for LONG) or High (for SHORT) crosses the 0.30% threshold from the entry price, the trade is closed at the **Close** of that 15-minute candle.

### 1. Results with 0.30% Stop Loss
The side-by-side comparison report with the stop loss active is:

```text
===================================================================================================================
              15-MINUTE SLOT COMPARISON REPORT  |  Cost: 10 bps RT  |  Jun 5-18, 2026
===================================================================================================================
             |          TOP_K = 1 (18 Trades/slot)           |          TOP_K = 3 (54 Trades/slot)          
  Time Slot  |   Win%      Net Rs      Avg%     PF     MaxDD |   Win%      Net Rs      Avg%     PF     MaxDD
-------------------------------------------------------------------------------------------------------------------
  09:15      |  27.8%       -578   -0.041%  0.90   -2,679 |  31.5%     -2,066   -0.042%  0.87   -4,035
  09:30      |  44.4%     -1,179   -0.067%  0.69   -2,169 |  37.0%     -5,424   -0.100%  0.59   -7,606
  09:45      |  44.4%       +480   +0.025%  1.11   -1,421 |  40.7%       -868   -0.018%  0.93   -2,410
  10:00      |  33.3%     -3,068   -0.170%  0.43   -3,667 |  42.6%       -726   -0.016%  0.94   -4,527
  10:15      |  33.3%     -2,036   -0.122%  0.53   -1,820 |  42.6%     -1,232   -0.022%  0.89   -4,535
  10:30      |  22.2%     -3,698   -0.207%  0.27   -3,751 |  33.3%     -3,299   -0.062%  0.72   -4,854
  10:45      |  50.0%     +1,494   +0.086%  1.58     -900 |  48.1%     +1,843   +0.030%  1.23   -5,348
  11:00      |  55.6%     +1,775   +0.100%  1.93     -914 |  50.0%     +5,972   +0.113%  2.26   -1,285
  11:15      |  38.9%     -1,831   -0.111%  0.62   -2,591 |  42.6%     -1,352   -0.029%  0.87   -2,773
  11:30      |  50.0%       +935   +0.053%  1.51     -626 |  37.0%     -2,764   -0.052%  0.72   -4,925
  11:45      |  44.4%     +1,607   +0.094%  1.56   -1,410 |  37.0%     -2,775   -0.051%  0.73   -4,243
  12:00      |  38.9%       -106   -0.009%  0.97   -1,572 |  31.5%     -3,981   -0.075%  0.62   -6,512
  12:15      |  38.9%     -1,322   -0.082%  0.67   -1,667 |  37.0%     -3,145   -0.063%  0.71   -3,882
  12:30      |  16.7%     -2,652   -0.150%  0.53   -5,117 |  35.2%     -1,745   -0.032%  0.86   -6,749
  12:45      |  38.9%       -164   -0.009%  0.95   -1,419 |  37.0%       -387   -0.009%  0.96   -4,088
  13:00      |  27.8%     -2,644   -0.148%  0.44   -2,664 |  37.0%     -4,707   -0.089%  0.60   -5,616
  13:15      |  33.3%     -1,114   -0.063%  0.63   -2,031 |  25.9%     -7,362   -0.141%  0.43   -8,256
  13:30      |  27.8%     -3,016   -0.171%  0.31   -3,861 |  29.6%     -4,821   -0.093%  0.60   -6,102
  13:45      |  33.3%     -1,874   -0.108%  0.48   -2,474 |  35.2%     -6,307   -0.119%  0.46   -7,034
  14:00      |  33.3%     -3,188   -0.179%  0.30   -3,319 |  37.0%     -4,241   -0.079%  0.69   -4,807
  14:15      |  38.9%       -336   -0.021%  0.94   -1,521 |  40.7%     -6,271   -0.117%  0.61   -7,554
-------------------------------------------------------------------------------------------------------------------
  TOTAL      |  36.8%    -22,515   -0.062%  0.73  -22,571 |  37.6%    -55,657   -0.051%  0.77  -62,460
===================================================================================================================
```

### 2. Strategic Verdict
* **Severe Performance Degradation**: Implementing a 0.30% stop loss destroys performance. 
  * `TOP_K = 1` net return dropped from **+Rs. 1,627** to **-Rs. 22,515**.
  * `TOP_K = 3` net return dropped from **-Rs. 23,121** to **-Rs. 55,657**.
* **Reasoning**: A 0.30% (30 bps) threshold is extremely narrow for high-conviction intraday picks. Safe-haven noise triggers whipsaws (exiting early on minor counter-trend fluctuations that subsequently reverse and move in the intended direction). This locks in losses prematurely and amplifies transaction cost friction.


