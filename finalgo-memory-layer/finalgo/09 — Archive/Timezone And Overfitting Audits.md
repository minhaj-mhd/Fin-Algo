---
title: "Timezone & Overfitting Audits (Dec 5, 2024)"
type: archive
status: archived
updated: 2026-06-12
tags: []
---
# 🛡️ Timezone & Overfitting Audits (Dec 5, 2024)

### The Timezone Offset Bug
- **The Issue**: A severeTimeZone offset was discovered between the training environment and backtester inference.
  - **Training**: `prepare_ranking_data.py` processed yfinance candles in **UTC**, where market hours mapped to `[3, 4, 5, 6, 7, 8, 9]`.
  - **Backtesting**: `analyze_historical_day.py` converted timestamps to **IST**, resulting in hours `[9, 10, 11, 12, 13, 14, 15]`.
- **The Impact**: A 6-hour offset in the `Hour` feature. The model interpreted a 9 AM IST market open as if it were a 3 PM EOD close, leading to incorrect tactical execution during backtesting.
- **The Fix**: Removed all UTC-to-IST timezone conversions during backtest feature generation, aligning both inference and training to use raw UTC hours.

### The V1 Overfitting Audit
- **The Issue**: Model showed a training Spearman correlation of **28.47%** vs a test correlation of **3.76%** (a massive **7.6x overfitting gap**).
- **Identified Root Causes & Resolutions**:
  1. *Random Train/Test Split*: The original trainer split queries randomly instead of temporally. Sandwiched queries caused severe data leakage. *Resolved by implementing a strict 80/20 temporal split.*
  2. *Weak Regularization*: `max_depth=6` was too high, allowing memorization of individual query IDs. *Resolved by lowering depth to 3-4 and adding L1/L2 penalties (`alpha`, `lambda`).*
  3. *Cumulative Features*: Cumulative volume (OBV) carried baseline context dependencies across years. *Resolved by migrating to rate-of-change volume and microstructure signals.*
