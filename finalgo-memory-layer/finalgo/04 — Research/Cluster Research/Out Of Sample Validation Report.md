---
title: "Out-of-Sample Validation Report: Veto Override Clustering"
type: report
status: concluded
updated: 2026-06-12
tags: []
---
# Out-of-Sample Validation Report: Veto Override Clustering

This report summarizes the performance of the veto override clustering strategy evaluated under a strict chronological train/test split. 
- **Training Set (In-Sample)**: First 1,000 trades (from 2026-05-12T10:15:45.706405 to 2026-05-25T14:18:05.240355)
- **Validation Set (Out-of-Sample)**: Last 466 trades (from 2026-05-25T14:18:17.020211 to 2026-06-03T15:02:55.532701)
- **Leverage**: 5x | **Slippage**: 0.06% per trade

---

## 1. High-Level Comparison (In-Sample vs. Out-of-Sample)

| Strategy Configuration | Dataset | Matched Trades | Catch Rate (%) | Win Rate (%) | Avg PnL (%) | Cumulative Net Return (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (All Trades)** | Train (In-Sample) | 1,000 | 100.0% | 50.90% | 0.0281% | N/A |
| | Val (Out-of-Sample) | 466 | 100.0% | 46.57% | -0.0294% | N/A |
| **Path 1: Fixed Researched Sweet Spots** | Train (In-Sample) | 206 | 20.6% | 50.97% | 0.1035% | 94.29% |
| | Val (Out-of-Sample) | 51 | 10.9% | 43.14% | -0.1545% | -42.47% |
| **Path 2: Dynamically Discovered Sweet Spots** | Train (In-Sample) | 202 | 20.2% | 62.87% | 0.1602% | 149.67% |
| | Val (Out-of-Sample) | 23 | 4.9% | 52.17% | -0.0040% | -1.84% |

---

## 2. Individual Sweet Spot Breakdown (Out-of-Sample Performance)

### Path 1: Fixed Researched Sweet Spots

| Sweet Spot Name | Threshold | Train Caught | Train WR | Train PnL | Val Caught | Val WR | Val PnL | Val Net Return | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Cluster 0 Sub 1 (K=3, dist < 0.75) | 0.75 | 7 | 71.4% | 0.4817% | 0 | 0.0% | 0.0000% | +0.00% | ⚪ No Match |
| Cluster 1 Sub 1 (K=3, dist < 0.50) | 0.50 | 37 | 59.5% | 0.1015% | 3 | 33.3% | 0.0773% | +0.98% | 🔴 Overfit |
| Cluster 2 Sub 0 (K=3, dist < 0.50) | 0.50 | 26 | 50.0% | 0.1087% | 8 | 62.5% | 0.1342% | +4.89% | 🟢 Robust |
| Cluster 2 Sub 1 (K=3, dist < 1.00) | 1.00 | 12 | 58.3% | 0.2863% | 3 | 66.7% | -0.4459% | -6.87% | 🔴 Overfit |
| Cluster 3 Sub 0 (K=5, dist < 0.75) | 0.75 | 124 | 46.8% | 0.0640% | 37 | 37.8% | -0.2121% | -41.46% | 🔴 Overfit |

### Path 2: Dynamically Discovered Sweet Spots

| Discovered Sweet Spot | Threshold | Train Caught | Train WR | Train PnL | Val Caught | Val WR | Val PnL | Val Net Return | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Cluster 0 Sub 0 (K=3, dist < 1.00) | 1.00 | 38 | 60.5% | 0.1457% | 0 | 0.0% | 0.0000% | +0.00% | ⚪ No Match |
| Cluster 0 Sub 1 (K=3, dist < 1.00) | 1.00 | 9 | 66.7% | 0.3973% | 0 | 0.0% | 0.0000% | +0.00% | ⚪ No Match |
| Cluster 1 Sub 0 (K=3, dist < 1.50) | 1.50 | 15 | 60.0% | 0.1945% | 1 | 100.0% | 0.3337% | +1.61% | 🟢 Robust |
| Cluster 1 Sub 1 (K=3, dist < 0.75) | 0.75 | 87 | 63.2% | 0.1154% | 11 | 63.6% | 0.0881% | +4.18% | 🟢 Robust |
| Cluster 3 Sub 1 (K=5, dist < 0.75) | 0.75 | 16 | 68.8% | 0.3519% | 5 | 40.0% | -0.2034% | -5.38% | 🔴 Overfit |
| Cluster 3 Sub 2 (K=5, dist < 0.50) | 0.50 | 37 | 62.2% | 0.1260% | 6 | 33.3% | -0.0629% | -2.25% | 🔴 Overfit |

## 3. Conclusions and Next Steps

> [!CAUTION]
> **Validation FAILED (Overfitting Warning)**: The out-of-sample win rate collapsed or PnL became negative.
> - Fixed configuration out-of-sample win rate: **43.14%** (Avg PnL: **-0.1545%**)
> - Dynamic configuration out-of-sample win rate: **52.17%** (Avg PnL: **-0.0040%**)
> This strongly indicates that the K-Means clustering configurations are overfitting to historical noise. **We recommend NOT using this strategy in live trading without modifying features or tightening quality gates.**
