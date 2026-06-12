# Walk-Forward Validation Report: Veto Override Clustering

This report presents a rigorous **rolling walk-forward backtest** of the veto override clustering strategy. 
To represent a realistic production deployment with daily retraining, we evaluate our clustering strategy using a **8-day rolling training window** and test on the **next 1 day**.

- **Total Trading Days**: 15
- **Rolling Splits**: 7 (Testing on days 9 through 15)
- **Leverage**: 5x | **Slippage**: 0.06% per trade

---

## 1. Walk-Forward Cumulative Results (Out-of-Sample)

| Strategy Configuration | Dataset | Matched Trades | Catch Rate (%) | Win Rate (%) | Avg PnL (%) | Cumulative Net Return (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (All Test Trades)** | Walk-Forward | 620 | 100.0% | 48.06% | -0.0284% | N/A |
| **Path 1: Fixed Researched Sweet Spots** | Walk-Forward | 54 | 8.7% | 53.70% | 0.0401% | +7.58% |
| **Path 2: Dynamically Discovered Sweet Spots** | Walk-Forward | 78 | 12.6% | 51.28% | -0.0496% | -24.03% |

---

## 2. Daily Walk-Forward Split Breakdown

| Split | Test Date | Test Size | Baseline WR | Baseline PnL | Fixed Overrides | Fixed WR | Fixed PnL | Fixed Net Return | Disc. Overrides | Disc. WR | Disc. PnL | Disc. Net Return |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2026-05-25 | 181 | 52.5% | -0.0228% | 41 | 58.5% | 0.0481% | +7.41% | 70 | 54.3% | -0.0504% | -21.84% |
| 2 | 2026-05-26 | 75 | 38.7% | -0.0661% | 11 | 36.4% | -0.0297% | -2.29% | 6 | 16.7% | -0.1050% | -3.51% |
| 3 | 2026-05-27 | 71 | 50.7% | -0.0871% | 1 | 0.0% | -0.2469% | -1.29% | 2 | 50.0% | 0.1439% | +1.32% |
| 4 | 2026-05-29 | 68 | 36.8% | -0.1328% | 1 | 100.0% | 0.7640% | +3.76% | 0 | N/A | N/A | 0.00% |
| 5 | 2026-06-01 | 68 | 39.7% | -0.0270% | 0 | N/A | N/A | 0.00% | 0 | N/A | N/A | 0.00% |
| 6 | 2026-06-02 | 87 | 55.2% | 0.0650% | 0 | N/A | N/A | 0.00% | 0 | N/A | N/A | 0.00% |
| 7 | 2026-06-03 | 70 | 54.3% | 0.0407% | 0 | N/A | N/A | 0.00% | 0 | N/A | N/A | 0.00% |

## 3. Findings and Key Insights

> [!WARNING]
> **Walk-Forward FAILED (Overfitting Confirmed)**:
> Under rolling walk-forward conditions, the strategy's win rates and returns collapse:
> - Fixed Sweet Spots WR: **53.70%** (Avg PnL: **0.0401%** | Net Return: **+7.58%**)
> - Dynamically Discovered WR: **51.28%** (Avg PnL: **-0.0496%** | Net Return: **-24.03%**)
> This confirms that the current 4-feature clustering configuration does not generalize well across sliding chronological windows.

## 4. Next Steps: Feature Expansion (Path 4)
Because the current 4 features do not provide robust walk-forward performance, we must proceed to Feature Expansion (Path 4). Refer to [Feature_Set_Analysis.md](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/07.%20Cluster%20Research/Feature_Set_Analysis.md) for candidates.
