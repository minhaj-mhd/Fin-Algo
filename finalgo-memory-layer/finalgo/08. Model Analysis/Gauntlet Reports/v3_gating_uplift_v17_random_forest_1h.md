# 🛡️ Gating Uplift Certification Report: `v17_random_forest_1h`

## 📌 Metadata
- **Daily Model**: `daily_macro_v3` (Run ID: `20260610T144343Z-5f7d069f`)
- **Downstream Model**: `v17_random_forest_1h` (Run ID: `20260610T121944Z-5f7d069f`)
- **Overlap OOS Window**: 374 days (2023-08-01 to 2026-05-29)
- **Friction Cost Level**: 6.0 bps round-trip
- **Primary Selection K**: 3 trades per query bar

## 📊 Gating Performance Details

### ⚙️ Symbol Gating (Top 30% Rank)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1654 | Net Return = -4.43 bps | WR = 49.3%
- **Unfavorable Trades (Blocked)**: Count = 1771 | Net Return = -4.66 bps | WR = 49.4%
- **Performance Uplift**: **`+0.22 bps`**
- **T-Statistic**: `0.08` (p-value: `0.9387`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 2528 | Net Return = -3.70 bps | WR = 52.9%
- **Unfavorable Trades (Blocked)**: Count = 1082 | Net Return = -6.79 bps | WR = 50.2%
- **Performance Uplift**: **`+3.09 bps`**
- **T-Statistic**: `0.86` (p-value: `0.3922`)
- **Certification Verdict**: **`❌ FAILED`**

### ⚙️ Day Gating (Tercile Market Aggregate)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1821 | Net Return = -3.76 bps | WR = 50.2%
- **Unfavorable Trades (Blocked)**: Count = 1584 | Net Return = -7.20 bps | WR = 48.0%
- **Performance Uplift**: **`+3.44 bps`**
- **T-Statistic**: `1.20` (p-value: `0.2301`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 1569 | Net Return = -3.92 bps | WR = 52.5%
- **Unfavorable Trades (Blocked)**: Count = 1197 | Net Return = -9.00 bps | WR = 50.2%
- **Performance Uplift**: **`+5.08 bps`**
- **T-Statistic**: `1.23` (p-value: `0.2189`)
- **Certification Verdict**: **`❌ FAILED`**

## ⚖️ Integration Verdict
> [!WARNING]
> **Gating Certification has FAILED for v17_random_forest_1h across all modes.** Gating does not provide a statistically significant, cost-adjusted return uplift. Production integration is rejected.

---
*Report generated programmatically via evaluate_v3_gating_sweeps.py.*
