# 🛡️ Gating Uplift Certification Report: `v11_utility_1h`

## 📌 Metadata
- **Daily Model**: `daily_macro_v3` (Run ID: `20260610T144343Z-5f7d069f`)
- **Downstream Model**: `v11_utility_1h` (Run ID: `20260610T111055Z-5f7d069f`)
- **Overlap OOS Window**: 374 days (2023-08-01 to 2026-05-29)
- **Friction Cost Level**: 6.0 bps round-trip
- **Primary Selection K**: 3 trades per query bar

## 📊 Gating Performance Details

### ⚙️ Symbol Gating (Top 30% Rank)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1667 | Net Return = -3.54 bps | WR = 51.5%
- **Unfavorable Trades (Blocked)**: Count = 1700 | Net Return = -4.01 bps | WR = 52.3%
- **Performance Uplift**: **`+0.47 bps`**
- **T-Statistic**: `0.22` (p-value: `0.8236`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 2543 | Net Return = -1.73 bps | WR = 53.8%
- **Unfavorable Trades (Blocked)**: Count = 1058 | Net Return = -2.68 bps | WR = 55.6%
- **Performance Uplift**: **`+0.95 bps`**
- **T-Statistic**: `0.28` (p-value: `0.7778`)
- **Certification Verdict**: **`❌ FAILED`**

### ⚙️ Day Gating (Tercile Market Aggregate)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1821 | Net Return = -1.00 bps | WR = 54.1%
- **Unfavorable Trades (Blocked)**: Count = 1584 | Net Return = -6.06 bps | WR = 50.2%
- **Performance Uplift**: **`+5.07 bps`**
- **T-Statistic**: `2.47` (p-value: `0.0134`)
- **Certification Verdict**: **`✅ PASSED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 1569 | Net Return = -2.92 bps | WR = 55.3%
- **Unfavorable Trades (Blocked)**: Count = 1197 | Net Return = -2.22 bps | WR = 54.3%
- **Performance Uplift**: **`-0.70 bps`**
- **T-Statistic**: `-0.19` (p-value: `0.8476`)
- **Certification Verdict**: **`❌ FAILED`**

## ⚖️ Integration Verdict
> [!WARNING]
> **Gating Certification has FAILED for v11_utility_1h across all modes.** Gating does not provide a statistically significant, cost-adjusted return uplift. Production integration is rejected.

---
*Report generated programmatically via evaluate_v3_gating_sweeps.py.*
