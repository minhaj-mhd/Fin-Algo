# 🛡️ Gating Uplift Certification Report: `v12_lambdamart_1h`

## 📌 Metadata
- **Daily Model**: `daily_macro_v3` (Run ID: `20260610T144343Z-5f7d069f`)
- **Downstream Model**: `v12_lambdamart_1h` (Run ID: `20260610T130912Z-5f7d069f`)
- **Overlap OOS Window**: 374 days (2023-08-01 to 2026-05-29)
- **Friction Cost Level**: 6.0 bps round-trip
- **Primary Selection K**: 3 trades per query bar

## 📊 Gating Performance Details

### ⚙️ Symbol Gating (Top 30% Rank)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1596 | Net Return = -5.68 bps | WR = 48.7%
- **Unfavorable Trades (Blocked)**: Count = 1734 | Net Return = -3.97 bps | WR = 50.3%
- **Performance Uplift**: **`-1.71 bps`**
- **T-Statistic**: `-0.63` (p-value: `0.5295`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 2327 | Net Return = +0.24 bps | WR = 55.6%
- **Unfavorable Trades (Blocked)**: Count = 1165 | Net Return = -4.60 bps | WR = 54.1%
- **Performance Uplift**: **`+4.84 bps`**
- **T-Statistic**: `1.46` (p-value: `0.1452`)
- **Certification Verdict**: **`❌ FAILED`**

### ⚙️ Day Gating (Tercile Market Aggregate)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1821 | Net Return = -4.11 bps | WR = 50.3%
- **Unfavorable Trades (Blocked)**: Count = 1584 | Net Return = -7.22 bps | WR = 47.2%
- **Performance Uplift**: **`+3.11 bps`**
- **T-Statistic**: `1.27` (p-value: `0.2039`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 1569 | Net Return = -3.34 bps | WR = 53.7%
- **Unfavorable Trades (Blocked)**: Count = 1197 | Net Return = -5.53 bps | WR = 51.9%
- **Performance Uplift**: **`+2.18 bps`**
- **T-Statistic**: `0.58` (p-value: `0.5625`)
- **Certification Verdict**: **`❌ FAILED`**

## ⚖️ Integration Verdict
> [!WARNING]
> **Gating Certification has FAILED for v12_lambdamart_1h across all modes.** Gating does not provide a statistically significant, cost-adjusted return uplift. Production integration is rejected.

---
*Report generated programmatically via evaluate_v3_gating_sweeps.py.*
