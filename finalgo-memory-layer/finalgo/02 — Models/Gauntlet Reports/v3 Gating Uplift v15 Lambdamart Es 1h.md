# 🛡️ Gating Uplift Certification Report: `v15_lambdamart_es_1h`

## 📌 Metadata
- **Daily Model**: `daily_macro_v3` (Run ID: `20260610T144343Z-5f7d069f`)
- **Downstream Model**: `v15_lambdamart_es_1h` (Run ID: `20260610T121241Z-5f7d069f`)
- **Overlap OOS Window**: 374 days (2023-08-01 to 2026-05-29)
- **Friction Cost Level**: 6.0 bps round-trip
- **Primary Selection K**: 3 trades per query bar

## 📊 Gating Performance Details

### ⚙️ Symbol Gating (Top 30% Rank)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1604 | Net Return = -2.30 bps | WR = 49.9%
- **Unfavorable Trades (Blocked)**: Count = 1728 | Net Return = -6.00 bps | WR = 50.3%
- **Performance Uplift**: **`+3.70 bps`**
- **T-Statistic**: `1.47` (p-value: `0.1410`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 2431 | Net Return = -4.09 bps | WR = 52.9%
- **Unfavorable Trades (Blocked)**: Count = 1157 | Net Return = -3.44 bps | WR = 53.8%
- **Performance Uplift**: **`-0.64 bps`**
- **T-Statistic**: `-0.22` (p-value: `0.8294`)
- **Certification Verdict**: **`❌ FAILED`**

### ⚙️ Day Gating (Tercile Market Aggregate)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1821 | Net Return = -5.10 bps | WR = 50.2%
- **Unfavorable Trades (Blocked)**: Count = 1584 | Net Return = -6.22 bps | WR = 50.4%
- **Performance Uplift**: **`+1.12 bps`**
- **T-Statistic**: `0.47` (p-value: `0.6351`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 1569 | Net Return = -4.01 bps | WR = 52.2%
- **Unfavorable Trades (Blocked)**: Count = 1197 | Net Return = -5.36 bps | WR = 53.4%
- **Performance Uplift**: **`+1.35 bps`**
- **T-Statistic**: `0.40` (p-value: `0.6926`)
- **Certification Verdict**: **`❌ FAILED`**

## ⚖️ Integration Verdict
> [!WARNING]
> **Gating Certification has FAILED for v15_lambdamart_es_1h across all modes.** Gating does not provide a statistically significant, cost-adjusted return uplift. Production integration is rejected.

---
*Report generated programmatically via evaluate_v3_gating_sweeps.py.*
