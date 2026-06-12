---
title: "Gating Uplift Certification Report: `v18_random_forest_1h"
type: reference
status: active
model: "Gauntlet Reports"
updated: 2026-06-12
tags: []
---
# 🛡️ Gating Uplift Certification Report: `v18_random_forest_1h`

## 📌 Metadata
- **Daily Model**: `daily_macro_v3` (Run ID: `20260610T144343Z-5f7d069f`)
- **Downstream Model**: `v18_random_forest_1h` (Run ID: `20260610T124108Z-5f7d069f`)
- **Overlap OOS Window**: 374 days (2023-08-01 to 2026-05-29)
- **Friction Cost Level**: 6.0 bps round-trip
- **Primary Selection K**: 3 trades per query bar

## 📊 Gating Performance Details

### ⚙️ Symbol Gating (Top 30% Rank)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1566 | Net Return = -6.37 bps | WR = 47.1%
- **Unfavorable Trades (Blocked)**: Count = 1823 | Net Return = -6.33 bps | WR = 48.8%
- **Performance Uplift**: **`-0.03 bps`**
- **T-Statistic**: `-0.02` (p-value: `0.9859`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 1789 | Net Return = -3.99 bps | WR = 53.9%
- **Unfavorable Trades (Blocked)**: Count = 1687 | Net Return = -7.89 bps | WR = 51.5%
- **Performance Uplift**: **`+3.90 bps`**
- **T-Statistic**: `1.70` (p-value: `0.0884`)
- **Certification Verdict**: **`❌ FAILED`**

### ⚙️ Day Gating (Tercile Market Aggregate)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1821 | Net Return = -5.49 bps | WR = 49.7%
- **Unfavorable Trades (Blocked)**: Count = 1584 | Net Return = -7.10 bps | WR = 48.4%
- **Performance Uplift**: **`+1.61 bps`**
- **T-Statistic**: `0.91` (p-value: `0.3637`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 1569 | Net Return = -5.71 bps | WR = 53.3%
- **Unfavorable Trades (Blocked)**: Count = 1197 | Net Return = -8.27 bps | WR = 50.9%
- **Performance Uplift**: **`+2.56 bps`**
- **T-Statistic**: `0.99` (p-value: `0.3207`)
- **Certification Verdict**: **`❌ FAILED`**

## ⚖️ Integration Verdict
> [!WARNING]
> **Gating Certification has FAILED for v18_random_forest_1h across all modes.** Gating does not provide a statistically significant, cost-adjusted return uplift. Production integration is rejected.

---
*Report generated programmatically via evaluate_v3_gating_sweeps.py.*
