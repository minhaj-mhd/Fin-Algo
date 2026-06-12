---
title: "Gating Uplift Certification Report: `v10_depth4_1h"
type: reference
status: dead
model: "Gauntlet Reports"
updated: 2026-06-12
tags: []
---
# 🛡️ Gating Uplift Certification Report: `v10_depth4_1h`

## 📌 Metadata
- **Daily Model**: `daily_macro_v3` (Run ID: `20260610T144343Z-5f7d069f`)
- **Downstream Model**: `v10_depth4_1h` (Run ID: `20260610T110350Z-5f7d069f`)
- **Overlap OOS Window**: 374 days (2023-08-01 to 2026-05-29)
- **Friction Cost Level**: 6.0 bps round-trip
- **Primary Selection K**: 3 trades per query bar

## 📊 Gating Performance Details

### ⚙️ Symbol Gating (Top 30% Rank)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1540 | Net Return = -1.27 bps | WR = 53.3%
- **Unfavorable Trades (Blocked)**: Count = 1792 | Net Return = -2.55 bps | WR = 52.5%
- **Performance Uplift**: **`+1.28 bps`**
- **T-Statistic**: `0.66` (p-value: `0.5100`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 2520 | Net Return = -3.05 bps | WR = 55.0%
- **Unfavorable Trades (Blocked)**: Count = 1072 | Net Return = +1.18 bps | WR = 55.7%
- **Performance Uplift**: **`-4.23 bps`**
- **T-Statistic**: `-1.31` (p-value: `0.1910`)
- **Certification Verdict**: **`❌ FAILED`**

### ⚙️ Day Gating (Tercile Market Aggregate)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1821 | Net Return = -0.95 bps | WR = 53.9%
- **Unfavorable Trades (Blocked)**: Count = 1584 | Net Return = -5.34 bps | WR = 51.2%
- **Performance Uplift**: **`+4.39 bps`**
- **T-Statistic**: `2.37` (p-value: `0.0180`)
- **Certification Verdict**: **`✅ PASSED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 1569 | Net Return = -1.12 bps | WR = 54.5%
- **Unfavorable Trades (Blocked)**: Count = 1197 | Net Return = -7.26 bps | WR = 53.2%
- **Performance Uplift**: **`+6.14 bps`**
- **T-Statistic**: `1.63` (p-value: `0.1032`)
- **Certification Verdict**: **`❌ FAILED`**

## ⚖️ Integration Verdict
> [!WARNING]
> **Gating Certification has FAILED for v10_depth4_1h across all modes.** Gating does not provide a statistically significant, cost-adjusted return uplift. Production integration is rejected.

---
*Report generated programmatically via evaluate_v3_gating_sweeps.py.*
