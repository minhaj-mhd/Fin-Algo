---
title: "Gating Uplift Certification Report: `v10_native_1h"
type: reference
status: active
model: "Gauntlet Reports"
updated: 2026-06-12
tags: []
---
# 🛡️ Gating Uplift Certification Report: `v10_native_1h`

## 📌 Metadata
- **Daily Model**: `daily_macro_v3` (Run ID: `20260610T144343Z-5f7d069f`)
- **Downstream Model**: `v10_native_1h` (Run ID: `20260610T110725Z-5f7d069f`)
- **Overlap OOS Window**: 374 days (2023-08-01 to 2026-05-29)
- **Friction Cost Level**: 6.0 bps round-trip
- **Primary Selection K**: 3 trades per query bar

## 📊 Gating Performance Details

### ⚙️ Symbol Gating (Top 30% Rank)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1684 | Net Return = +0.21 bps | WR = 51.5%
- **Unfavorable Trades (Blocked)**: Count = 1711 | Net Return = -3.90 bps | WR = 52.1%
- **Performance Uplift**: **`+4.11 bps`**
- **T-Statistic**: `1.78` (p-value: `0.0759`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 2456 | Net Return = -1.73 bps | WR = 54.2%
- **Unfavorable Trades (Blocked)**: Count = 1094 | Net Return = -3.53 bps | WR = 55.4%
- **Performance Uplift**: **`+1.80 bps`**
- **T-Statistic**: `0.57` (p-value: `0.5717`)
- **Certification Verdict**: **`❌ FAILED`**

### ⚙️ Day Gating (Tercile Market Aggregate)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 1821 | Net Return = -1.91 bps | WR = 52.8%
- **Unfavorable Trades (Blocked)**: Count = 1584 | Net Return = -2.93 bps | WR = 50.9%
- **Performance Uplift**: **`+1.01 bps`**
- **T-Statistic**: `0.45` (p-value: `0.6540`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 1569 | Net Return = -3.98 bps | WR = 53.5%
- **Unfavorable Trades (Blocked)**: Count = 1197 | Net Return = -4.45 bps | WR = 52.9%
- **Performance Uplift**: **`+0.47 bps`**
- **T-Statistic**: `0.14` (p-value: `0.8921`)
- **Certification Verdict**: **`❌ FAILED`**

## ⚖️ Integration Verdict
> [!WARNING]
> **Gating Certification has FAILED for v10_native_1h across all modes.** Gating does not provide a statistically significant, cost-adjusted return uplift. Production integration is rejected.

---
*Report generated programmatically via evaluate_v3_gating_sweeps.py.*
