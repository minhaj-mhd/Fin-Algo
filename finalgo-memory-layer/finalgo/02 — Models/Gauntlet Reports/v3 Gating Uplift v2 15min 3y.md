---
title: "Gating Uplift Certification Report: `v2_15min_3y"
type: reference
status: dead
model: "Gauntlet Reports"
updated: 2026-06-12
tags: []
---
# 🛡️ Gating Uplift Certification Report: `v2_15min_3y`

## 📌 Metadata
- **Daily Model**: `daily_macro_v3` (Run ID: `20260610T144343Z-5f7d069f`)
- **Downstream Model**: `v2_15min_3y` (Run ID: `20260610T095723Z-5f7d069f`)
- **Overlap OOS Window**: 220 days (2024-09-02 to 2026-06-04)
- **Friction Cost Level**: 6.0 bps round-trip
- **Primary Selection K**: 3 trades per query bar

## 📊 Gating Performance Details

### ⚙️ Symbol Gating (Top 30% Rank)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 4524 | Net Return = -3.42 bps | WR = 54.1%
- **Unfavorable Trades (Blocked)**: Count = 4616 | Net Return = -2.26 bps | WR = 55.0%
- **Performance Uplift**: **`-1.16 bps`**
- **T-Statistic**: `-1.43` (p-value: `0.1526`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 6587 | Net Return = -1.86 bps | WR = 57.5%
- **Unfavorable Trades (Blocked)**: Count = 3291 | Net Return = -3.33 bps | WR = 56.1%
- **Performance Uplift**: **`+1.47 bps`**
- **T-Statistic**: `1.39` (p-value: `0.1638`)
- **Certification Verdict**: **`❌ FAILED`**

### ⚙️ Day Gating (Tercile Market Aggregate)

#### Downstream LONG Trades:
- **Favorable Trades (Gated)**: Count = 5322 | Net Return = -2.33 bps | WR = 54.6%
- **Unfavorable Trades (Blocked)**: Count = 5118 | Net Return = -2.93 bps | WR = 55.1%
- **Performance Uplift**: **`+0.60 bps`**
- **T-Statistic**: `0.83` (p-value: `0.4069`)
- **Certification Verdict**: **`❌ FAILED`**

#### Downstream SHORT Trades:
- **Favorable Trades (Gated)**: Count = 5253 | Net Return = -3.36 bps | WR = 56.5%
- **Unfavorable Trades (Blocked)**: Count = 5118 | Net Return = -3.41 bps | WR = 55.9%
- **Performance Uplift**: **`+0.06 bps`**
- **T-Statistic**: `0.06` (p-value: `0.9537`)
- **Certification Verdict**: **`❌ FAILED`**

## ⚖️ Integration Verdict
> [!WARNING]
> **Gating Certification has FAILED for v2_15min_3y across all modes.** Gating does not provide a statistically significant, cost-adjusted return uplift. Production integration is rejected.

---
*Report generated programmatically via evaluate_v3_gating_sweeps.py.*
