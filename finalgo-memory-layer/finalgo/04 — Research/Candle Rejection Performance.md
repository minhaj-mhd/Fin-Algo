---
title: "Candle Rejection Performance"
type: research
status: wip
updated: 2026-06-18T06:53:33
---
# ⚠️ UNVERIFIED / Exploratory: Candle Rejection & Veto Performance

> [!WARNING]
> This report contains exploratory analysis from scripts/research/analyze_candle_rejections.py.
> Per AGENTS.md, these research scripts hold no verdict authority. Grade metrics are for research only.

## 📊 Rejection / Veto Performance Summary
| Reject Reason | Side | N | Mean Net (bps) | Median Net (bps) | SL-Hit Rate | t-stat | 95% Bootstrap CI | Guard Value | Significant? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CONFIRMED_ENTRY | LONG | 2 | +87.75 | +87.75 | 0.0% | +2.44 | [+51.73, +123.78] | N/A | Yes |
| LIMIT_EXPIRED | LONG | 1 | -93.33 | -93.33 | 100.0% | +0.00 | [-93.33, -93.33] | +93.33 bps | No |
| THRUST_VETO | SHORT | 1 | -87.92 | -87.92 | 100.0% | +0.00 | [-87.92, -87.92] | +87.92 bps | No |

## ⚔️ Strategy Comparison: Fade vs Market (Limit Retracement)
Comparing retracement limit strategy (A) vs immediate market entry (B) on signals that failed immediate look-back confirmation.
- **Number of retracement signals**: 2
- **Fade Strategy Mean Net**: 61.89 bps
- **Market Strategy Mean Net**: -1.67 bps
- **Mean Paired Difference (A - B)**: +63.56 bps (Median: +63.56 bps)
- **t-statistic**: +2.13
- **95% Bootstrap CI of Difference**: [+33.78, +93.33]
- **Is Fade significantly better than Market?**: Yes

## 🔍 Confirmation Value Analysis
Does immediate candle direction confirmation select higher-performing entries than the baseline average of all signals?
- **Confirmed Entry Mean Net**: 51.73 bps (N=1)
- **All Signals Baseline Mean Net**: -9.88 bps (N=4)
- **Confirmation Edge (Confirmed - All)**: +61.61 bps
