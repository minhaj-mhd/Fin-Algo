# 🏛️ Daily Gatekeeper V2 — Rebuild & Certification Report (D0–D6)

> **Status**: 🔴 CONCLUDED — Gating Certification FAILED (V2 & V3 Gating Rejected)
> **Date**: 2026-06-10
> **Authors**: Antigravity & Claude
> **Pre-registered Run ID**: `20260610T135608Z-5f7d069f` (V2) & `20260610T144343Z-5f7d069f` (V3)
> **Linked Specs**: [[02. Model Suite/Daily Gatekeeper V2 Rebuild Plan]], [[01. Core Architecture/Validation Gauntlet Architecture]]

---

## 1. Executive Summary

This report documents the end-to-end execution of the **Daily Gatekeeper V2 Rebuild and Gating Certification Plan**. We successfully built a 10-year point-in-time daily macro dataset, trained a rectified XGBoost ranker (`daily_macro_v2`), and evaluated it under the statistics-corrected **Gauntlet Criteria v2** (incorporating a magnitude-based filter alternative and a 24-month recent window).

While the standalone model achieved **TRIGGER_GRADE** (Long) and **FILTER_GRADE** (Short) standing in isolation, the downstream out-of-sample gating certification **failed** to show a statistically significant positive net return uplift ($t \ge 2.0$) on either 1-hour or 15-minute models. Consequently, we recommend **against** integrating the daily gatekeeper as a live execution gate in the production engine.

---

## 2. Dataset Construction & Timing Contract (D0–D2)

The daily dataset was rebuilt from the ground up to incorporate macro, breadth, and sector relative strength indicators spanning **10 years (2016-2026)** to purge noise and capture multi-regime properties.

* **Decision-Time Contract**: Every daily feature in row $T$ is aligned strictly using the last available value prior to **09:00 IST on trade day T** (pre-open).
* **Point-in-Time Joins**:
  - India equities, sector indices, and India VIX are lagged to $T-1$ 15:30 close.
  - US close data (S&P 500, NASDAQ, DXY, US 10Y Treasury) is aligned based on the US calendar date $T-1$ closing value, which becomes available in India by 02:30 IST on trade day $T$.
  - Asian indices (Nikkei, HSI) are locked at $T-1$ close (no morning session leakage).
* **Labeling**: 3-bar forward close-to-close returns:
  $$\text{Label} = \frac{\text{Close}_{T+2}}{\text{Close}_{T-1}} - 1.0$$
* **Verification**: Lookahead leakage tests in `tests/test_daily_macro_dataset.py` passed with 100% conformance, verifying that no feature in row $T$ matches or correlates 1.0 with any forward data timestamped after 09:00 IST on day $T$.

---

## 3. Standalone Model Training & Gauntlet Certification (D3–D4)

### A. Training Architecture
* **Algorithm**: XGBoost `rank:pairwise` (dual long/short models via label inversion).
* **Regularization**: Deeply regularized to prevent noise fitting ($\eta = 0.01$, $\text{max\_depth} = 5$, $\alpha = 2.0$, $\lambda = 4.0$, $\text{min\_child\_weight} = 40$).
* **Folds Configuration**: 4 temporal walk-forward folds (val size = 6 months, test size = 6 months).

### B. Validation Gauntlet v2 Verdict
Evaluated under Criteria v2 over 26 walk-forward test folds (10-year span):

| Metric | LONG Model | SHORT Model | Status / Verdict |
| :--- | :---: | :---: | :--- |
| **Walk-Forward Avg Rho** | $+0.0295$ | $+0.0350$ | Significant ($p < 0.02$) |
| **Top-3 Win Rate (K=3)** | **$56.8\%$** | **$53.2\%$** | Meets Criteria v2 |
| **Top-3 Avg Return** | $+40.8$ bps/day | $+15.7$ bps/day | Raw Outperformance |
| **Expected Return Edge**| **$+4.08$ bps/day** | **$+1.57$ bps/day**| Outperforms Universe Mean |
| **Final Gauntlet Stamp** | **`TRIGGER_GRADE`** | **`FILTER_GRADE`** | **PASSED Certification** |

The model's registry signature was successfully computed, verified, and stamped into `models/daily_macro_v2/metadata.json` under signature hash `0b6aef2421074f7cbc8b34ce313af41570388989e4d6d7a418a5d2d1d4bdb899`.

---

## 4. Downstream Gating Uplift Certification (D5)

We evaluated the performance uplift of gating downstream out-of-sample trades using `daily_macro_v2` predictions. The analysis joined pre-open daily sentiment forecasts at $T-1$ with downstream $1\text{H}$ and $15\text{M}$ trades executed on day $T$ (cost adjusted to 6 bps).

### A. Dynamic Resolving & Timezone Fixes
During development, we identified and hotfixed two issues in the uplift certification harness:
1. **Downstream Label Bug**: Stripped a legacy assumption that hardcoded `Return` (the z-scored input feature) as the target. The harness now dynamically resolves the correct raw forward target labels (e.g. `Next_Hour_Return` for 1H and `Next_15Min_Return` for 15M) from the Gauntlet central registry.
2. **Timezone Typemismatch**: Localized all Datetime indexes to tz-naive format (`dt.tz_localize(None)`) to prevent pandas comparison TypeErrors when joining intraday (UTC/IST hybrid) and daily (naive) date files.

### B. Gating Uplift Performance Matrix

Under pre-registered uplift criteria, gating must achieve **uplift $\ge +2.0$ bps net with $t$-statistic $\ge 2.0$**:

```
Mode: Day Gating (Terclie-split daily sentiment, 374 overlap trading days)
-------------------------------------------------------------------------
Downstream Model: v8_upstox_3y (1-Hour)
- LONG Gated Trades  : Fav = 1,851 | Unfav = 2,214
  Favorable Net Return: -1.05 bps | Unfavorable: -4.24 bps
  Net Uplift         : +3.19 bps (95% CI: [-0.89, +7.38] bps)
  T-statistic        : 1.54 (p-value: 0.1234)                   --> [FAILED]
- SHORT Gated Trades : Fav = 2,244 | Unfav = 1,137
  Favorable Net Return: -3.13 bps | Unfavorable: -2.28 bps
  Net Uplift         : -0.85 bps (95% CI: [-6.91, +5.26] bps)
  T-statistic        : -0.28 (p-value: 0.7802)                  --> [FAILED]

Mode: Symbol Gating (Top 30% daily-rank symbol filter)
-------------------------------------------------------------------------
Downstream Model: v8_upstox_3y (1-Hour)
- LONG Gated Trades  : Fav = 1,529 | Unfav = 1,782
  Favorable Net Return: -2.41 bps | Unfavorable: -3.99 bps
  Net Uplift         : +1.57 bps (95% CI: [-2.88, +5.98] bps)
  T-statistic        : 0.69 (p-value: 0.4920)                   --> [FAILED]
- SHORT Gated Trades : Fav = 2,327 | Unfav = 1,202
  Favorable Net Return: -3.23 bps | Unfavorable: -4.22 bps
  Net Uplift         : +0.99 bps (95% CI: [-5.14, +7.35] bps)
  T-statistic        : 0.30 (p-value: 0.7605)                   --> [FAILED]
```

```
Mode: Day Gating (Tercile-split daily sentiment, 219 overlap trading days)
-------------------------------------------------------------------------
Downstream Model: v2_15min_3y (15-Minute)
- LONG Gated Trades  : Fav = 5,046 | Unfav = 6,594
  Favorable Net Return: -5.88 bps | Unfavorable: -5.39 bps
  Net Uplift         : -0.49 bps (95% CI: [-1.78, +0.89] bps)
  T-statistic        : -0.70 (p-value: 0.4867)                  --> [FAILED]
- SHORT Gated Trades : Fav = 6,033 | Unfav = 4,035
  Favorable Net Return: -5.58 bps | Unfavorable: -5.54 bps
  Net Uplift         : -0.03 bps (95% CI: [-1.38, +1.30] bps)
  T-statistic        : -0.05 (p-value: 0.9617)                  --> [FAILED]

Mode: Symbol Gating (Top 30% daily-rank symbol filter)
-------------------------------------------------------------------------
Downstream Model: v2_15min_3y (15-Minute)
- LONG Gated Trades  : Fav = 4,698 | Unfav = 4,706
  Favorable Net Return: -5.15 bps | Unfavorable: -6.56 bps
  Net Uplift         : +1.41 bps (95% CI: [-0.12, +3.00] bps)
  T-statistic        : 1.78 (p-value: 0.0755)                   --> [FAILED]
- SHORT Gated Trades : Fav = 4,840 | Unfav = 4,573
  Favorable Net Return: -6.47 bps | Unfavorable: -6.17 bps
  Net Uplift         : -0.31 bps (95% CI: [-1.77, +1.18] bps)
  T-statistic        : -0.41 (p-value: 0.6804)                  --> [FAILED]
```

---

## 5. D6 — Live Integration Verdict & Action Items

### A. Verdict: Gating Integration Rejected
* **The Evidence**: Gating does not yield a statistically significant, positive out-of-sample edge. For 1-Hour Long trades, day gating produced a positive $+3.19$ bps uplift, but the t-statistic of $1.54$ ($p = 0.12$) indicates high likelihood of noise. For 15-Minute Long trades, symbol gating reached $t = 1.78$ ($p = 0.075$) with $+1.41$ bps uplift, failing the $+2.0$ bps threshold. Short gating was uniformly inactive or negative.
* **Live Integration Action**: **Do NOT deploy `daily_macro_v2` as an execution gate inside `orchestrator.py`.** The live daily filter updates will remain bypassed for trade-trigger veto duty.

### B. Alternative Deployment Opportunities
Instead of utilizing the model as a binary execution gate, we recommend:
1. **Auxiliary Feature Conditioning**: Pass the daily long/short ranking scores of day $T-1$ as input features directly into the training sets of the intraday (1H / 15M) models. This allows downstream decision trees to find complex non-linear combinations with intraday indicators.
2. **Dynamic Volatility Scaling**: Use the macro VIX / dispersion metrics from the daily pipeline to adjust maximum live portfolio leverage rather than gating individual trades.

---

## 7. Daily Gatekeeper V3 (1-Day Return Target)

To resolve the 3-day holding period horizon contradiction, we trained `daily_macro_v3` on a **1-day close-to-close target** (`Label_1D = Close(T)/Close(T-1) - 1.0` via `shift(-1)`) over the same 10-year point-in-time dataset.

### A. Walk-Forward Results (1-Day Target)
* **Average Spearman Rho**: Long = $+0.0207$, Short = $+0.0290$ ($p < 0.05$)
* **Top-3 Win Rate (K=3)**: Long = $51.7\%$, Short = $55.7\%$
* **Top-3 Avg Return Edge (over market)**: Long = $+5.86$ bps/day, Short = $+22.16$ bps/day
* **Validation Gauntlet v2 Verdict** (Run ID: `20260610T144343Z-5f7d069f`):
  - **LONG**: **`DEAD`** (due to flat/negative net returns in the recent 24-month window).
  - **SHORT**: **`FILTER_GRADE`** (net return edge of $+11.41$ bps/trade in the recent 24-month window).

### B. Downstream Gating Uplift (V3 Model)
Evaluating `daily_macro_v3` gating against downstream OOS trades:

```
Mode: Symbol Gating (Top 30% daily-rank symbol filter)
-------------------------------------------------------------------------
Downstream Model: v8_upstox_3y (1-Hour, 374 overlap days)
- LONG Gated Trades  : Fav = 1,684 | Unfav = 1,711
  Favorable Net Return: +0.21 bps | Unfavorable: -3.90 bps
  Net Uplift         : +4.11 bps (95% CI: [-0.47, +9.02] bps)
  T-statistic        : 1.78 (p-value: 0.0759)                   --> [FAILED]
- SHORT Gated Trades : Fav = 2,456 | Unfav = 1,094
  Favorable Net Return: -1.73 bps | Unfavorable: -3.53 bps
  Net Uplift         : +1.80 bps (95% CI: [-4.46, +7.72] bps)
  T-statistic        : 0.57 (p-value: 0.5717)                   --> [FAILED]

Downstream Model: v2_15min_3y (15-Minute, 220 overlap days)
- LONG Gated Trades  : Fav = 4,524 | Unfav = 4,616
  Favorable Net Return: -3.42 bps | Unfavorable: -2.26 bps
  Net Uplift         : -1.16 bps (95% CI: [-2.75, +0.47] bps)
  T-statistic        : -1.43 (p-value: 0.1526)                  --> [FAILED]
- SHORT Gated Trades : Fav = 6,587 | Unfav = 3,291
  Favorable Net Return: -1.86 bps | Unfavorable: -3.33 bps
  Net Uplift         : +1.47 bps (95% CI: [-0.68, +3.55] bps)
  T-statistic        : 1.39 (p-value: 0.1638)                   --> [FAILED]
```

```
Mode: Day Gating (Tercile daily market aggregate filter)
-------------------------------------------------------------------------
Downstream Model: v8_upstox_3y (1-Hour, 374 overlap days)
- LONG Gated Trades  : Fav = 1,821 | Unfav = 1,584
  Favorable Net Return: -1.91 bps | Unfavorable: -2.93 bps
  Net Uplift         : +1.01 bps (95% CI: [-3.07, +5.70] bps)
  T-statistic        : 0.45 (p-value: 0.6540)                   --> [FAILED]
- SHORT Gated Trades : Fav = 1,569 | Unfav = 1,197
  Favorable Net Return: -3.98 bps | Unfavorable: -4.45 bps
  Net Uplift         : +0.47 bps (95% CI: [-6.57, +7.49] bps)
  T-statistic        : 0.14 (p-value: 0.8921)                   --> [FAILED]

Downstream Model: v2_15min_3y (15-Minute, 220 overlap days)
- LONG Gated Trades  : Fav = 5,322 | Unfav = 5,118
  Favorable Net Return: -2.33 bps | Unfavorable: -2.93 bps
  Net Uplift         : +0.60 bps (95% CI: [-0.78, +1.96] bps)
  T-statistic        : 0.83 (p-value: 0.4069)                   --> [FAILED]
- SHORT Gated Trades : Fav = 5,253 | Unfav = 5,118
  Favorable Net Return: -3.36 bps | Unfavorable: -3.41 bps
  Net Uplift         : +0.06 bps (95% CI: [-1.87, +1.94] bps)
  T-statistic        : 0.06 (p-value: 0.9537)                   --> [FAILED]
```

### C. Verdict
Aligning the horizons (1-day label gating 1-day intraday trades) significantly improved the Long uplift on the 1-Hour model in Symbol Gating mode to **$+4.11$ bps** (`t-statistic = 1.78`, `p-value = 0.0759`), and pushed the favorable trades' net return positive for the first time (`+0.21 bps` net of costs). 

However, because the $t$-statistic is still below the pre-registered significance threshold of $2.0$ ($p \le 0.05$) and the 15-minute model shows no response (with Symbol Gating LONG return actually degraded by $-1.16$ bps), **V3 gating is also rejected for live deployment**. Gating reduces overall trading volume without a statistically certifiable return boost.

---

## 6. Archives & Repository Hygiene
* The obsolete, unrectified daily gatekeeper model `daily_xgb` has been officially marked **DEAD/DEAD** in the registry.
* All evaluation results, runs, and configuration setups have been locked under the pre-registered ledger run paths `data/gauntlet/20260610T135608Z-5f7d069f/` (V2) and `data/gauntlet/20260610T144343Z-5f7d069f/` (V3).

---
*Follow protocol exactly. This document preserves our core memory and ensures subsequent tuning attempts inherit verified, reproducible benchmarks.*
