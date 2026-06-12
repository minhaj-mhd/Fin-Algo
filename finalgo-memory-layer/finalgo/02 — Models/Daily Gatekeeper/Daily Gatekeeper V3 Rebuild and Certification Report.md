# 🏛️ Daily Gatekeeper V3 — Rebuild & Certification Report

> **Status**: 🔴 CONCLUDED — Gating Certification FAILED (V3 Gating Rejected)
> **Date**: 2026-06-10
> **Authors**: Antigravity
> **Certified Run ID**: `20260610T144343Z-5f7d069f`
> **Linked Specs**: [[02 — Models/Daily Gatekeeper/Daily Gatekeeper V2 Rebuild Plan]], [[01 — Architecture/Validation Gauntlet/Validation Gauntlet Architecture]]

---

## 1. Executive Summary

This report documents the end-to-end execution, standalone certification, and downstream gating uplift results of the **Daily Gatekeeper V3 (1-Day Return Target)** model.

To resolve the 3-day holding period horizon contradiction of the V2 model (which gates intraday trades that exit at 15:15 IST same-day), we rebuilt the daily gatekeeper dataset using a **1-day close-to-close returns target** (`Label_1D`). 

While the standalone V3 model successfully certified its short-side signal as **`FILTER_GRADE`** under the Validation Gauntlet, downstream out-of-sample gating uplift tests **failed** to satisfy the required certification threshold (uplift $\ge +2.0$ bps net and $t\text{-statistic} \ge 2.0$) on both 1-hour and 15-minute models. Consequently, we recommend **against** utilizing V3 as a live binary execution gate.

---

## 2. Dataset Construction & Target Alignment

The Daily Gatekeeper V3 dataset (`data/ranking_data_daily_macro_v3.csv`) spans **10 years (2016-2026)** to wash out market noise across multiple regimes:

* **Decision-Time Contract**: Every daily feature in row $T$ is aligned strictly using the last available value prior to **09:00 IST on trade day T** (pre-open).
* **Target Label**: 1-bar close-to-close forward return:
  $$\text{Label\_1D} = \frac{\text{Close}_{T}}{\text{Close}_{T-1}} - 1.0$$
* **Verification**: Lookahead leakage tests in `tests/test_daily_macro_dataset.py` passed with 100% conformance, verifying that no feature in row $T$ matches or correlates 1.0 with any forward data timestamped after 09:00 IST on day $T$.

---

## 3. Standalone Model Training & Gauntlet Certification

### A. Training Setup
* **Algorithm**: XGBoost `rank:pairwise` (dual long/short models via label inversion).
* **Regularization**: Deeply regularized to prevent noise fitting ($\eta = 0.01$, $\text{max\_depth} = 5$, $\alpha = 2.0$, $\lambda = 4.0$, $\text{min\_child\_weight} = 40$).
* **Folds Configuration**: 26 temporal walk-forward folds.

### B. Standalone Walk-Forward Results
Averaged over all walk-forward folds:
* **Average Spearman Rho**: Long = $+0.0207$, Short = $+0.0290$ ($p < 0.05$)
* **Top-3 Win Rate (K=3)**: Long = $51.7\%$, Short = $55.7\%$
* **Top-3 Avg Return Edge (over market)**: Long = $+5.86$ bps/day, Short = $+22.16$ bps/day

### C. Validation Gauntlet v2 Verdict
Evaluated under Criteria v2 over the test timeline:
* **LONG**: **`DEAD`** (due to poor performance in the recent 24-month window).
* **SHORT**: **`FILTER_GRADE`** (net return edge of $+11.41$ bps/trade in the recent 24-month window).

The model registry signature was successfully stamped into `models/daily_macro_v3/metadata.json` under signature hash `c913e8a559f085a44531ccb26ec504aadc493e079910cf8426f6b30f7b6fc6d3`.

---

## 4. Downstream Gating Uplift Certification

We evaluated the performance uplift of gating downstream out-of-sample trades using `daily_macro_v3` predictions (cost adjusted to 6 bps). Gating must achieve **uplift $\ge +2.0$ bps net with $t$-statistic $\ge 2.0$** to pass certification.

### A. Symbol Gating Mode (Top 30% daily-rank symbol filter)
Checks if trading is restricted only to the top 30% daily-ranked stocks:

* **Downstream Model: `v8_upstox_3y` (1-Hour Model, 374 overlap days)**:
  - **LONG Gated Trades**: Fav = 1,684 | Unfav = 1,711
    - Favorable Net Return: $+0.21$ bps | Unfavorable: $-3.90$ bps
    - Net Uplift: **`+4.11 bps`** (95% CI: `[-0.47, +9.02]` bps)
    - T-statistic: **`1.78`** ($p\text{-value}: 0.0759$) $\rightarrow$ **`[FAILED]`**
  - **SHORT Gated Trades**: Fav = 2,456 | Unfav = 1,094
    - Favorable Net Return: $-1.73$ bps | Unfavorable: $-3.53$ bps
    - Net Uplift: **`+1.80 bps`** (95% CI: `[-4.46, +7.72]` bps)
    - T-statistic: **`0.57`** ($p\text{-value}: 0.5717$) $\rightarrow$ **`[FAILED]`**

* **Downstream Model: `v2_15min_3y` (15-Minute Model, 220 overlap days)**:
  - **LONG Gated Trades**: Fav = 4,524 | Unfav = 4,616
    - Favorable Net Return: $-3.42$ bps | Unfavorable: $-2.26$ bps
    - Net Uplift: **`-1.16 bps`** (95% CI: `[-2.75, +0.47]` bps)
    - T-statistic: **`-1.43`** ($p\text{-value}: 0.1526$) $\rightarrow$ **`[FAILED]`** (Return degraded)
  - **SHORT Gated Trades**: Fav = 6,587 | Unfav = 3,291
    - Favorable Net Return: $-1.86$ bps | Unfavorable: $-3.33$ bps
    - Net Uplift: **`+1.47 bps`** (95% CI: `[-0.68, +3.55]` bps)
    - T-statistic: **`1.39`** ($p\text{-value}: 0.1638$) $\rightarrow$ **`[FAILED]`**

### B. Day Gating Mode (Tercile daily market aggregate filter)
Checks if trading is blocked entirely on unfavorable market days:

* **Downstream Model: `v8_upstox_3y` (1-Hour Model, 374 overlap days)**:
  - **LONG Gated Trades**: Fav = 1,821 | Unfav = 1,584
    - Net Uplift: **`+1.01 bps`** ($t\text{-statistic}: 0.45$, $p\text{-value}: 0.6540$) $\rightarrow$ **`[FAILED]`**
  - **SHORT Gated Trades**: Fav = 1,569 | Unfav = 1,197
    - Net Uplift: **`+0.47 bps`** ($t\text{-statistic}: 0.14$, $p\text{-value}: 0.8921$) $\rightarrow$ **`[FAILED]`**

* **Downstream Model: `v2_15min_3y` (15-Minute Model, 220 overlap days)**:
  - **LONG Gated Trades**: Fav = 5,322 | Unfav = 5,118
    - Net Uplift: **`+0.60 bps`** ($t\text{-statistic}: 0.83$, $p\text{-value}: 0.4069$) $\rightarrow$ **`[FAILED]`**
  - **SHORT Gated Trades**: Fav = 5,253 | Unfav = 5,118
    - Net Uplift: **`+0.06 bps`** ($t\text{-statistic}: 0.06$, $p\text{-value}: 0.9537$) $\rightarrow$ **`[FAILED]`**

---

## 5. Live Integration Verdict & Recommendations

### A. Verdict: Live Gating Rejected
* **The Horizon Fix Results**: Resolving the holding period mismatch (using `daily_macro_v3`'s 1-day target to gate intraday trades) significantly improved the Long 1-Hour model's symbol gating performance: it generated a positive net return of $+0.21$ bps and a net outperformance uplift of **$+4.11$ bps**.
* **Why it cannot be deployed**: Despite the positive direction, the $t$-statistic of $1.78$ is still below the statistical significance threshold of $2.0$ ($p \le 0.05$). Under gating, the overall trading volume is cut by ~50% without a statistically certain return improvement. Additionally, on the 15-minute model, gating LONG trades actually *harmed* net returns by $-1.16$ bps.
* **Live Action**: **Do NOT deploy `daily_macro_v3` as an execution gate in `orchestrator.py`.** Daily filters will remain bypassed for live trade-trigger duty.

### B. Recommendations for Future Research
1. **Auxiliary Feature Integration**: Instead of binary gating, feed the raw V3 daily long/short rank scores as input features to downstream 1H and 15M model training sets, allowing the models to dynamically learn sector and macro regimes.
2. **Dynamic Volatility Scaling**: Use the daily VIX / breadth dispersion indicators to dynamically adjust position size or total portfolio leverage rather than gating individual trades.

---

## 6. Archives & Repository Hygiene
* All evaluation results, runs, and configuration setups are archived and locked under the pre-registered ledger run path: [data/gauntlet/20260610T144343Z-5f7d069f/](file:///c:/Users/loq/Desktop/Trading/finalgo/data/gauntlet/20260610T144343Z-5f7d069f/).
