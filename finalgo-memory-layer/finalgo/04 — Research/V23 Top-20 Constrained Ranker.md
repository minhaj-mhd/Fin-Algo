---
title: "V23 Top-20 Constrained Ranker"
type: research
status: concluded
updated: 2026-07-14
model: v23_rolling_1h
tags: [research, 1h, ranker, overlapping-windows, feature-selection]
---
# V23 Top-20 Constrained Ranker

> [!warning] ⚠️ UNVERIFIED — research only, NO Gauntlet run, no verdict authority.
> All numbers below are walk-forward **point estimates** from the training harness
> ([train_ranking_clean.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_ranking_clean.py) `--tf 1h_roll_v23`).
> Overlapping windows make consecutive rows ~75% autocorrelated → effective N ≈ ¼ of row
> count, so **significance/CI are inflated; do NOT t-test these.** Not in `models/registry.json`.

## Hypothesis
If we strip away 75% of the `v20_rolling_1h` features and force the XGBoost ranker to strictly use the top 20 predictors (identified via SHAP values and Permutation Drop tests), we can reduce overfitting and potentially improve out-of-sample generalizability and raw predictability. 

## Build 
- **Data Universe**: Uses the identical rolling 1h windows (15-min step) panel generated for `v20`.
- **Feature Set**: Stripped the 86 original features down to the following **20 features**:
  `Dist_Keltner_Lower`, `Relative_Return`, `Keltner_Width`, `Return`, `CMF_20`, `Log_Return`, `PPO_Signal`, `Dist_52W_Low`, `RVOL`, `Lower_Shadow`, `Dollar_Volume`, `Intraday_Return`, `Dist_HMA_12`, `Hour`, `TRIX_15`, `Dist_Donchian_Upper`, `Rolling_Skew`, `IBS`, `Donchian_Width`, `VWAP_Dist`.
- **Label**: `Next_Hour_Return = close(T+1h)/close(T)-1`, exact-timestamp reindex = session mask.
- **Data Source**: `data/research/v20_rolling_1h/panel.parquet`

## Result (purged monthly walk-forward) — EQUAL SPAN
| WF avg | **v23 rolling-1h (20 feats)** | v20 rolling-1h (86 feats) |
| --- | --- | --- |
| Long ρ | **0.0322** | 0.0322 |
| Short ρ | **0.0308** | 0.0326 |
| Long WR@3 | 53.8% | **54.2%** |
| Short WR@3 | **53.7%** | 53.6% |
| Combined gross edge | +7.46 bps per bar | +7.40 bps per bar |
| Span | 2022–2026 | 2022–2026 |

## Verdict (research)
- **Falsified Hypothesis**: Removing the "noise tail" of 66 secondary features did **not** improve raw predictability. In fact, forcing the model into a strictly sparse 20-feature environment kept Long Spearman correlation identical (0.0322) and slightly degraded the Short Spearman correlation (from 0.0326 down to 0.0308).
- **The Information Ceiling Holds**: The tree-based architecture inherently manages feature redundancy well. Slicing off the bottom 75% of features merely deprives the model of secondary proxies when its primary splits are unavailable.
- **Still Sub-Cost**: With a combined gross edge of ~7.46 bps per bar, `v23` remains strictly net-negative after applying the binding 10 bps statutory+slippage cost. It operates firmly inside the information ceiling we have observed across all 1h ranking architectures.

## Probability Threshold Experiment (Sigmoid Gate)
Attempted to convert `rank:pairwise` raw scores into strict probabilities via Sigmoid activation `1/(1+exp(-x))` and filter trades at >0.70.
- **Score Compression**: Because `rank:pairwise` minimizes relative distances rather than producing absolute log-odds, raw scores were tightly bound (-0.12 to +0.11), yielding maximum probabilities of ~52.8%. The 70% threshold filtered out **100% of trades**.
- **Proxy OOS Check (June 2026)**: Dropping the threshold to >0.52 (extreme upper tail) yielded 15 Longs and 7 Shorts for the entire month.
  - *Longs > 0.52*: 33.3% WR | -13.30 bps Edge (Anti-selected; highest conviction longs are the worst trades).
  - *Shorts > 0.52*: 57.1% WR | **+58.27 bps Edge** (Massive edge, but critically low frequency).
- **Conclusion**: The ranking model correctly identifies highly profitable Shorts at the extreme tail, but a strict absolute probability filter is structurally incompatible with a pairwise objective function without score normalization or re-training as `binary:logistic`.

## Reusable artifacts
- Panel: `data/research/v20_rolling_1h/panel.parquet` (gitignored)
- Model: `models/research/v23_rolling_1h/` (XGB long/short, not registered)
- Rerun: `python scripts/training/train_ranking_clean.py --tf 1h_roll_v23`

## Links
- Predecessor model: [[04 — Research/V20 Rolling-1h Overlapping-Window Model|v20_rolling_1h]]
- Session: [[06 — Logs/Conversations/Conv-2026-07-14-Train-V23-Top20-Features|Conversation Log]]
