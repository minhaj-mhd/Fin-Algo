---
title: "V24 Binary Logistic Threshold Experiment"
type: research
status: concluded
updated: 2026-07-14
model: v24_binary_1h
tags: [research, 1h, classification, binary, sigmoid]
---
# V24 Binary Logistic Threshold Experiment

> [!warning] ⚠️ UNVERIFIED — research only, NO Gauntlet run, no verdict authority.
> All numbers below are walk-forward **point estimates** from the training harness
> ([train_binary_clean.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_binary_clean.py) `--tf 1h_roll_v24` & `1h_roll_v24_top20`).
> Overlapping windows make consecutive rows ~75% autocorrelated → effective N ≈ ¼ of row
> count, so **significance/CI are inflated; do NOT t-test these.** Not in `models/registry.json`.

## Hypothesis
Previous 1h models (like `v20`) were trained using the `rank:pairwise` objective, which compresses scores around 0 and prevents strict probability threshold gating (e.g. `>0.70`). By switching the objective to `binary:logistic`, the model will inherently learn to output raw probabilities bounded between [0, 1]. We hypothesize that applying strict absolute thresholds (`>0.55`, `>0.60`) to these true probabilities will isolate high-conviction trades and push the gross edge above the binding 10 bps statutory execution cost.

## Build 
- **Data Universe**: Identical rolling 1h windows (15-min step) panel generated for `v20`.
- **Target Label**: Converted the continuous `Next_Hour_Return` into a strict binary label.
  - Long side: `1` if `Return > 0`, else `0`
  - Short side: `1` if `Return < 0`, else `0`
- **Runs**:
  - `v24` Base: Uses all 86 original features.
  - `v24_top20`: Uses only the Top 20 features (Long/Short unique) as determined by XGBoost `gain` from the Base model.

## Result (purged monthly walk-forward) 

### Full 86-Feature Run (Average across folds)
| Threshold | Long Edge (Gross) | Long Trades | Short Edge (Gross) | Short Trades |
| --- | --- | --- | --- | --- |
| **>0.50** | +1.77 bps | 17,286 | +0.59 bps | 61,146 |
| **>0.55** | -1.44 bps | 1,165 | **+16.92 bps** | 4,534 |
| **>0.60** | +8.59 bps | 184 | +5.80 bps | 170 |
| **>0.65** | -5.49 bps | 47 | +4.54 bps | 74 |
| **>0.70** | +11.10 bps | 14 | -9.93 bps | 48 |

### Top 20 Features Only (Average across folds)
| Threshold | Long Edge (Gross) | Long Trades | Short Edge (Gross) | Short Trades |
| --- | --- | --- | --- | --- |
| **>0.50** | +2.06 bps | 16,133 | +0.70 bps | 61,610 |
| **>0.55** | +3.64 bps | 886 | +10.42 bps | 3,490 |
| **>0.60** | +2.88 bps | 116 | **+15.35 bps** | 247 |
| **>0.65** | +8.78 bps | 39 | +7.33 bps | 92 |
| **>0.70** | +9.39 bps | 16 | +3.58 bps | 63 |

## Verdict (research)
- **Falsified Hypothesis**: While the binary classifier correctly output wide probabilities allowing for strict threshold gating, the resulting Long trades heavily regressed, and the Short trades topped out at ~16 bps gross edge.
- **Still Sub-Cost / Marginal**: A gross edge of +16.92 bps (from the 86-feature model at `>0.55`) yields a net profit of only ~6.9 bps per trade after the 10 bps statutory ceiling. While this is technically profitable over a large sample, the Long side entirely collapses.
- **Top-20 Constraining Degrades Calibration**: Reducing the model to the Top 20 features actually forced the calibration threshold higher (the peak edge of ~15 bps shifted from `>0.55` up to `>0.60`), but simultaneously reduced the overall maximum edge compared to the 86-feature base.

## Reusable artifacts
- Panel: `data/research/v20_rolling_1h/panel.parquet`
- Models: `models/research/v24_binary_1h/` and `models/research/v24_binary_1h_top20/`
- Rerun: `python scripts/training/train_binary_clean.py --tf 1h_roll_v24`

## Links
- Predecessor model: [[04 — Research/V23 Top-20 Constrained Ranker]]
- Session: [[06 — Logs/Conversations/Conv-2026-07-14-Train-V23-Top20-Features|Conversation Log]]
