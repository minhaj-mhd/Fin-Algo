---
title: "Conv 2026-07-14 — Train V23 Top20 Features"
type: "log"
status: "active"
updated: "2026-07-14"
---

# 💬 Conversation Context: Train V23 Top20 Features

## 📌 Metadata
- **Conversation ID**: 7cd0f0c7-604e-4cbe-966b-9062e287040f
- **Start Date**: 2026-07-14
- **Status**: 🟢 Active
- **Focus Area**: Model Suite

## 🎯 Objectives
- [ ] Implement `v23_rolling_1h` configuration in `train_ranking_clean.py`.
- [ ] Restrict feature space to Top 20 features from SHAP/Permutation importance.
- [ ] Execute walk-forward training for `v23_rolling_1h`.
- [ ] Evaluate walk-forward performance.

## 💻 Active Code Files Modified
- [train_ranking_clean.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_ranking_clean.py)

## 📝 Compacted Session Log
- **Initial Analysis**: The goal is to constrain the rolling-1h XGBoost model to only use the top 20 features found via SHAP and Permutation tests to see if generalizability improves over v20.
- **Step 1**: Modified `train_ranking_clean.py` to add `1h_roll_v23` configuration using only the 20 features.
- **Step 2**: Executed the walk-forward training for `v23_rolling_1h`.
- **Results**: 
  - Walk-Forward Aggregate: Long Rho `0.0322`, Short Rho `0.0308`.
  - L-WR@3 `53.8%`, S-WR@3 `53.7%`.
  - Combined edge: `+0.0746%` per bar.
  - *Conclusion*: The model trained successfully and generated `xgb_long_model.json`, `xgb_short_model.json`, and `metadata.json` in `models/research/v23_rolling_1h`. However, note that `v20_rolling_1h` had a Long Rho of `0.0345` and Short Rho of `0.0322`, meaning this strict 20-feature constraint caused a slight degradation in raw Spearman performance compared to the full 86-feature model.
- **Step 3 (Sigmoid Probability Threshold Experiment)**: Attempted to cast the raw outputs to a strict probability gate (>0.70) using a Sigmoid function `1/(1+np.exp(-x))`.
  - *Result*: Because `rank:pairwise` heavily compresses scores around 0 (-0.12 to 0.11), the max probability was ~52.8%. The 70% threshold filtered out 100% of trades.
  - *Proxy OOS (June 2026) Check*: Lowering the threshold to >0.52 isolated massive short edge (+58.27 bps) but only fired 7 times a month. The long side was heavily anti-selected (33% WR, -13.3 bps edge). 
  - *Conclusion*: A hard absolute probability gate requires `binary:logistic` training or local Z-score normalization; applying it directly to pairwise rank scores yields zero volume.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[02. Model Suite/Model Registry & File Structures]]
