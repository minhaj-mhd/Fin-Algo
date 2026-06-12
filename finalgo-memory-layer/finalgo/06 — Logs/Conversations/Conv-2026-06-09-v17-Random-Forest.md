# 💬 Conversation Context: v17 Random Forest Experiment

## 📌 Metadata
- **Conversation ID**: 97a0de4c-e037-44e0-9aa7-25be15769179
- **Start Date**: 2026-06-09
- **Status**: 🟢 Active
- **Focus Area**: Model Suite / Trading Strategies

## 🎯 Objectives
- [x] Create v17 Random Forest algorithm with bagging.
- [x] Run walk-forward evaluation on v17.
- [x] Update all model training scripts (v16, v17, hybrid) to output Raw Return, Net Return, Raw Winrate, and Net Hitrate.
- [ ] Incorporate v17 into Hybrid v10 ranker.

## 💻 Active Code Files Modified
- [train_v17_random_forest.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_v17_random_forest.py)
- [train_v16_binary_breakout.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_v16_binary_breakout.py)
- [train_hybrid_v10_v16.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_hybrid_v10_v16.py)

## 📝 Compacted Session Log
- **Initial Analysis**: The v16 baseline (Gradient Boosting) struggled to filter noise on the Long side, leading to over-regularization and negative net returns (-14.36 bps).
- **Step 1**: Created `v17` using XGBoost configured as a parallel Random Forest (`num_parallel_tree=100`, `num_boost_round=1`, `max_depth=10`, `subsample=0.8`).
- **Step 2**: Evaluated v17 in 8-fold walk-forward validation with `TRADE_PROB > 0.62`.
- **Step 3**: Updated logging in v16, v17, and hybrid scripts to calculate Raw vs Net Return and Raw Winrate vs Net Hitrate. 

### v17 Random Forest Evaluation Results (8-Fold Walk-Forward)
**Longs (7,381 trades):**
- Raw Return: `+7.64 bps`
- Net Return: `-2.36 bps` (Cost: 10 bps)
- Raw Winrate: `54.02%` (Trades > 0)
- Net Hitrate: `46.21%` (Trades > 10 bps)

**Shorts (2,618 trades):**
- Raw Return: `-3.97 bps`
- Net Return: `-13.97 bps`
- Raw Winrate: `49.66%`
- Net Hitrate: `43.39%`

**Comparison Rationale**: The v17 Random Forest significantly improved the Long side's raw edge (+7.64 bps raw return, -2.36 bps net) compared to Gradient Boosting, and found nearly double the number of breakout signatures. Since it handles high-variance data better via bagging, it should be tested within the v10 Hybrid Ranker.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[02 — Models/_Shared/Model Registry & File Structures]]
