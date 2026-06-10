# 💬 Conversation Context: Hybrid Model V10 + V16 Optimization

## 📌 Metadata
- **Conversation ID**: 97a0de4c-e037-44e0-9aa7-25be15769179
- **Start Date**: 2026-06-09
- **Status**: 🟢 Active
- **Focus Area**: Model Suite / Trading Strategies

## 🎯 Objectives
- [x] Integrate V10 Ranker and V16 Binary Breakout Classifier into a Hybrid architecture.
- [x] Test varying thresholds (Top 1, Top 3, Top 5) and confidence levels (60%, 62%, 57%).
- [x] Test restricting max_depth (from 5 to 4) to reduce overfitting.
- [ ] Develop and implement a Regime Filter for the Long side.

## 💻 Active Code Files Modified
- [train_hybrid_v10_v16.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_hybrid_v10_v16.py)

## 📝 Compacted Session Log
- **Initial Analysis**: The user sought to understand the best combinations of the V10 and V16 models. Initial tests showed high fee erosion across most configurations.
- **Step 1**: Built `train_hybrid_v10_v16.py` to evaluate Rank-then-Veto and Filter-then-Rank logic.
- **Step 2**: Evaluated expanding to Top 5 at 60% probability. Result: Massive fee bleed (net negative bps).
- **Step 3**: Evaluated restricting `max_depth` from 5 to 4. Result: **Massive improvement.** Regularizing the trees allowed the model to generalize. Logic A1 (Top 1, 62% Prob) achieved `+7.82 net bps` on Shorts (up from +3.4 bps) and `+2.30 net bps` on Longs.
- **Step 4**: User requested to see results for high volume configurations (Top 3 at 57% probability). Ran the test. Result: Negative edge (`-8.82 net bps` on shorts). Proved mathematically that forcing 10-20+ trades a day on 1-hour Indian cash equities fails due to the 10 bps statutory fee hurdle.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[02. Model Suite/Model Registry & File Structures]]
