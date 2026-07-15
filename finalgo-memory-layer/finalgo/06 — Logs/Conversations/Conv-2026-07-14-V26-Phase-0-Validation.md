# 💬 Conversation Context: V26 Phase 0 Validation

## 📌 Metadata
- **Conversation ID**: b7c7a624-460e-48b6-b0e7-c1b17148c62f
- **Start Date**: 2026-07-14
- **Status**: 🟢 Active
- **Focus Area**: Trading Strategies / Validation

## 🎯 Objectives
- [x] E0.1: Lookahead audit of 100-DMA routing key
- [x] E0.2: Train/serve skew fix
- [x] E0.3: Survivorship bias check
- [x] E0.4: Data accounting reconciliation

## 💻 Active Code Files Modified
- [eval_regime_router.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/eval_regime_router.py)
- [train_binary_clean.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_binary_clean.py)

## 📝 Compacted Session Log
- **Initial Analysis**: Brought V26 validation plan and walkthrough from previous session. Commenced Phase 0 correctness gates.
- **E0.1 (Lookahead Bias)**: Found severe lookahead bias in `eval_regime_router.py` and `train_binary_clean.py` where regime assignment for day T used day T's close. Shifted Nifty 100-DMA features by 1 day.
- **E0.2 (Train/Serve Skew)**: Excluded the 1.5% buffer zone from `train_binary_clean.py` when training Bull/Bear models. Retrained models.
- **Performance Impact**: Post-fix, the strategy's edge collapsed. 6m Testing combined edge dropped from +27.44 bps to +2.38 bps. Long edge flipped to negative (-12.56 bps). 
- **E0.3 (Survivorship Bias)**: Checked `build_rolling_1h_panel.py` and `collect_upstox_15min_3y.py`. Found the dataset is built using a static, hardcoded list of 148 recent/current tickers (`scripts/tickers.py`), meaning fatal survivorship bias exists.
- **E0.4 (Data Reconciliation)**: Verified that exactly 306,093 rows were dropped due to the 100-DMA initialization window (NaNs). Remaining 2.79M rows are perfectly partitioned between Bull, Bear, and Chop. No unexplained row dropping.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[02. Model Suite/Model Registry & File Structures]]
