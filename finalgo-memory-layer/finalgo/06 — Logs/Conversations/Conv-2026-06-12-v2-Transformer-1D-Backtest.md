# 💬 Conversation Context: v2 + Transformer 1-Day Holding Backtest

## 📌 Metadata
- **Conversation ID**: 55f04678-1e9b-4eab-b81f-1b9de53f2517
- **Start Date**: 2026-06-12
- **Status**: 🔴 Concluded
- **Focus Area**: Trading Strategies / Models

## 🎯 Objectives
- [x] Evaluate the user's conflicting requests regarding the Transformer veto and v2.
- [x] Research the existing v2 and Transformer model scripts.
- [x] Implement or configure a backtest for the v2 + Transformer model with a 1-day holding period.
- [x] Review results and determine the validity of the 1-day holding period for this combo.
- [x] Extract isolated performance for the final month (May/June 2026) on both 1-day and 3-day holding periods.

## 💻 Active Code Files Modified
- [scripts/transformer/daily_veto_1d_walkforward.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/daily_veto_1d_walkforward.py) (Created by duplicating the 3D version and replacing the labels to 1D returns).
- `data/daily_transformer_panel/Y_1d.npy` (Extracted from the base CSV `Label_1D`).
- [scripts/transformer/eval_last_month.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/eval_last_month.py) (Custom isolation script).

## 📝 Compacted Session Log
- **Initial Analysis**: The user provided two consecutive prompts: one strongly advocating for deploying v2 alone based on solid statistical arguments, and another requesting to backtest the v2+Transformer model with a 1-day holding period.
- **Execution**: Generated `Y_1d.npy` from `ranking_data_daily_macro_v2.csv` containing the 1-day forward return labels. Copied and modified the daily veto walkforward script to evaluate on `Y_1d.npy`.
- **Results**: The 1-day holding period reduces the `v2-alone` LONG edge from a statistically significant +32.1 bps to a non-significant +7.46 bps (at K=1) and +9.48 bps (at K=3). The 10bps cost consumes a significant chunk of the 1-day gross returns.
- **Transformer Impact**: The transformer continues to provide negligible and non-significant uplift (+0.28 to +1.21 bps). It still agrees with `v2` on 98-99% of the Long picks.
- **Conclusion**: The user's original thesis that the transformer adds no value is fully confirmed in the 1-day period as well. Furthermore, the 1-day holding period itself destroys the certified edge that `v2` had on the 3-day holding period. Deploying `v2-alone` with a 3-day hold remains the statistically optimal choice.
- **Last Month Isolation**: Filtered the OOS days to only the final 30 days of the dataset (23 trading days from 2026-05-04 to 2026-06-04). Due to the small N, variance dominates, but the macro structural conclusions hold: the 3-Day holding period vastly outperforms the 1-Day holding period, and the transformer has negligible utility on the long side.
- **Custom Exits (Hybrid & Stop Losses) Evaluation**: Evaluated a "Hybrid" exit (quit on Day 1 if negative) and End-of-Day Stop Losses (2%, 3%, 5%) across the *entire 478-day OOS period*. The findings conclusively reject the Hybrid rule (Long edge drops from +34.86 bps to +15.16 bps net). Stop losses perform marginally worse than the pure 3-Day hold across the full dataset (+32.6 bps for a 5% SL vs +34.8 bps for no SL). The pure 3-Day hold remains the structurally dominant execution method.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[02 — Models/Transformer/Cross-Sectional Transformer Architecture Proposal]]
