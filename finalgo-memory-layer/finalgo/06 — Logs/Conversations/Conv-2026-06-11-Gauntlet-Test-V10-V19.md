---
title: "Conversation Context: Gauntlet Test V10-V19 Models with step_months=2"
type: log
status: concluded
verdict: FILTER_GRADE
updated: 2026-06-12
tags: []
---
# 💬 Conversation Context: Gauntlet Test V10-V19 Models with step_months=2

## 📌 Metadata
- **Conversation ID**: e5ccf53e-60ba-41a5-84a4-e0e836148ed3
- **Start Date**: 2026-06-11
- **Status**: 🔴 Concluded
- **Focus Area**: Model Suite

## 🎯 Objectives
- [x] Create a batch runner script to run the decontaminated step=2 walk-forward Validation Gauntlet on all models from v10 to v19.
- [x] Execute the batch test on supported models.
- [x] Record the run results.
- [x] Rebuild the candidate trade panel, run the orthogonality audit, and execute the capacity ladder using v10-v19 predictions.

## 💻 Active Code Files Modified
- [run_batch_v10_v19.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/gauntlet/run_batch_v10_v19.py)
- [build_trade_panel_v10_v19.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/gauntlet/meta/build_trade_panel_v10_v19.py)
- [orthogonality_audit_v10_v19.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/gauntlet/meta/orthogonality_audit_v10_v19.py)
- [dev_run_v10_v19.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/gauntlet/meta/dev_run_v10_v19.py)
- [freeze_v10_v19.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/gauntlet/meta/freeze_v10_v19.py)

## 📝 Compacted Session Log
- **Initial Analysis**: Starting session to construct and execute a batch testing script running `step_months=2` purged walk-forward Validation Gauntlet on models `v10` through `v19`.
- **Gauntlet Batch Execution**: Executed `run_batch_v10_v19.py` in the background (task-458). The run completed in 2543.2 seconds.
- **Results**:
  - `v10_native_1h`, `v10_depth4_1h`, `v11_utility_1h`, `v14_lambdamart_no_es_1h`, `v15_lambdamart_es_1h`, `v16_binary_breakout_1h`, and `v19_catboost_1h` successfully achieved `FILTER_GRADE` on both Long and Short.
  - `v12_lambdamart_1h` and `v13_ndcg_raw_1h` achieved `DEAD` on Long and `FILTER_GRADE` on Short.
  - `v15_lambdamart_map5_1h` failed instantly due to missing metadata file.
  - `v17_random_forest_1h` and `v18_random_forest_1h` timed out after 12 minutes due to large model size and high execution complexity.
- **Decontamination Wiring**: Modified `build_trade_panel_v10_v19.py` to exclude the 3 unsupported/failed models to allow building a clean panel.
- **Execution Run (v10-v19 Stack)**:
  - **Build Panel**: Rebuilt panel (`build_trade_panel_v10_v19.py`) successfully with 23,506 DEV trades (17 months) and 22,207 VAULT trades, passing the G1 gate check.
  - **Orthogonality Audit**: Passed M1 gate check with a Max Absolute Partial IC of `0.025839` (feature: `daily_v3_sent`).
  - **Capacity Ladder**: Swept all rungs using `dev_run_v10_v19.py --all-rungs`. Rung 1 (L2 Logistic) = `-7.53 bps`; Rung 2 (Shallow GBDT) = `-5.12 bps` (passed G4 gate beating logistic by +2.41 bps); Rung 3 (MLP Small) = `-5.09 bps` (passed G5 gate worst-of-3 seeds).
  - **Freeze & G2 Gate**: Ran `freeze_v10_v19.py` to seal the winning Neural Network. The pre-check triggered **G2 check** and aborted with `RuntimeError: [G2 PRE-CHECK FAILED]` because the winning candidate has negative kept-net (`-5.09 bps`).
- **Conclusion**: The meta-veto stacking line is permanently closed for `v10-v19` models as well due to negative kept-net under 10 bps statutory costs.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[01 — Architecture/Validation Gauntlet/Validation Gauntlet Architecture]], [[02 — Models/_Shared/Model Performance & Statistics]]
