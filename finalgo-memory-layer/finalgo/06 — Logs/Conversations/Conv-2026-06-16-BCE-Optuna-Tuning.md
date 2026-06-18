---
title: "BCE Veto Transformer — Optuna Tuning (agent log)"
type: log
status: 🟢 Active
updated: 2026-06-16
---

# 💬 Conversation Context: BCE Veto Transformer — Optuna Tuning

## 📌 Metadata
- **Start Date**: 2026-06-16
- **Status**: 🟢 Active
- **Focus Area**: Model tuning (DualRes BCE transformer veto on v20)
- **Learning walkthrough (plain-language, user-facing)**: [[BCE Optuna Tuning — Step-by-Step]]

## 🎯 Objectives
- [ ] Optuna search over loss family + architecture to strengthen the K=3 LONG veto edge
- [ ] Honest single-shot test confirmation vs baseline (+1.14 bps / t +2.27)

## 💻 Active Code Files
- [train.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/train.py) — 4 BCE-family losses, arch/loss CLI args, `--no_save`, `chrono_split`
- [veto_lib.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/veto_lib.py) — window-agnostic veto scorer (coverage-matched, bootstrap, neg-control, block floor)
- [tune_bce_optuna.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/tune_bce_optuna.py) — the search

## 📝 Compacted Log
- **Design**: objective = coverage-matched K=3 LONG veto **t-stat** on VAL − worst-block penalty;
  neg-control rejection; TEST frozen for Phase-2 only. User chose: downstream objective + full
  (incl. architecture) scope + add the 4 loss variants + stability + threshold/calibration handling.
- **Phase 0 ✅**: losses verified numerically (finite/grad/O(1); hybrid terms balanced 0.73 vs 0.15).
  Timing run: ~110 s/epoch on RTX 5050, cost-acct clean, `--no_save` protects production.
- **Phase 1**: `tune_bce_optuna.py` written; epochs capped 10, fast subsample-AUC pruning,
  2.5h timeout; 1-trial sanity run in progress.

## 🔗 Core Memory Links
- [[BCE-Transformer-V20-Veto]] · [[project_dualres_transformer_result]] · [[Conv-2026-06-16-v20-Cadence-Transformer]]
