---
title: "Conversation Context: Dual-TF Entry/Exit Overlay Research"
type: log
status: active
updated: 2026-06-12
tags: []
---
# 💬 Conversation Context: Dual-TF Entry/Exit Overlay Research

## 📌 Metadata
- **Conversation ID**: 789886c3-1bd0-49c2-ab88-dd82c9b2743a
- **Start Date**: 2026-06-11
- **Status**: 🟢 Active
- **Focus Area**: Model Suite / 15m overlay research

## 🎯 Objectives
- [x] Answer: can 15m model dictate entry/exit/block for 1h signals? (NO — full-hold optimal)
- [x] Test entry rank-trajectory gating (THRIVING/DIMINISHING) with price-momentum control
- [x] Test early-exit on conviction decay vs dumb price-stop control
- [x] Test asymmetric "let winners run" exit rules
- [x] Test conviction-momentum × price-direction → remaining-return buckets (user-designed)
- [x] Build reusable dual-TF trade panel + save all in-depth results for future research
- [x] Fresh edge probes (depth, joint gate, time-of-day) → hour-13 EOD reversion finding
- [x] Write findings to memory layer

## 💻 Active Code Files Modified
- [wf_rank_trajectory.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/wf_rank_trajectory.py) (new)
- [wf_early_exit.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/wf_early_exit.py) (new)
- [exit_rule_sweep.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/exit_rule_sweep.py) (new; fixed require_red logic bug mid-study)
- [exit_momentum_buckets.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/exit_momentum_buckets.py) (new; fixed short-side sgn bug mid-study)
- [build_dualtf_panel.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/build_dualtf_panel.py) (new)
- [panel_edge_probes.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/panel_edge_probes.py) (new)
- Data/results: `data/research/entry_exit/` (panel 13,020 trades, candidates, README, 5 result logs)

## 📝 Compacted Session Log
- **Initial question**: 15m model for entry/exit/blocking of 1h signals. Reviewed memory: dual-TF snapshot-confirm already dead, meta-veto closed, host (1h) net-negative.
- **Study 1 (entry trajectory)**: 4-candle rank slope → directionally right (+2.0/+2.4 bps) but ns (p≈0.22/0.28), orthogonal to price-mom (corr≈0); kept bucket net-negative. LATER RETRACTED as feature (Probe B sign flip).
- **Study 2 (early exit)**: conviction decay cuts losers −40→−23 but clips winners −24.6; net ≈ full-hold. Beats dumb price-stop (which is worst rule — underwater trades recover).
- **User asked "why no profit if cutting 20bps?"** → decomposition: clip:save ≈ 1.4:1, exit fires on 75–85% of trades, anti-selective.
- **Study 3 (asymmetric, user idea "protect winners")**: weak∧red gating collapses clip −24.6→−2.9 but save collapses too → converges to full-hold. Full-hold OPTIMAL for the {conviction, P&L} rule family.
- **Study 4 (user-designed momentum×price buckets, remaining return)**: only significant result (long p=0.046) but INVERTED — FAV−/CONV− best (+3.0), FAV+/CONV+ worst (−0.5). ~80% of effect = price mean-reversion axis.
- **Study 5 (fresh probes as Fable)**: depth-flat reversion (dead late-dip-entry); joint gate no positive cell (⇒ slope retraction); ⭐ EOD finding — hour-13 FAV− remaining = +6.7 long (p=0.0037) / +8.9 short (p=0.0066), coherent ramp both sides, but < 10 bps binding cost.
- **Synthesis**: residual price/volume edge = small, depth-insensitive, EOD-concentrated mean reversion. Momentum framings 4× inverted. Next edge needs new info (options OI/IV, order-flow) or lower friction.
- **Memory layer**: wrote `08. Model Analysis/15-Minute Vanguard Model/Dual-TF Entry-Exit Overlay Research.md`; updated auto-memory + research README.

## 🔗 Core Memory Links & Backlinks
- [[04 — Research/Dual-TF Entry-Exit Overlay Research|Findings note (permanent)]]
- [[02 — Models/_Shared/Multi-Timeframe Models]]
- [[02 — Models/Meta-Veto/Meta-Veto Rectification Plan MV2]]
- [[01 — Architecture/Validation Gauntlet/Validation Gauntlet Architecture]]
