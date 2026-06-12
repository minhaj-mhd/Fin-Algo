# 💬 Conversation Context: Side-Specialist Transformer + Cost-Aware Veto Gate

## 📌 Metadata
- **Conversation ID**: (session 32a1f623)
- **Start Date**: 2026-06-12
- **Status**: 🔴 Concluded
- **Focus Area**: Model Suite (transformer), Trading Strategies (v10 veto)

## 🎯 Objectives
- [x] Test whether splitting the single-head P(up) transformer into long/short side-specialists helps.
- [x] Correct the objective to the real one: per-side STRENGTH estimator as a VETO on v10's picks.
- [x] Design + build a custom cost-aware veto-gate loss for that objective.
- [x] Verify honestly (OOS, t-stats, raw vs net WR) and log the verdict.

## 💻 Active Code Files Modified
- [train.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/train.py) — `--objective {listwise,gate}`, `--target`, `--v10_restrict`, custom gate loss
- [eval_sided.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/eval_sided.py)
- [gate_veto_v10.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/gate_veto_v10.py)
- [make_v10_pickmask.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/transformer/make_v10_pickmask.py)

## 📝 Compacted Session Log
- **Init**: started from the question of how many v10 shorts the existing single-head transformer veto removed (n=493/610 ≈19% @K1). User proposed two side-specialist models for long/short *strength*.
- **Survivorship audit (gating)**: panel is a fixed 172-name universe (170/172 span full 2023→2026 window, ~1 true delisting retained) → ≈today's constituents backfilled, mild bias. Doesn't block exploratory phases; caveat binds any deploy claim.
- **Arm 1 — listwise ranker (ListNet)**: DEAD. Short rank-IC ρ=+0.006 (~9× below the 0.05 bar), best net@10 +0.78 t=0.13=noise; long dead. **Answered the wrong question** (standalone selection, not veto). User correctly flagged the loss didn't fit the objective.
- **Objective correction**: the real goal is a per-side STRENGTH/conviction gate that vetoes v10's picks, judged by **uplift to v10's net** — not rank-IC.
- **Arm 2 — custom cost-aware veto-gate loss** `L = −mean(g·n)·1e4 + λ·(mean(g)−ρ)²` (g=sigmoid, n=side_sign·y−cost, ρ=0.70, λ=100). Unit-tested: drives g→1 on net-winners, →0.19 on losers, budget binds. Plain gate (full xsec) short Δ vs v10 +2.75/+1.46/+0.32 @K1/3/5.
- **Arm 3 — v10-focused gate** (`--v10_restrict`, hard-restrict loss to v10 Top-5 picks via precomputed masks; torch-free `make_v10_pickmask.py` to dodge a Windows OpenMP/MKL segfault). Consistent short Δ +2.04/+1.99/+1.62 across K (best K1@6 +5.83 t=1.21).
- **Verification (net vs raw WR)**: short K1 raw WR **58%** but **net WR collapses to 55% @6 / 51% @10 / 44% @20**; net@10 = +1.83 bps **t=0.38 = noise**. K3/K5 net-negative at 10bps. Classic **hit-rate ≠ edge**: costs convert a real 58% directional accuracy (z≈2.85) into a coin-flip P&L.
- **Gemini-veto "rescue" rejected**: proposed adding the live Gemini S1/S2 veto to cut the gate's losers. Rejected on three grounds — (1) no measured uplift (Gemini already layered on v8/v10, still net-negative); (2) it's the **MV2 meta-veto stacking line, permanently closed** (best stacked meta-model net −5 bps); (3) the S2 search-grounded LLM **cannot be honestly backtested** (querying today about a past date = lookahead). Only intellectually honest path is a **forward/paper test** (log live Gemini decisions on gate-kept shorts, net with vs without, to t>2).
- **Verdict**: ❌ does not clear the kill bar. Custom loss was the RIGHT lever (user's instinct vindicated) and extracted the most that's there — a faint, consistent, sub-significance / sub-cost short veto. Binding constraint is **information, not loss/architecture**. Committed (`5 files +849`, then net-WR follow-up). No keep_rate/λ sweep per stop rule.

## 🔗 Core Memory Links & Backlinks
- [[02 — Models/Transformer/Sided Transformer Preregistration|Pre-registration + full results]]
- [[02 — Models/Transformer/DualRes Transformer netPnL10 Report|Single-head DualRes veto report]]
- [[02 — Models/Meta-Veto/Meta-Veto Rectification Plan MV2|Meta-Veto MV2 (closed stacking line)]]
- [[01 — Architecture/Execution & Runtime/AI Veto & Gemini Audit|Gemini Veto Layer]]
- Way-through (all arms agree): order-flow/microstructure data, not architecture — cf. [[02 — Models/Transformer/Cross-Sectional Transformer Architecture Proposal]]
