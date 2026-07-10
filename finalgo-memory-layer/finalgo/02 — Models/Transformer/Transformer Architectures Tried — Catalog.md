---
title: "Transformer Architectures Tried — Catalog"
type: reference
status: active
model: "Transformer"
updated: 2026-06-28
tags: [transformer, catalog, dead-end, info-ceiling]
---
# 🧠 Transformer Architectures Tried — Catalog

> One place listing every transformer (and transformer-adjacent deep) architecture attempted on the
> intraday/daily ranking problem, with its verdict. **All numbers are research-tier point estimates —
> ⚠️ UNVERIFIED, no Gauntlet `run_id` exists for any of these.** Bottom line up front: **every variant
> is sub-cost / info-limited, and gradient-boosted trees (v10/v20/v21 XGBoost) meet or beat them on
> rank-IC.** The binding constraint is *information*, not architecture/loss/capacity. External
> corroboration: [[04 — Research/Financial Machine Learning (Kelly & Xiu 2023)]].

## Summary table
| # | Architecture | Target / role | Verdict (⚠️ UNVERIFIED) |
|---|---|---|---|
| 1 | Single-head P(up) BCE cross-sectional transformer | 1h up-prob | superseded by sided design; faint |
| 2 | **CST** (lead-lag cross-sectional) | 1h ranker | **DEAD on arrival** — lead-lag Δρ −0.003 (arbitraged at 1h) |
| 3 | **DualRes** dual-resolution (1h+15m) direction+confidence | 1h ranker | faint (AUC 0.521, +6.4bps gross) but **sub-cost** (net −3.6bps @10bps) |
| 4 | **Sided** dual side-specialist (listwise) + cost-aware **veto-gate loss** | long/short rankers | listwise **DEAD** (ρ 0.006); gate-loss is the *right tool* (short Δ +1.6–2.0bps vs v10) but **no t>2, sub-cost** |
| 5 | **Co-sign v10 decision layer** (directional confirmation) | trade iff v10 ∧ TF agree | **DEAD** — OOS uplift overfits from epoch 2; neg-control reproduces it |
| 6 | **Daily transformer veto** overlay on daily_macro_v2 | daily veto gate | **DEAD** — no significant net improvement; only CI>0 cell failed neg-control |
| 7 | **BCE veto transformer** (+ Optuna loss-zoo) | veto gate | plain_bce won; **no TEST improvement** over +1.14/t2.27 baseline; seed-fragile |
| 8 | **PA/SMC transformer** (27 smart-money feats) + **level-graph gated GCN** | 1h ranker | **DEAD** — rank-IC Δ ≈ noise; GCN short ρ +0.0032 < baseline |
| 9 | **Contrastive pretrain + FAISS retrieval hybrid** → LightGBM ranker | 1h top-k | marginal: **+0.94 bps / +1.17 Sharpe** top-3 LONG only |
| 10 | **DualRes listwise on CLEAN v21 data** (2026-06-28) | 1h ranker | **MISSES** — long ρ 0.0041 / short 0.0087 vs **XGBoost 0.0223 / 0.0192**; neg-control ρ≈0 |

## Details & links
1. **Single-head P(up) BCE** — original cross-sectional transformer (binary up-prob head). Superseded by the sided ranking design. See [[02 — Models/Transformer/Cross-Sectional Transformer Architecture Proposal]].
2. **CST (Cross-Sectional Transformer)** — lead-lag/cross-asset attention. Stage-0 falsification: lead-lag features add Δρ −0.003 at 1h granularity (signal arbitraged away). **Do not build.** [[02 — Models/Transformer/Cross-Sectional Transformer Architecture Proposal]].
3. **DualRes** — 1h+15m dual-resolution cross-sectional direction+confidence transformer (GPU). Faint signal, **sub-cost**; dataset is enough to train but **information-limited, not size-limited** (data-size ablation flat). Panels `data/transformer_panel/`. [[02 — Models/Transformer/DualRes Cross-Sectional Transformer Architecture]], [[02 — Models/Transformer/DualRes Transformer netPnL10 Report]], [[02 — Models/Transformer/DualRes Transformer Flowchart]].
4. **Sided dual side-specialist** — `dualres_long`/`dualres_short` listwise rankers + a custom **cost-aware veto-gate loss** (the correct objective). Listwise ranker dead (ρ 0.006); gate loss gives a consistent small short edge vs v10 but no cell hits t>2 and it's sub-cost at 10bps. `train.py --objective gate [--v10_restrict]`, `gate_veto_v10.py`. [[02 — Models/Transformer/Sided Transformer Preregistration]].
5. **Co-sign v10 decision layer** — trained-for-the-job directional confirmation (trade iff v10 AND transformer agree). Loss descends, but OOS uplift overfits to noise from epoch 2; no net-positive cell @10bps; negative control reproduces the "uplift" → capacity surplus, **not** more layers. Info-limited.
6. **Daily transformer veto** — daily-only cross-sectional transformer as a veto overlay on daily_macro_v2. No significant net improvement (LONG +2.2bps t1.32; SHORT +4.9/+6.2bps t<1.6); the only CI>0 cell failed the neg-control. Panel `data/daily_transformer_panel/`.
7. **BCE veto transformer** — Optuna over a "loss zoo"; plain BCE won, no TEST improvement over the +1.14/t2.27 baseline, seed-fragile. Fixed a single-shuffle neg-control bug. [[04 — Research/BCE-Transformer-V20-Veto]].
8. **PA/SMC transformer + level-graph gated GCN** — 27 explicit smart-money features (S/R, candles, order blocks, FVGs, sweeps) added to the 81-TA transformer in a byte-identical controlled test → no tradable edge. A purpose-built level-graph gated GCN was also dead (short test ρ +0.0032 < +0.0066 baseline; neg-control ≈ same). Structure contributes ~nothing at 1h.
9. **Contrastive pretrain + retrieval hybrid** — contrastive dual-resolution pre-training + FAISS bucket-filtered retrieval + time-decay + 12-D neighbor-return-distribution features → LightGBM ranker with graded relevance. Net uplift small and LONG-only (+0.94 bps net, +1.17 Sharpe on top-3 long).
10. **DualRes listwise on CLEAN v21 data (2026-06-28)** — pre-registered falsifiable benchmark: trained the DualRes transformer on the clean v21 tensor panel (`data/transformer_panel_v21/`, 110-name universe + `clean_v21` 1h features). On the SAME chrono test window + universe: **transformer long ρ 0.0041 / short 0.0087 vs v21 XGBoost 0.0223 / 0.0192** → misses on both sides (long ≈5× worse). Neg-control (`train.py --shuffle_labels`) ρ=0.0005 ≈0 (no leak; long signal barely above its own shuffle). **Definitively closes "transformer + cleaner data".** Reusable: `scripts/transformer/{build_tensor_panel_v21,benchmark_v21_xgb_vs_transformer}.py`. See [[06 — Logs/Daily Logs/2026-06-28]].

## Standing conclusion
Across **10 architecture/loss/capacity variants** (attention topology, dual-resolution, side-specialist
losses, cost-aware gates, GCN structure, contrastive+retrieval, clean data) the 1h price/volume edge
never clears cost, and **trees ≥ transformers** on rank-IC. The lever is **new information**
(order-flow / microstructure / alt-data) or a **different horizon** (the certified daily_macro_v2
multi-day edge; the intraday overnight-reversal open-auction edge) — **not** another deep architecture.

## Links
- Champion trees (beat these): [[02 — Models/_Shared/Model Performance & Statistics]]
- Clean-data ceiling: [[04 — Research/V20 Rolling-1h Overlapping-Window Model]]
- Literature: [[04 — Research/Financial Machine Learning (Kelly & Xiu 2023)]]
