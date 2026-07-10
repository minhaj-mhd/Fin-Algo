---
title: "Financial Machine Learning (Kelly & Xiu 2023) — Reference & Relevance"
type: reference
status: active
updated: 2026-06-28
tags: [reference, literature, machine-learning, asset-pricing]
---
# 📄 Financial Machine Learning — Kelly & Xiu (2023)

> External literature reference (not our research). Survey of ML in financial markets;
> strong corroboration of this repo's "information-ceiling, net-of-cost" findings.

## Citation & access
- **Bryan T. Kelly (Yale / AQR) & Dacheng Xiu (Chicago Booth)**, *Financial Machine Learning*, 2023. 160 pp.
- **NBER Working Paper 31502** — https://www.nber.org/papers/w31502 (DOI 10.3386/w31502)
- **SSRN (free PDF)** — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4501707
- **AQR (free working-paper PDF)** — https://www.aqr.com/Insights/Research/Working-Paper/Financial-Machine-Learning
- **Published:** *Foundations and Trends in Finance*, 13(3–4), 205–363.

## What it covers (digest — paraphrased, not reproduced)
Frames empirical asset pricing as a **prediction problem** and surveys ML applied to it:
- **Return prediction**: estimating E[return | characteristics] from a high-dimensional predictor
  set — penalized regression (ridge/LASSO/elastic-net), dimension reduction (PCA/PLS/autoencoders),
  tree ensembles (random forests, gradient boosting), and neural networks.
- **SDF / factor models**: ML estimation of the stochastic discount factor and conditional factor
  models (e.g. instrumented PCA, autoencoder asset pricing).
- **Portfolios**: mapping predictions into optimal weights ("Markowitz meets ML").
- **The "virtue of complexity"**: heavily over-parameterized models (more parameters than
  observations) can still improve out-of-sample performance.
- **Future directions**: alternative data, NLP/textual signals, higher-frequency data, reinforcement
  learning for trading/execution, and the central role of **transaction costs**.

## Why it matters for us (maps onto our results)
1. **Low signal-to-noise is the defining feature of financial ML.** Genuine predictability is *tiny*
   per prediction (an OOS monthly R² of ~0.5–1% is considered large). Our sub-0.03 rank-IC results are
   the *expected* regime, not a failure — directly supports the repo's info-ceiling line.
2. **ML edge is documented in the cross-section of many assets at monthly/daily horizons**, via
   flexibility (nonlinearity/interactions) + heavy regularization. That is *not* the single-name,
   high-frequency, few-predictor regime our 1h ranker sits in — explains why 1h price/volume is so hard.
3. **No universal architecture winner**; trees and neural nets both feature. The survey does NOT claim
   transformers dominate tabular cross-sectional prediction — consistent with our benchmark where
   XGBoost beat the transformer ~5× ([[02 — Models/Transformer/Transformer Architectures Tried — Catalog]]).
4. **Statistical predictability ≠ tradeable profit** — they stress transaction costs as the gap. That
   is exactly our "sub-cost" wall.

## Pointers it suggests for our roadmap
- Lean toward **cross-sectional, lower-frequency** signals and **cost-aware portfolio construction**.
- **Alternative data / NLP / order-flow** as the information lever (matches our order-flow conclusion).
- Factor/SDF methods (IPCA, autoencoder asset pricing) as a comparison point for [[03 — Strategies/.. daily_macro_v2|daily_macro_v2]].

## Links
- [[02 — Models/Transformer/Transformer Architectures Tried — Catalog|Transformer Architectures Tried]]
- [[04 — Research/V20 Rolling-1h Overlapping-Window Model]] · info-ceiling line (CST / PA-SMC / dual-TF dead-ends)
