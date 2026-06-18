---
title: Graph-Based Models & Open-Interest Information Layer
type: log
status: active
updated: 2026-06-14
---

# 💬 Conversation Context: Graph-Based Models & OI Information Layer

## 📌 Metadata
- **Conversation ID**: (not exposed this session)
- **Start Date**: 2026-06-14
- **Status**: 🟢 Active
- **Focus Area**: Research / Models (graph models, new information sources)

## 🎯 Objectives
- [ ] Decide which graph-model family + edge sources are worth exploring (vs. CST/transformer ceiling)
- [ ] Source two new exogenous data layers: NSE quarterly shareholding + Upstox historical F&O Open Interest
- [ ] **Gate 0** (pivotal): does OI add net-of-cost signal as plain node features in the existing daily ranker?
- [ ] Gate 1: structural graph (group+promoter+co-holding) → Neo4j GDS embeddings → XGBoost
- [ ] Gate 2 (only if 0+1 survive): temporal GNN with OI node features over the structural graph

## 💻 Active Code Files Modified
- [scripts/collectors/probe_upstox_oi.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/collectors/probe_upstox_oi.py) — access/depth feasibility probe (print-only)
- [scripts/collectors/upstox_login.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/collectors/upstox_login.py) — OAuth full-access token minter
- [scripts/structural/business_groups.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/structural/business_groups.py) — 11 promoter-house groups over 40/172 tickers
- [scripts/structural/build_relation_graph.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/structural/build_relation_graph.py) — networkx graph + topology/spectral features
- Planned: graph panel features + Gate-1 WF eval; OI forward-logger + collector scaffold

## 🔁 /loop build status (dynamic mode)
- [x] **Chunk 1 — structural graph foundation**: `data/research/graph/{edges.csv(1209: 1130 sector + 79 group), node_features.csv(172×19, no feature NaNs), meta.json}`; 11 Louvain communities; PageRank used (eigenvector-centrality was NaN on disconnected sector graph). networkx path (no Neo4j installed).
- [x] **Chunk 2 — Gate-1 eval (single split)**: [scripts/structural/gate1_eval.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/structural/gate1_eval.py). Result (478 OOS days, Y_3d, cost 10bps, k10; cost-sanity PASS −10/−10): BASE rankIC 0.0132(t2.55), **+DYNAMIC 0.0253(t5.02)** L+11.2/S+8.5bps, **NEG-CONTROL 0.0133** (kills it → not a fixed-effect artifact). Decomp: **gain is mostly SECTOR (+DYN_SEC 0.0251) not the novel group edges (+DYN_GRP 0.0155)**; static topo HURTS (+BOTH 0.0180). ⚠️ SINGLE SPLIT — exactly the [[project_v8_1h_walkforward_demoted|v8 static-inflation trap]]; NOT believed until purged WF.
- [x] **Chunk 3 — purged walk-forward** (5 folds, expanding, purge 3): [scripts/structural/gate1_walkforward.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/structural/gate1_walkforward.py). **rankIC SURVIVES WF** (BASE 0.0170 → +DYNAMIC 0.0266, mean Δ +0.010, 4/5 folds positive — NOT a v8 collapse). But **net-of-cost trade NOT robust**: long +23bps high-variance (one fold neg), **short net-NEGATIVE −5bps mean** (fold3 −48). Verdict: graph features = real modest WF-robust *ranker enhancer*, not a standalone trade. **Gate 2 GNN NOT justified** (info-limited; 1-layer message-pass already captures it).
- [x] **Chunk 4 — OI track**: [scripts/collectors/log_option_chain_oi.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/collectors/log_option_chain_oi.py) — option-chain OI forward-logger, **works without Plus** (smoke-tested RELIANCE/TCS/HDFCBANK → PCR/max-pain/conc), appends to `data/oi_snapshots/option_chain_oi.csv`; run daily after close to accrue history. [scripts/collectors/collect_upstox_oi.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/collectors/collect_upstox_oi.py) — futures-OI collector scaffold, **fails gracefully on Plus paywall**, ready when Plus active. Added `.gitignore: data/raw_upstox_oi_cache/`.

## 🏁 Loop concluded (build-everything-as-planned)
Buildable scope DONE. Remaining: **Gate 0 blocked on user's Upstox-Plus decision**; Gate 2 GNN not justified (info-limited). Memories written: [[project_gate1_graph_features]], [[project_oi_plus_paywall]].
- ⛔ **Gate 0 (OI on history) blocked** — needs paid Upstox Plus (`UDAPI1149`); awaiting user decision.

## 📝 Compacted Session Log
- **Framing**: User asked about unexplored graph-based models (Neo4j + direction prediction). Grounded against repo's hard lesson — bottleneck is *information, not architecture* ([[04 — Research/CST Stage-0 Lead-Lag|CST dead-on-arrival]], DualRes/sided transformers info-limited). A cross-sectional transformer is already a dense learned graph over the 172 nodes; a price-derived GNN ≈ reproduces the CST ceiling.
- **Key reframe**: Panels are *already graph-shaped* `[T, 172 nodes, F]`. The 81/55 existing features are all OHLCV-derived **node features**. The repo has **zero edge data** — that is the entire opportunity. A graph helps only if edges carry *exogenous* information.
- **Two new data sources confirmed available by user**:
  1. **Upstox historical F&O OI — by date (deep history confirmed)** → first genuinely *non-price* signal. Highest EV. Best as **node features** (PCR, ΔOI×Δprice buildup quadrant, max-pain dist, futures basis, OI concentration).
  2. **NSE quarterly shareholding pattern** → slow (quarterly, ~21–45d lag). Yields node features (FII/DII/promoter/pledge %), promoter-group edges, and a *partial* co-holding graph (>1% named holders). For dense co-holding use AMFI monthly MF portfolios.
- **Unifying thesis**: slow structural connections = **edges** (business-group + promoter + co-holding + sector); fast OI positioning = **node signal**; hypothesis = *positioning lead-lag* (not price lead-lag, which is dead) propagates along structural edges → GNN captures spillover flat models miss.
- **Infra recon**: `UpstoxSandboxBroker` exposes analytics/live data client (`UPSTOX_ANALYTICS_ACCESS_TOKEN`); collectors use per-ticker CSV cache + chunked dates + rate pauses; `instrument_cache.json` maps TICKER→`NSE_EQ|ISIN` (F&O contract keys still needed via instrument master).
- **Discipline**: all exploratory tier (no Gauntlet verdict authority); must clear ~10bps net; OI strictly as-of (EOD-T → predict T+1); shareholding keyed to disclosure date not quarter-end.

- **2026-06-14 BLOCKER (probe artifact)**: Upstox expired-instruments API confirmed via docs — `/expiries` → `/future/contract` → `/historical-candle` (OI at field [6]); requires Upstox Plus. Probe run against `UPSTOX_ANALYTICS_ACCESS_TOKEN`: **401 `UDAPI100067` "not permitted with a read only token"** on `/expiries`. → cheap futures-OI path is **token-scope blocked**, not built yet. `/v2/market/oi` for an old date returned `success` + `data:null` (inconclusive — wrong expiry or also gated). **Next: need a full-access (non read-only) daily token w/ Plus, then re-run probe.** Docs `/expiries` also hint depth ≈ 6 months → verify before committing to a thin Gate-0 window.

- **2026-06-14 token diagnosis (JWT decode artifact)**: ANALYTICS token `isPlusPlan=False, isExtended=True` (extended/read-only → double-blocked); SANDBOX token `isPlusPlan=True` but expired 2026-06-12 + sandbox-only. ⇒ account likely *has* Plus; need a fresh **live full-access non-extended** token (`isPlusPlan=True, isExtended!=True`). Helper written: [scripts/collectors/upstox_login.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/collectors/upstox_login.py) (OAuth exchange → writes `UPSTOX_FULL_ACCESS_TOKEN`, decodes claims). Correct claim name is `isPlusPlan` (not `isPlus`).

## 🔗 Core Memory Links & Backlinks
- [[04 — Research/CST Stage-0 Lead-Lag]] — why price-derived graphs hit a ceiling
- [[02 — Models/Daily/Model Card - daily_macro_v2]] — candidate Gate-0 baseline ranker
