"""
Gate-1 step 1: build the STATIC structural relation graph over the 172-ticker
universe and emit per-ticker topology features.

Edges (exogenous — NOT price-derived):
  - same business group (promoter house)  -> weight 1.0   [strong]
  - same sector                            -> weight 0.3   [weak]

Per-ticker static features emitted:
  degree_cent, eigen_cent, betweenness, clustering, group_size, in_group,
  sector_size, community (Louvain id), spectral emb_0..emb_{K-1}

NOTE (methodology): these are CONSTANT per ticker over time. In a per-day
cross-sectional ranker they behave like fixed effects — they can encode persistent
structural risk premia OR just memorize which names won in-sample. The Gate-1 eval
(next step) MUST therefore include a ticker-shuffled negative control and walk-forward
splits before any feature here is believed. Exploratory tier — no Gauntlet authority.

Output: data/research/graph/{edges.csv, node_features.csv, meta.json}
Run:    python scripts/structural/build_relation_graph.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import SpectralEmbedding

sys.path.append(os.getcwd())
from scripts.sector_map import SECTOR_MAP                      # noqa: E402
from scripts.structural.business_groups import group_of, base  # noqa: E402

PANEL_META = "data/daily_transformer_panel/meta.json"
OUT_DIR = "data/research/graph"
SECTOR_W = 0.3
GROUP_W = 1.0
EMB_DIM = 8
SEED = 42


def load_universe():
    """Authoritative ordered ticker list (base names) from the daily panel."""
    with open(PANEL_META) as f:
        meta = json.load(f)
    return [base(t) for t in meta["tickers"]]


def sector_of(tkr):
    return SECTOR_MAP.get(tkr + ".NS", "MISC")


def build_graph(tickers):
    G = nx.Graph()
    G.add_nodes_from(tickers)
    sect = {t: sector_of(t) for t in tickers}
    grp = {t: group_of(t) for t in tickers}
    # add edges (keep the strongest weight if a pair shares both group and sector)
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            w = 0.0
            etype = None
            if grp[a] and grp[a] == grp[b]:
                w, etype = GROUP_W, "group"
            elif sect[a] == sect[b]:
                w, etype = SECTOR_W, "sector"
            if w > 0:
                G.add_edge(a, b, weight=w, type=etype)
    return G, sect, grp


def topology_features(G):
    deg = nx.degree_centrality(G)
    # PageRank instead of eigenvector centrality: well-defined even when the
    # sector graph is disconnected (eigenvector centrality returned all-NaN there).
    pr = nx.pagerank(G, weight="weight")
    btw = nx.betweenness_centrality(G, weight=None, seed=SEED)
    clu = nx.clustering(G, weight="weight")
    comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    comm_id = {n: c for c, nodes in enumerate(comms) for n in nodes}
    return deg, pr, btw, clu, comm_id, len(comms)


def spectral_embedding(G, tickers):
    A = nx.to_numpy_array(G, nodelist=tickers, weight="weight")
    if not nx.is_connected(G):
        print("  [warn] graph not fully connected — adding tiny global affinity for stable embedding")
        A = A + 1e-4                       # weak all-to-all keeps the Laplacian connected
        np.fill_diagonal(A, 0.0)
    k = min(EMB_DIM, len(tickers) - 2)
    emb = SpectralEmbedding(n_components=k, affinity="precomputed",
                            random_state=SEED).fit_transform(A)
    return emb


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tickers = load_universe()
    G, sect, grp = build_graph(tickers)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    by_type = {}
    for _, _, d in G.edges(data=True):
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    print(f"  edges by type: {by_type}")
    grouped = sum(1 for t in tickers if grp[t])
    print(f"  tickers in a business group: {grouped}/{len(tickers)}")

    deg, pr, btw, clu, comm_id, n_comm = topology_features(G)
    print(f"  Louvain communities: {n_comm}")
    emb = spectral_embedding(G, tickers)

    sector_size = pd.Series(sect).value_counts().to_dict()
    group_size = pd.Series({t: grp[t] for t in tickers if grp[t]}).value_counts().to_dict()

    rows = []
    for i, t in enumerate(tickers):
        row = {
            "ticker": t, "sector": sect[t], "group": grp[t] or "",
            "degree_cent": deg[t], "pagerank": pr[t], "betweenness": btw[t],
            "clustering": clu[t], "community": comm_id[t],
            "in_group": int(bool(grp[t])),
            "group_size": group_size.get(grp[t], 0) if grp[t] else 0,
            "sector_size": sector_size[sect[t]],
        }
        for j in range(emb.shape[1]):
            row[f"emb_{j}"] = emb[i, j]
        rows.append(row)
    feat = pd.DataFrame(rows)
    feat.to_csv(os.path.join(OUT_DIR, "node_features.csv"), index=False)

    edges = [{"src": u, "dst": v, "weight": d["weight"], "type": d["type"]}
             for u, v, d in G.edges(data=True)]
    pd.DataFrame(edges).to_csv(os.path.join(OUT_DIR, "edges.csv"), index=False)

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                   "edges_by_type": by_type, "n_communities": n_comm,
                   "emb_dim": int(emb.shape[1]), "sector_w": SECTOR_W,
                   "group_w": GROUP_W, "seed": SEED,
                   "feature_cols": [c for c in feat.columns if c not in
                                    ("ticker", "sector", "group")]}, f, indent=2)
    print(f"  wrote node_features.csv ({feat.shape}), edges.csv ({len(edges)}), meta.json -> {OUT_DIR}")
    print(feat.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
