"""
Level-Graph Gated GCN (cross-sectional).

Per ticker at a decision timestamp: a small market-structure graph (NOW + nearest
S/R / order-block / FVG / round nodes, from scripts/structural/level_graph). An
edge-gated graph conv (Bresson-Laurent / GGNN style) message-passes over the level
graph; the NOW node's readout is the per-ticker STRUCTURAL token. Then — exactly like
DualResCSTransformer — sector + macro context are added and all tickers attend to each
other (cross-sectional encoder) before a per-ticker logit head.

The ONLY architectural difference from the listwise transformer baseline is the encoder
(level-graph gated GCN vs dual-res temporal TA): same cross-sectional layer, same macro/
sector context, same objective/eval. So any rank-IC delta is attributable to the GNN.
"""
import torch
import torch.nn as nn


class GatedGCNLayer(nn.Module):
    """Edge-gated message passing over a padded complete graph (masked)."""
    def __init__(self, d, e_dim=3, dropout=0.1):
        super().__init__()
        self.A = nn.Linear(d, d)      # target
        self.Bn = nn.Linear(d, d)     # source (for gate)
        self.C = nn.Linear(e_dim, d)  # edge
        self.W = nn.Linear(d, d)      # source (for message)
        self.U = nn.Linear(d, d)      # self update
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, h, mask, e):           # h:(G,K,d) mask:(G,K) e:(G,K,K,e_dim)
        hi = self.A(h).unsqueeze(2)          # (G,K,1,d) target i
        hj = self.Bn(h).unsqueeze(1)         # (G,1,K,d) source j
        eta = torch.sigmoid(hi + hj + self.C(e))          # (G,K,K,d) edge gate
        msg = eta * self.W(h).unsqueeze(1)                # (G,K,K,d) gated source message
        vj = mask[:, None, :, None].float()              # valid source j
        agg = (msg * vj).sum(2) / ((eta * vj).sum(2) + 1e-6)   # (G,K,d) gated mean
        h_new = h + self.drop(self.act(self.norm(self.U(h) + agg)))
        return h_new * mask.unsqueeze(-1)                 # zero padded nodes


class LevelGraphGCN(nn.Module):
    def __init__(self, node_dim, n_macro, n_sectors, d_model=64, gcn_layers=3,
                 c_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, d_model)
        self.gcn = nn.ModuleList([GatedGCNLayer(d_model, 3, dropout) for _ in range(gcn_layers)])
        self.readout_norm = nn.LayerNorm(d_model)
        self.sector_emb = nn.Embedding(n_sectors, d_model)
        self.macro_mlp = nn.Sequential(
            nn.Linear(n_macro, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        cs = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout,
                                        batch_first=True, norm_first=True, activation='gelu')
        self.cross = nn.TransformerEncoder(cs, c_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, nodes, node_mask, macro, sector_ids, pad_mask):
        """nodes (B,N,K,D)  node_mask (B,N,K)  macro (B,M)  sector_ids (N,)
        pad_mask (B,N) True=absent ticker.  -> logit (B,N)"""
        B, N, K, D = nodes.shape
        x = nodes.reshape(B * N, K, D)
        m = node_mask.reshape(B * N, K)
        # edge features from signed ATR-distance (feat 0) and above/below flag (feat 2)
        sd = x[:, :, 0]                                   # (G,K)
        diff = sd[:, :, None] - sd[:, None, :]           # (G,K,K)
        same = (x[:, :, 2][:, :, None] == x[:, :, 2][:, None, :]).float()
        e = torch.stack([diff, diff.abs(), same], dim=-1)   # (G,K,K,3)
        h = self.node_proj(x) * m.unsqueeze(-1)
        for layer in self.gcn:
            h = layer(h, m, e)
        now = self.readout_norm(h[:, 0]).reshape(B, N, -1)   # NOW readout = structural token
        tok = now + self.sector_emb(sector_ids).unsqueeze(0)
        tok = tok + self.macro_mlp(macro).unsqueeze(1)
        z = self.cross(tok, src_key_padding_mask=pad_mask)
        return self.head(z).squeeze(-1)
