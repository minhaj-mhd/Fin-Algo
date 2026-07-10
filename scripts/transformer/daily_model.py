"""
Daily-only Cross-Sectional Transformer (veto-overlay variant).

Single resolution: one decision = one trading day. For every ticker present that day the model sees
its last L daily candles (per-day cross-sectionally z-scored features) -> daily TemporalEncoder.
Each bar carries a day-of-week embedding (the daily analog of the intraday time-of-day slot) on top of
sinusoidal positional order. Per-ticker tokens then attend to EACH OTHER (cross-sectional encoder) with
a broadcast daily macro/VIX/breadth/global FiLM context. Head -> one score per ticker.

Reuses PositionalEncoding / TemporalEncoder from model.py (same attention block: nn.TransformerEncoder,
pre-LN, GELU, multi-head scaled-dot-product). Deliberately SMALL + heavily regularized: daily data is
sample-starved (~2,470 days, ~5 independent names/day), so capacity is capped to avoid memorization.
"""
import torch
import torch.nn as nn

from scripts.transformer.model import PositionalEncoding, TemporalEncoder


class DailyCSTransformer(nn.Module):
    def __init__(self, n_feat, n_macro, n_sectors, n_dow=5,
                 d_model=48, t_layers=2, c_layers=2, nhead=4, dropout=0.4):
        super().__init__()
        self.enc = TemporalEncoder(n_feat, d_model, nhead, t_layers, 2 * d_model, dropout)
        self.dow_emb = nn.Embedding(n_dow, d_model)
        self.sector_emb = nn.Embedding(n_sectors, d_model)
        self.macro_mlp = nn.Sequential(
            nn.Linear(n_macro, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        cs_layer = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout,
                                              batch_first=True, norm_first=True, activation='gelu')
        self.cross = nn.TransformerEncoder(cs_layer, c_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x, dow, macro, sector_ids, pad_mask):
        """
        x (B,N,L,F)  dow (B,L)  macro (B,M)  sector_ids (N,)  pad_mask (B,N) True=absent
        -> score (B,N)
        """
        B, N, L, F = x.shape
        de = self.dow_emb(dow).unsqueeze(1).expand(B, N, L, -1).reshape(B * N, L, -1)
        h = self.enc(x.reshape(B * N, L, F), de).reshape(B, N, -1)        # (B,N,d)
        h = h + self.sector_emb(sector_ids).unsqueeze(0)                  # broadcast sector
        h = h + self.macro_mlp(macro).unsqueeze(1)                        # broadcast macro context (FiLM)
        z = self.cross(h, src_key_padding_mask=pad_mask)                  # cross-sectional attention
        return self.head(z).squeeze(-1)                                   # (B,N)
