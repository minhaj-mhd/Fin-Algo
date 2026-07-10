"""
Dual-Resolution Cross-Sectional Transformer.

One decision = one 1h timestamp t. For every ticker present at t the model sees:
  * its last 30 x 1h candles  (incl. the 14:15 2:15-3:15 context candle)  -> 1h encoder
  * its last 60 x 15m candles  (aligned to close WITH the 1h bar)          -> 15m encoder
  * a learned sector embedding
Each candle is identified in TIME two ways: sinusoidal positional encoding (order within the
window) + a learned clock-time-of-day slot embedding (1h: 6 slots 09:15..14:15; 15m: 25 slots
09:15..15:15) so the model can tell the open hour from the close hour and see day boundaries.
Then ALL tickers' per-ticker tokens attend to EACH OTHER (cross-sectional encoder) with a
broadcast daily macro/VIX/breadth/global context (FiLM-style add).
Head -> one logit per ticker = P(next 1h candle is UP). Confidence = sigmoid(logit).
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):                     # x: (B, L, d)
        return x + self.pe[:, :x.size(1)]


class TemporalEncoder(nn.Module):
    """Transformer encoder over a single ticker's candle sequence -> pooled embedding."""
    def __init__(self, n_feat, d_model, nhead, nlayers, dim_ff, dropout):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, slot_emb=None):      # x: (B', L, F)  slot_emb: (B', L, d)
        h = self.proj(x)
        if slot_emb is not None:
            h = h + slot_emb                  # inject clock-time-of-day identity
        h = self.enc(self.pos(h))             # (B', L, d)
        return self.norm(h[:, -1] + h.mean(dim=1))   # last-step + mean-pool


class DualResCSTransformer(nn.Module):
    def __init__(self, n_feat, n_macro, n_sectors, n_slots_1h=6, n_slots_15m=25,
                 d_model=64, t_layers=2, c_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.enc_1h = TemporalEncoder(n_feat, d_model, nhead, t_layers, 2 * d_model, dropout)
        self.enc_15m = TemporalEncoder(n_feat, d_model, nhead, t_layers, 2 * d_model, dropout)
        self.slot_emb_1h = nn.Embedding(n_slots_1h, d_model)
        self.slot_emb_15m = nn.Embedding(n_slots_15m, d_model)
        self.sector_emb = nn.Embedding(n_sectors, d_model)
        self.macro_mlp = nn.Sequential(
            nn.Linear(n_macro, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.token_proj = nn.Linear(2 * d_model, d_model)
        cs_layer = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout,
                                              batch_first=True, norm_first=True, activation='gelu')
        self.cross = nn.TransformerEncoder(cs_layer, c_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x1h, x15m, slot1h, slot15m, macro, sector_ids, pad_mask):
        """
        x1h (B,N,L1,F)  x15m (B,N,L2,F)  slot1h (B,L1)  slot15m (B,L2)
        macro (B,M)  sector_ids (N,)  pad_mask (B,N) True=absent.  -> logit (B,N)
        """
        B, N, L1, F = x1h.shape
        L2 = x15m.size(2)
        se1 = self.slot_emb_1h(slot1h).unsqueeze(1).expand(B, N, L1, -1).reshape(B * N, L1, -1)
        se2 = self.slot_emb_15m(slot15m).unsqueeze(1).expand(B, N, L2, -1).reshape(B * N, L2, -1)
        h1 = self.enc_1h(x1h.reshape(B * N, L1, F), se1).reshape(B, N, -1)
        h2 = self.enc_15m(x15m.reshape(B * N, L2, F), se2).reshape(B, N, -1)
        tok = self.token_proj(torch.cat([h1, h2], dim=-1))            # (B,N,d)
        tok = tok + self.sector_emb(sector_ids).unsqueeze(0)          # broadcast sector
        tok = tok + self.macro_mlp(macro).unsqueeze(1)                # broadcast macro context
        z = self.cross(tok, src_key_padding_mask=pad_mask)           # cross-sectional attention
        return self.head(z).squeeze(-1)                               # (B,N)
