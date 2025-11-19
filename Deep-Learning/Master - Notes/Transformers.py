# transformer_from_scratch.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- utility: positional encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # buffer, not a parameter

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

# ---- Scaled dot-product attention (batch-friendly) ----
def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: (B, heads, T, d_k)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, T_q, T_k)
    if mask is not None:
        # mask should be broadcastable to (B, 1, T_q, T_k) or (B, h, T_q, T_k)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # (B, h, T_q, d_k)
    return out, attn

# ---- Multi-head attention ----
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        # Linear projections
        q = self.w_q(q)  # (B, T_q, d_model)
        k = self.w_k(k)
        v = self.w_v(v)
        # reshape to (B, h, T, d_k)
        def reshape(x):
            return x.view(B, -1, self.h, self.d_k).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        # attention per head
        attn_out, attn = scaled_dot_product_attention(q, k, v, mask)
        # attn_out: (B, h, T, d_k) -> concat heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.w_o(attn_out)
        return self.dropout(out), attn  # out: (B, T, d_model)

# ---- Feed-forward ----
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)

# ---- Encoder layer ----
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# ---- Decoder layer ----
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        # masked self-attention (causal)
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # encoder-decoder cross-attention
        attn2, _ = self.cross_attn(x, enc_out, enc_out, memory_mask)
        x = self.norm2(x + self.dropout(attn2))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

# ---- Encoder / Decoder stacks ----
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.tok_embed(src)  # (B, T, d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, tgt_mask=None, memory_mask=None):
        x = self.tok_embed(tgt)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        return self.norm(x)

# ---- Full Transformer ----
class TransformerSimple(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, N=2, heads=4, d_ff=512, max_len=200, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, heads, d_ff, max_len, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc = self.encoder(src, src_mask)
        dec = self.decoder(tgt, enc, tgt_mask, None)
        logits = self.out(dec)  # (B, T_tgt, tgt_vocab)
        return logits

# ---- Mask helpers ----
def make_pad_mask(seq, pad_idx=0):
    # seq: (B, T)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T) broadcastable

def make_causal_mask(size):
    # returns lower-triangular mask for causal attention (1s where allowed)
    return torch.tril(torch.ones((size, size), dtype=torch.uint8)).unsqueeze(0).unsqueeze(1)  # (1,1,T,T)
