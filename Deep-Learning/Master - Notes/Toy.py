# Toy.py -- minimal, self-contained Transformer (encoder+decoder) + toy training
# Save as Toy.py and run: python Toy.py

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)            # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                          # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

# -------------------------
# Scaled dot-product attention
# -------------------------
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v: (B, heads, T, d_k)
    mask: None or boolean mask broadcastable to (B, 1, T_q, T_k) or (B, heads, T_q, T_k)
          mask==True indicates positions *allowed* (i.e., not masked).
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, heads, T_q, T_k)
    if mask is not None:
        # mask is allowed positions True; we want to set disallowed to -inf
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # (B, heads, T_q, d_k)
    return out, attn

# -------------------------
# Multi-head attention
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
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
        # linear projections
        q = self.w_q(q)  # (B, T_q, d_model)
        k = self.w_k(k)  # (B, T_k, d_model)
        v = self.w_v(v)  # (B, T_k, d_model)

        # reshape -> (B, heads, T, d_k)
        def reshape(x):
            return x.view(B, -1, self.h, self.d_k).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        # adjust mask shape if provided: convert to (B, 1, T_q, T_k) or (B, heads, T_q, T_k)
        if mask is not None:
            # if mask is (B,1,1,T_k) or (B,1,T_q,T_k) it's broadcastable
            if mask.dim() == 4 and mask.size(1) == 1:
                # expand heads dimension to match (B, heads, T_q, T_k)
                mask = mask.expand(-1, self.h, -1, -1)
        attn_out, attn = scaled_dot_product_attention(q, k, v, mask)
        # concat heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.w_o(attn_out)
        return self.dropout(out), attn

# -------------------------
# Feed-forward
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Encoder layer
# -------------------------
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

# -------------------------
# Decoder layer
# -------------------------
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
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        attn2, _ = self.cross_attn(x, enc_out, enc_out, memory_mask)
        x = self.norm2(x + self.dropout(attn2))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

# -------------------------
# Encoder / Decoder stacks
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.tok_embed(src)                 # (B, T, d_model)
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

# -------------------------
# Full Transformer
# -------------------------
class TransformerSimple(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, N=2, heads=4, d_ff=512, max_len=200, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, heads, d_ff, max_len, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc = self.encoder(src, src_mask)
        dec = self.decoder(tgt, enc, tgt_mask, memory_mask)
        logits = self.out(dec)  # (B, T_tgt, tgt_vocab)
        return logits

# -------------------------
# Mask helpers
# -------------------------
def make_pad_mask(seq, pad_idx=0):
    # seq: (B, T)
    # returns boolean mask with True where tokens are NOT padding (allowed)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)

def make_causal_mask(size, device=None):
    # returns lower-triangular boolean mask of shape (1,1,size,size) with True where j <= i
    m = torch.tril(torch.ones((size, size), dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(1)  # (1,1,T,T)

# -------------------------
# Toy training: copy task
# -------------------------
def random_batch(batch_size, seq_len, vocab_size, pad_idx=0):
    # generate random sequences of ints in 1..vocab_size-1 (0 reserved for PAD)
    seq = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return seq, seq  # src, tgt (copy task)

def train_toy():
    # hyperparams (small so it runs quickly)
    SRC_VOCAB = 11   # tokens 0..10; 0 = PAD
    TGT_VOCAB = SRC_VOCAB
    BATCH = 32
    SRC_LEN = 10
    TGT_LEN = SRC_LEN
    EPOCHS = 200
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", DEVICE)

    model = TransformerSimple(SRC_VOCAB, TGT_VOCAB, d_model=64, N=2, heads=4, d_ff=256, max_len=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD

    for ep in range(1, EPOCHS + 1):
        model.train()
        src, tgt = random_batch(BATCH, SRC_LEN, SRC_VOCAB)
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        # masks
        src_mask = make_pad_mask(src).to(DEVICE)           # (B,1,1,T_src)
        tgt_mask = make_causal_mask(TGT_LEN, device=DEVICE)  # (1,1,T_tgt,T_tgt)
        # memory_mask for cross-attention (allow all non-pad memory positions)
        memory_mask = make_pad_mask(src).to(DEVICE)        # (B,1,1,T_src) -> expanded in MHA

        # prepare decoder input (teacher forcing): start token = 1
        start_token = 1
        tgt_input = torch.cat([torch.full((BATCH, 1), start_token, dtype=torch.long, device=DEVICE), tgt[:, :-1]], dim=1)

        logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)  # (B, T, V)
        loss = criterion(logits.view(-1, TGT_VOCAB), tgt.view(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 20 == 0 or ep == 1:
            print(f"Epoch {ep}/{EPOCHS} - loss: {loss.item():.4f}")

    # quick test on small batch
    model.eval()
    with torch.no_grad():
        src, tgt = random_batch(4, SRC_LEN, SRC_VOCAB)
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = torch.cat([torch.full((4, 1), 1, dtype=torch.long, device=DEVICE), tgt[:, :-1]], dim=1)
        logits = model(src, tgt_input, src_mask=make_pad_mask(src).to(DEVICE),
                       tgt_mask=make_causal_mask(TGT_LEN, device=DEVICE),
                       memory_mask=make_pad_mask(src).to(DEVICE))
        pred = logits.argmax(-1)
        print("\nExample (small batch) — src, tgt, pred:")
        print("src:\n", src.cpu().numpy())
        print("tgt:\n", tgt.cpu().numpy())
        print("pred:\n", pred.cpu().numpy())

if __name__ == "__main__":
    # reproducibility
    random.seed(0)
    torch.manual_seed(0)
    train_toy()
