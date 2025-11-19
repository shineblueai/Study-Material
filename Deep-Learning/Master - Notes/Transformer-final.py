# Interactive_Transformer.py
# Save and open in Jupyter/VSCode Interactive to run cell-by-cell.
# This file is organized into "cells" using '# %%' markers so you can execute
# each block independently and inspect intermediate outputs.
#
# Features:
# - Full encoder + decoder Transformer (readable, educational)
# - Debug mode prints shapes and intermediate tensors at every major step
# - Functions to visualize attention maps
# - Clear places (TODO comments) where you can add your own comments/notes
# - Works with CPU or GPU automatically

# %%
# 0) Imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional

# plotting utilities (works in notebook)
import matplotlib.pyplot as plt

# %%
# 1) Utility: positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]

# %%
# 2) Scaled dot-product attention (returns attention weights as well)
def scaled_dot_product_attention(q,k,v,mask:Optional[torch.Tensor]=None):
    # q,k,v: (B, heads, T, d_k)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)  # (B, heads, T_q, T_k)
    if mask is not None:
        # mask True means allowed (so invert to set disallowed to -inf)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn

# %%
# 3) MultiHeadAttention (with debug prints)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.1):
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

    def _split_heads(self, x:torch.Tensor):
        # x: (B, T, d_model) -> (B, heads, T, d_k)
        B, T, _ = x.size()
        return x.view(B, T, self.h, self.d_k).transpose(1,2)

    def _combine_heads(self, x:torch.Tensor):
        # x: (B, heads, T, d_k) -> (B, T, d_model)
        B, h, T, d_k = x.size()
        return x.transpose(1,2).contiguous().view(B, T, h * d_k)

    def forward(self, q,k,v, mask:Optional[torch.Tensor]=None, debug:bool=False):
        # q,k,v: (B, T, d_model)
        B = q.size(0)
        # linear projections
        q_lin = self.w_q(q)  # (B,T,d_model)
        k_lin = self.w_k(k)
        v_lin = self.w_v(v)
        if debug:
            print(f"[MHA] after linear: q_lin {q_lin.shape}, k_lin {k_lin.shape}, v_lin {v_lin.shape}")

        q_ = self._split_heads(q_lin)  # (B,h,T,d_k)
        k_ = self._split_heads(k_lin)
        v_ = self._split_heads(v_lin)
        if debug:
            print(f"[MHA] split heads: q_ {q_.shape}, k_ {k_.shape}, v_ {v_.shape}")

        # adapt mask shape if needed: accept (B,1,1,T) or (B,1,T,T) or (1,1,T,T)
        if mask is not None and mask.dim() == 4 and mask.size(1) == 1:
            # (B,1,1,T_k) or (B,1,T_q,T_k) -> expand heads
            mask = mask.expand(-1, self.h, -1, -1)
            if debug:
                print(f"[MHA] expanded mask to {mask.shape}")

        attn_out, attn_weights = scaled_dot_product_attention(q_, k_, v_, mask)
        if debug:
            print(f"[MHA] attn_out {attn_out.shape}, attn_weights {attn_weights.shape}")

        concat = self._combine_heads(attn_out)  # (B,T,d_model)
        out = self.w_o(concat)
        out = self.dropout(out)
        if debug:
            print(f"[MHA] output combined {out.shape}")
        return out, attn_weights

# %%
# 4) Positionwise Feed-Forward
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x:torch.Tensor):
        return self.net(x)

# %%
# 5) Encoder and Decoder layers (with debug hooks)
class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, heads:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask:Optional[torch.Tensor]=None, debug:bool=False):
        if debug: print(f"[EncoderLayer] input x {x.shape}")
        attn_out, attn_w = self.self_attn(x,x,x, mask=src_mask, debug=debug)
        x = self.norm1(x + self.dropout(attn_out))
        if debug: print(f"[EncoderLayer] after self-attn + norm {x.shape}")
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        if debug: print(f"[EncoderLayer] after FFN + norm {x.shape}")
        return x, attn_w

class DecoderLayer(nn.Module):
    def __init__(self, d_model:int, heads:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask:Optional[torch.Tensor]=None, memory_mask:Optional[torch.Tensor]=None, debug:bool=False):
        if debug: print(f"[DecoderLayer] input x {x.shape}, enc_out {enc_out.shape}")
        attn1, w1 = self.self_attn(x,x,x, mask=tgt_mask, debug=debug)  # masked self-attn
        x = self.norm1(x + self.dropout(attn1))
        if debug: print(f"[DecoderLayer] after self-attn {x.shape}")
        attn2, w2 = self.cross_attn(x, enc_out, enc_out, mask=memory_mask, debug=debug)  # cross-attn
        x = self.norm2(x + self.dropout(attn2))
        if debug: print(f"[DecoderLayer] after cross-attn {x.shape}")
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        if debug: print(f"[DecoderLayer] after FFN {x.shape}")
        return x, (w1, w2)

# %%
# 6) Encoder and Decoder stacks
class Encoder(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, N:int, heads:int, d_ff:int, max_len:int=512, dropout:float=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask:Optional[torch.Tensor]=None, debug:bool=False):
        x = self.tok_embed(src)
        if debug: print(f"[Encoder] token embeddings {x.shape}")
        x = self.pos_enc(x)
        if debug: print(f"[Encoder] after pos enc {x.shape}")
        attn_weights = []
        for i,layer in enumerate(self.layers):
            x, w = layer(x, src_mask, debug=debug)
            attn_weights.append(w)
            if debug: print(f"[Encoder] layer {i} done")
        x = self.norm(x)
        if debug: print(f"[Encoder] final output {x.shape}")
        return x, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, N:int, heads:int, d_ff:int, max_len:int=512, dropout:float=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, tgt_mask:Optional[torch.Tensor]=None, memory_mask:Optional[torch.Tensor]=None, debug:bool=False):
        x = self.tok_embed(tgt)
        if debug: print(f"[Decoder] token embeddings {x.shape}")
        x = self.pos_enc(x)
        if debug: print(f"[Decoder] after pos enc {x.shape}")
        attn_weights = []
        for i,layer in enumerate(self.layers):
            x, w = layer(x, enc_out, tgt_mask, memory_mask, debug=debug)
            attn_weights.append(w)
            if debug: print(f"[Decoder] layer {i} done")
        x = self.norm(x)
        if debug: print(f"[Decoder] final output {x.shape}")
        return x, attn_weights

# %%
# 7) Full Transformer class
class InteractiveTransformer(nn.Module):
    def __init__(self, src_vocab:int, tgt_vocab:int, d_model:int=128, N:int=2, heads:int=4, d_ff:int=512, max_len:int=512, dropout:float=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, heads, d_ff, max_len, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask:Optional[torch.Tensor]=None, tgt_mask:Optional[torch.Tensor]=None, memory_mask:Optional[torch.Tensor]=None, debug:bool=False):
        # returns logits and a dict of intermediate activations for inspection
        enc_out, enc_attn = self.encoder(src, src_mask, debug=debug)
        dec_out, dec_attn = self.decoder(tgt, enc_out, tgt_mask, memory_mask, debug=debug)
        logits = self.out(dec_out)
        activations = {
            'enc_out': enc_out,
            'enc_attn': enc_attn,  # list per layer
            'dec_out': dec_out,
            'dec_attn': dec_attn,  # list per layer, each is (self_attn, cross_attn)
            'logits': logits
        }
        return logits, activations

# %%
# 8) Mask helpers

def make_pad_mask(seq, pad_idx=0):
    # seq: (B, T)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)


def make_causal_mask(T:int, device=None):
    # returns (1,1,T,T) boolean mask where j <= i
    m = torch.tril(torch.ones((T,T), dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(1)

# %%
# 9) Visualization helpers for attention

def plot_attention(attn:torch.Tensor, title:str="Attention", show=True):
    # attn: (B, heads, T_q, T_k) or (heads, T_q, T_k) or (T_q, T_k)
    if attn.dim() == 4:
        attn = attn[0]  # take batch 0
    if attn.dim() == 3:
        heads = attn.size(0)
        fig, axes = plt.subplots(1, heads, figsize=(3*heads,3))
        if heads == 1:
            axes = [axes]
        for i in range(heads):
            axes[i].imshow(attn[i].cpu().detach().numpy())
            axes[i].set_title(f"{title} head {i}")
        plt.tight_layout()
    elif attn.dim() == 2:
        plt.imshow(attn.cpu().detach().numpy())
        plt.title(title)
    if show:
        plt.show()

# %%
# 10) Small demo: create model, run a single forward pass with debug True

def demo_forward(debug:bool=True, device=None):
    # Small toy vocab and sequence lengths so we can see reasonable-sized tensors
    SRC_VOCAB = 12
    TGT_VOCAB = 12
    BATCH = 2
    SRC_LEN = 6
    TGT_LEN = 5
    DEVICE = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('Device used:', DEVICE)

    model = InteractiveTransformer(SRC_VOCAB, TGT_VOCAB, d_model=32, N=2, heads=4, d_ff=64, max_len=64).to(DEVICE)

    # random example (1..vocab-1 are tokens, 0 reserved for PAD)
    src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN), dtype=torch.long, device=DEVICE)
    tgt = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN), dtype=torch.long, device=DEVICE)

    print('\n--- Input tokens ---')
    print('src:\n', src)
    print('tgt:\n', tgt)

    src_mask = make_pad_mask(src).to(DEVICE)
    tgt_mask = make_causal_mask(TGT_LEN, device=DEVICE)
    memory_mask = make_pad_mask(src).to(DEVICE)

    logits, acts = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask, debug=debug)

    print('\n--- Shapes of outputs ---')
    print('logits', logits.shape)
    print('enc_out', acts['enc_out'].shape)
    print('dec_out', acts['dec_out'].shape)

    # visualize first layer attention for encoder and decoder
    print('\n--- Visualizing encoder layer 0 self-attention (batch 0) ---')
    enc_attn_layer0 = acts['enc_attn'][0]  # (B, heads, T, T)
    plot_attention(enc_attn_layer0, title='Encoder layer0 self-attn')

    print('\n--- Visualizing decoder layer 0 self-attention (batch 0) ---')
    dec_self_attn_layer0 = acts['dec_attn'][0][0]
    plot_attention(dec_self_attn_layer0, title='Decoder layer0 self-attn')

    print('\n--- Visualizing decoder layer 0 cross-attention (batch 0) ---')
    dec_cross_attn_layer0 = acts['dec_attn'][0][1]
    plot_attention(dec_cross_attn_layer0, title='Decoder layer0 cross-attn')

    return model, src, tgt, acts, logits

# %%
# 11) Toy training loop (optional) - small and instrumented

def train_copy_task(epochs:int=100, debug_every:int=10, device=None):
    SRC_VOCAB = 12
    TGT_VOCAB = SRC_VOCAB
    BATCH = 16
    SEQ_LEN = 8
    DEVICE = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = InteractiveTransformer(SRC_VOCAB, TGT_VOCAB, d_model=64, N=2, heads=4, d_ff=128, max_len=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for ep in range(1, epochs+1):
        model.train()
        src = torch.randint(1, SRC_VOCAB, (BATCH, SEQ_LEN), dtype=torch.long, device=DEVICE)
        tgt = src.clone()
        tgt_input = torch.cat([torch.ones((BATCH,1), dtype=torch.long, device=DEVICE), tgt[:,:-1]], dim=1)
        src_mask = make_pad_mask(src).to(DEVICE)
        tgt_mask = make_causal_mask(SEQ_LEN, device=DEVICE)
        memory_mask = make_pad_mask(src).to(DEVICE)

        logits, acts = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask, debug=False)
        loss = criterion(logits.view(-1, TGT_VOCAB), tgt.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % debug_every == 0 or ep == 1:
            print(f"Epoch {ep}/{epochs} loss {loss.item():.4f}")
            # show sample predictions
            pred = logits.argmax(-1)
            print('sample src[0]:', src[0].cpu().numpy())
            print('sample tgt[0]:', tgt[0].cpu().numpy())
            print('sample pred[0]:', pred[0].cpu().numpy())

    return model

# %%
# 12) Instructions for interactive commenting and exploration
# TODO: Add your comments in the cells below or inline as Python comments.
# - Run the demo_forward() cell to inspect shapes and attention plots step-by-step.
# - Put `debug=True` to print intermediate tensors' shapes.
# - Add `print()` statements where you'd like to see specific values.
# - If you want to inspect a particular layer's weights or biases, access them like:
#     model.encoder.layers[0].self_attn.w_q.weight
#   and add `print(model.encoder.layers[0].self_attn.w_q.weight)` in a cell.

# Example: uncomment and run the next line in a separate cell to run the demo
# model, src, tgt, acts, logits = demo_forward(debug=True)

# End of file
