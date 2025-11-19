"""
Positional Encoding and BPE Tokenization — Interactive Demo (Gradio)

This standalone script mirrors the notebook functionality. It provides:
- Training a tiny Byte-Pair Encoding (BPE) tokenizer on the input sentence (demo-only)
- Tokenization details: token count and token→id mapping
- Random token embeddings E (seq_len, d_model)
- Sinusoidal positional encodings PE (seq_len, d_model)
- Summed inputs X = sqrt(d_model) * E + PE as in "Attention Is All You Need"
- A downloadable vocabulary JSON and a PDF report summarizing inputs and results

Run:
  python "Positional-Encoding.py"

Then open the local Gradio link shown in the console.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

# --- Optional, on-the-fly dependency installation to keep the script self-contained ---
def _ensure_pkg(pkg: str) -> None:
    try:
        __import__(pkg)
    except ImportError:  # pragma: no cover
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


for _p in ("numpy", "gradio", "tokenizers", "reportlab"):
    _ensure_pkg(_p)

import numpy as np
import gradio as gr
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


# Reproducibility for the demo
np.random.seed(42)


def train_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = 2000,
    special_tokens: List[str] = ("[PAD]", "[UNK]", "[BOS]", "[EOS]"),
) -> Tokenizer:
    """Train a minimal BPE tokenizer on the provided texts (demo-only).

    Parameters
    ----------
    texts: list of str
        Training texts for BPE merges/vocab.
    vocab_size: int
        Target vocabulary size including special tokens.
    special_tokens: list of str
        Reserved tokens placed at the start of the vocab.
    """
    model = BPE(unk_token="[UNK]")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=list(special_tokens))
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def build_embeddings_and_lookup(
    vocab_size: int, d_model: int, token_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a random embedding matrix and look up embeddings for token_ids.

    Returns (embedding_matrix, token_embeddings)
      - embedding_matrix shape: (vocab_size, d_model)
      - token_embeddings shape: (len(token_ids), d_model)
    """
    embedding_matrix = np.random.normal(
        loc=0.0, scale=0.02, size=(vocab_size, int(d_model))
    ).astype(np.float32)
    token_ids_array = np.array(token_ids, dtype=np.int64)
    token_embeddings = embedding_matrix[token_ids_array]
    return embedding_matrix, token_embeddings


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encodings from Vaswani et al. (2017).

      PE[pos, 2i]   = sin(pos / (10000^(2i/d_model)))
      PE[pos, 2i+1] = cos(pos / (10000^(2i/d_model)))
    """
    positions = np.arange(int(seq_len), dtype=np.float32)[:, np.newaxis]
    i = np.arange(0, int(d_model), 2, dtype=np.float32)[np.newaxis, :]
    denom = np.power(10000.0, i / float(d_model))
    angle_rates = positions / denom
    pe = np.zeros((int(seq_len), int(d_model)), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle_rates)
    pe[:, 1::2] = np.cos(angle_rates)
    return pe


def create_pdf_report(
    sentence: str,
    d_model: int,
    tokens: List[str],
    token_ids: List[int],
    token_embeddings: np.ndarray,
    positional_enc: np.ndarray,
    summed_matrix: np.ndarray,
    out_path: str,
    preview_rows: int = 6,
    preview_cols: int = 8,
) -> str:
    """Create a compact PDF report summarizing inputs and results."""
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter
    current_y = height - 1 * inch

    def writeln(text: str, x: float = 1 * inch, y: float | None = None, leading: int = 14):
        nonlocal current_y
        if y is not None:
            current_y = y
        max_chars = 95
        lines = [text[i : i + max_chars] for i in range(0, len(text), max_chars)] if len(text) > max_chars else [text]
        for line in lines:
            c.drawString(x, current_y, line)
            current_y -= leading

    def write_matrix_preview(title: str, mat: np.ndarray):
        nonlocal current_y
        writeln(title)
        r = min(preview_rows, mat.shape[0])
        k = min(preview_cols, mat.shape[1])
        header = f"cols 0..{k-1} (rounded to 4 dp)"
        writeln(header)
        for ri in range(r):
            row_vals = ", ".join(f"{v:.4f}" for v in mat[ri, :k])
            writeln(f"row {ri}: [ {row_vals} ]")
            if current_y < 1 * inch:
                c.showPage()
                current_y = height - 1 * inch

    writeln("Positional Encoding & BPE — Report", y=current_y)
    writeln("Reference: Vaswani et al., 2017 — 'Attention Is All You Need'")
    writeln("")
    writeln(f"Input sentence: {sentence}")
    writeln(f"d_model: {int(d_model)}")
    writeln(f"Token count: {len(token_ids)}")
    writeln(f"Tokens: {tokens}")
    writeln(f"Token IDs: {token_ids}")
    writeln("")
    writeln("Shapes:")
    writeln(f" - token_embeddings: {token_embeddings.shape}")
    writeln(f" - positional_enc:   {positional_enc.shape}")
    writeln(f" - summed_matrix:     {summed_matrix.shape}")
    writeln("")
    writeln("We compute: X = sqrt(d_model) * E + PE")
    writeln("")
    write_matrix_preview("Preview: token embeddings (E)", token_embeddings)
    write_matrix_preview("Preview: positional encodings (PE)", positional_enc)
    write_matrix_preview("Preview: X = sqrt(d_model) * E + PE", summed_matrix)
    c.showPage()
    c.save()
    return out_path


def process_sentence(sentence: str, d_model: int = 64):
    """Full pipeline used by the Gradio UI.

    Returns in this order:
      - embeddings_df
      - posenc_df
      - summed_df
      - token_count
      - token_map_df
      - vocab_file_path
      - pdf_file_path
    """
    if not sentence or not sentence.strip():
        sentence = "Transformers are powerful sequence models!"

    tokenizer = train_bpe_tokenizer([sentence], vocab_size=256)
    encoding = tokenizer.encode(sentence)
    token_ids = encoding.ids
    tokens = encoding.tokens
    token_count = len(token_ids)
    token_map_df = [[tok, int(tid)] for tok, tid in zip(tokens, token_ids)]

    vocab_size = len(tokenizer.get_vocab())
    _, token_embeddings = build_embeddings_and_lookup(
        vocab_size=vocab_size, d_model=int(d_model), token_ids=token_ids
    )

    embeddings_df = token_embeddings.astype(float).tolist()

    # Save vocab JSON beside this script
    out_dir = os.path.abspath(os.path.dirname(__file__))
    vocab_file_path = os.path.join(out_dir, "demo_vocab.json")
    import json

    vocab_dict: Dict[str, int] = tokenizer.get_vocab()
    vocab_items = sorted(vocab_dict.items(), key=lambda kv: kv[1])
    with open(vocab_file_path, "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in vocab_items}, f, ensure_ascii=False, indent=2)

    # Positional encodings and summed inputs
    pe = positional_encoding(seq_len=token_count, d_model=int(d_model))
    emb_scaled = token_embeddings * np.sqrt(float(d_model))
    summed = emb_scaled + pe

    posenc_df = pe.astype(float).tolist()
    summed_df = summed.astype(float).tolist()

    # PDF report beside this script
    pdf_file_path = os.path.join(out_dir, "positional_encoding_report.pdf")
    try:
        create_pdf_report(
            sentence=sentence,
            d_model=int(d_model),
            tokens=tokens,
            token_ids=token_ids,
            token_embeddings=token_embeddings,
            positional_enc=pe,
            summed_matrix=summed,
            out_path=pdf_file_path,
        )
    except Exception as e:  # pragma: no cover
        # If PDF creation fails, write a small .txt explaining the error
        pdf_file_path = os.path.join(out_dir, "positional_encoding_report_failed.txt")
        with open(pdf_file_path, "w", encoding="utf-8") as f:
            f.write(f"PDF generation failed: {e}\n")
            f.write("Proceeding without the PDF preview.\n")

    return (
        embeddings_df,
        posenc_df,
        summed_df,
        int(token_count),
        token_map_df,
        vocab_file_path,
        pdf_file_path,
    )


# Build the Gradio interface
inp_sentence = gr.Textbox(label="Enter a sentence", placeholder="Type any sentence to tokenize with BPE...")
inp_dmodel = gr.Slider(minimum=16, maximum=512, step=16, value=64, label="Embedding dimension (d_model)")

out_embeddings = gr.Dataframe(label="Initial Token Embeddings E (rows=tokens, cols=d_model)")
out_posenc = gr.Dataframe(label="Positional Encodings PE (rows=positions, cols=d_model)")
out_summed = gr.Dataframe(label="X = sqrt(d_model) * E + PE (rows=tokens, cols=d_model)")
out_token_count = gr.Number(label="Number of BPE tokens")
out_token_map = gr.Dataframe(headers=["token", "id"], label="Token→ID mapping (BPE tokenization)")
out_vocab_file = gr.File(label="Saved vocabulary file (JSON)")
out_pdf = gr.File(label="Download Report (PDF)")

demo = gr.Interface(
    fn=process_sentence,
    inputs=[inp_sentence, inp_dmodel],
    outputs=[
        out_embeddings,
        out_posenc,
        out_summed,
        out_token_count,
        out_token_map,
        out_vocab_file,
        out_pdf,
    ],
    title="BPE Tokenization, Embeddings, Positional Encodings, and Summation",
    description=(
        "Enter a sentence to see BPE tokenization, initial token embeddings (E), sinusoidal positional encodings (PE),\n"
        "their sum X = sqrt(d_model) * E + PE as in 'Attention Is All You Need', a saved vocab, and a downloadable PDF report."
    ),
)


if __name__ == "__main__":
    # Launch locally; set share=True if you need a public link
    demo.launch(share=False)
