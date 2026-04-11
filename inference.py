"""
inference.py — MoMask iterative decoding and Kaggle submission generation
==========================================================================

Pipeline
─────────
  1. Load trained KSLTextEncoder + MoMaskWrapper from checkpoint.
  2. Load a LengthEstimator to predict motion sequence length from text.
  3. For each row in test.csv:
       a. Encode text with T5 → mean-pool → predict sequence length L.
       b. Iteratively decode Layer-0 tokens via MaskGIT (cosine schedule,
          temperature + top-k sampling).
       c. Predict Layers 1-5 residuals in a single forward pass conditioned
          on the decoded Layer-0 tokens.
  4. Format the L×6 token matrix and write submission.csv.

MaskGIT decoding (Layer 0)
───────────────────────────
  Start: all S positions filled with [MASK] token (id = 512).

  Each iteration t ∈ {1 … N}:
    ① Predict logits at every position                  [B, S, V]
    ② Apply top-k filter + temperature → sample tokens  [B, S]
    ③ Compute confidence = max(softmax probability)     [B, S]
    ④ Lock already-decoded positions (confidence → ∞)
    ⑤ Cosine schedule: ratio = cos(t/N · π/2)
       → num_masked = ⌊ratio · S⌋ tokens remain masked
    ⑥ Unmask the most-confident positions; re-mask the rest.

  After N iterations, greedily fill any residual [MASK] positions.

Run
───
  python inference.py \\
    --checkpoint checkpoints/checkpoint_epoch_100.pth \\
    --test_csv   /scratch/katoch.aa/text-to-motion/test.csv \\
    --output     submission.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import T5TokenizerFast

from model import MASK_TOKEN_ID, VOCAB_SIZE, MoMaskWrapper
from text_encoder import KSLTextEncoder


# ─── Constants ────────────────────────────────────────────────────────────────

NUM_RVQ_LAYERS: int = 6
DEFAULT_NUM_ITER: int = 12     # MaskGIT decoding steps for Layer 0
DEFAULT_TEMPERATURE: float = 1.0
DEFAULT_TOP_K: int = 256       # restrict sampling to top-256 logits
MIN_SEQ_LEN: int = 4
MAX_SEQ_LEN: int = 256


# ─── Length Estimator ─────────────────────────────────────────────────────────

class LengthEstimator(nn.Module):
    """
    Lightweight MLP that predicts motion sequence length from mean-pooled
    T5 text embeddings.

    This model is trained separately (regressing against the actual sequence
    lengths in train.csv) and loaded from ``length_estimator.pth`` at inference.

    Architecture:
        Linear(text_dim → 512) → LayerNorm → GELU → Dropout →
        Linear(512 → 256)      → LayerNorm → GELU → Dropout →
        Linear(256 → 1)

    Args:
        text_dim : dimension of the input text embeddings (e.g. 768 for t5-base)
        dropout  : dropout probability in hidden layers

    Forward
    ───────
        x      : [B, text_dim]   mean-pooled text embeddings
        return : [B]             predicted sequence lengths (float, pre-rounding)
    """

    def __init__(self, text_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, text_dim]
        Returns:
              : [B]  raw (non-integer) length predictions
        """
        return self.net(x).squeeze(-1)   # [B]

    @torch.inference_mode()
    def predict(
        self,
        x: torch.Tensor,
        min_len: int = MIN_SEQ_LEN,
        max_len: int = MAX_SEQ_LEN,
    ) -> torch.Tensor:
        """
        Predict and clamp integer sequence lengths.

        Args:
            x       : [B, text_dim]
            min_len : minimum allowed output length
            max_len : maximum allowed output length

        Returns:
            lengths : [B]  int64 predicted lengths in [min_len, max_len]
        """
        raw = self.forward(x)                                # [B]
        return raw.round().long().clamp(min=min_len, max=max_len)


# ─── Checkpoint loading ───────────────────────────────────────────────────────

def load_inference_components(
    checkpoint_path: str,
    length_estimator_path: str,
    device: torch.device,
) -> Tuple[KSLTextEncoder, MoMaskWrapper, LengthEstimator, T5TokenizerFast]:
    """
    Load all components required for inference from saved checkpoints.

    Args:
        checkpoint_path       : path to the training checkpoint produced by
                                ``train.py``  (contains both encoder + model states)
        length_estimator_path : path to a separately trained ``LengthEstimator``
                                state dict (``.pth``)
        device                : target device

    Returns:
        text_encoder     : KSLTextEncoder in eval mode
        model            : MoMaskWrapper in eval mode
        length_estimator : LengthEstimator in eval mode
        tokenizer        : T5TokenizerFast matching the checkpoint's T5 model
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("config", {})

    t5_model_name = cfg.get("t5_model", "t5-base")
    tokenizer     = T5TokenizerFast.from_pretrained(t5_model_name)

    # ── text encoder ──────────────────────────────────────────────────────
    text_encoder = KSLTextEncoder(model_name=t5_model_name, freeze_base=True)
    text_encoder.load_state_dict(ckpt["text_encoder_state"])
    text_encoder.to(device).eval()

    # ── motion model ──────────────────────────────────────────────────────
    model = MoMaskWrapper(
        text_dim        = text_encoder.hidden_size,
        hidden_dim      = cfg.get("hidden_dim",      512),
        base_num_heads  = cfg.get("base_num_heads",  8),
        base_num_layers = cfg.get("base_num_layers", 8),
        res_num_heads   = cfg.get("res_num_heads",   8),
        res_num_layers  = cfg.get("res_num_layers",  8),
        ffn_dim         = cfg.get("ffn_dim",         2048),
        max_seq_len     = cfg.get("max_seq_len",     256),
        dropout         = 0.0,   # always 0 at inference
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # ── length estimator ──────────────────────────────────────────────────
    le_state = torch.load(length_estimator_path, map_location=device)
    text_dim = text_encoder.hidden_size
    length_estimator = LengthEstimator(text_dim=text_dim)

    # handle both bare state-dict and wrapped dict formats
    if isinstance(le_state, dict) and "state_dict" in le_state:
        le_state = le_state["state_dict"]
    length_estimator.load_state_dict(le_state)
    length_estimator.to(device).eval()

    print(f"[inference] Loaded checkpoint  : {checkpoint_path}")
    print(f"[inference] T5 model           : {t5_model_name}")
    print(f"[inference] Length estimator   : {length_estimator_path}")
    print(f"[inference] Device             : {device}\n")

    return text_encoder, model, length_estimator, tokenizer


# ─── Sampling helpers ─────────────────────────────────────────────────────────

def apply_topk_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Filter logits to the top-k entries and apply temperature scaling.

    Applying top-k *before* softmax prevents the model from sampling very
    unlikely tokens that would produce jerky, unnatural motions.  Temperature
    controls the sharpness of the distribution:
      - temperature < 1.0  →  sharper (more confident, less diverse)
      - temperature > 1.0  →  flatter  (more diverse, less confident)
      - temperature → 0    →  greedy argmax

    Args:
        logits      : [..., vocab_size]  raw logits (any leading batch dims)
        temperature : softmax temperature
        top_k       : keep only the top-k logit values; 0 = no filtering

    Returns:
        filtered logits [..., vocab_size] ready for softmax / multinomial
    """
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        # threshold = k-th largest value per distribution
        threshold = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if temperature != 1.0 and temperature > 1e-8:
        logits = logits / temperature

    return logits


def mean_pool_text(
    text_emb: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute attention-mask-weighted mean pooling over the sequence dimension.

    Padding positions (mask = 0) are excluded from the average so they do not
    dilute the signal from real tokens.

    Args:
        text_emb       : [B, T, D]
        attention_mask : [B, T]   1 = real token, 0 = padding

    Returns:
        pooled : [B, D]
    """
    mask   = attention_mask.unsqueeze(-1).float()             # [B, T, 1]
    summed = (text_emb * mask).sum(dim=1)                     # [B, D]
    count  = mask.sum(dim=1).clamp(min=1e-8)                  # [B, 1]
    return summed / count                                      # [B, D]


# ─── MaskGIT iterative decoding ───────────────────────────────────────────────

def cosine_mask_schedule(step: int, total_steps: int) -> float:
    """
    Return the fraction of positions that should remain MASKED after ``step``
    decoding iterations.

    The cosine schedule starts near 1.0 (almost all tokens masked) and smoothly
    decays to 0.0 (no tokens masked) by the final step.

    Math:
        ratio(t) = cos(t / N · π/2)

    Args:
        step        : current iteration (1-indexed)
        total_steps : total number of decoding iterations

    Returns:
        mask_ratio ∈ [0.0, 1.0]

    Examples:
        >>> [f"{cosine_mask_schedule(t, 12):.3f}" for t in range(1, 13)]
        ['0.991', '0.966', '0.924', '0.866', '0.793', '0.707',
         '0.609', '0.500', '0.383', '0.259', '0.131', '0.000']
    """
    return math.cos((step / total_steps) * (math.pi / 2))


@torch.inference_mode()
def iterative_decode_layer0(
    text_emb:    torch.Tensor,
    model:       MoMaskWrapper,
    seq_len:     int,
    num_iter:    int   = DEFAULT_NUM_ITER,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k:       int   = DEFAULT_TOP_K,
) -> torch.Tensor:
    """
    Iteratively decode Layer-0 (base motion) tokens using the MaskGIT algorithm.

    The algorithm frames token generation as iterative masked prediction.
    Starting from a fully-masked sequence, it progressively replaces [MASK]
    tokens with predicted tokens, always keeping the most confident predictions
    and re-masking the least confident ones for the next round.

    Detailed step-by-step per iteration ``t`` ∈ {1 … N}
    ──────────────────────────────────────────────────────
    ① Forward pass
          model.base_transformer(text_emb, layer0_tokens)
          → logits  : [B, S, 512]

    ② Top-k filter + temperature → sample candidate tokens
          filtered  = apply_topk_temperature(logits, temperature, top_k)
          probs     = softmax(filtered)           [B, S, 512]
          sampled   = multinomial(probs)          [B, S]   integer token ids

    ③ Confidence score
          confidence = probs.max(dim=-1)          [B, S]
          Using the *maximum* probability (rather than the sampled token's
          probability) gives a deterministic, temperature-independent signal
          of how certain the model is at each position.

    ④ Lock already-decoded positions
          confidence[~is_masked] = +∞
          Positions decoded in prior iterations are never re-masked.

    ⑤ Cosine mask schedule
          ratio      = cos(t/N · π/2)             scalar ∈ [0, 1]
          num_masked = ⌊ratio · S⌋                how many positions stay masked

    ⑥ Update the sequence
          • Replace ALL currently-masked positions with their sampled tokens.
          • Then re-mask the ``num_masked`` positions with the LOWEST confidence
            (sorted ascending → head of sorted list = least confident → re-masked).
          • Already-decoded positions (confidence = +∞) always land at the
            tail of ascending sort and are never re-masked.

    After N iterations, any lingering [MASK] is filled greedily (argmax).

    Args:
        text_emb    : [B, T, hidden_dim]  projected text embeddings
        model       : MoMaskWrapper (in eval mode)
        seq_len     : length S of the motion sequence to generate
        num_iter    : number of MaskGIT decoding steps (default 12)
        temperature : sampling temperature (default 1.0)
        top_k       : top-k logit filtering; 0 = disabled (default 256)

    Returns:
        layer0_tokens : [B, S]  finalised Layer-0 integer token ids (0–511)

    Shapes at each sub-step (B = batch, S = seq_len, V = 512)::

        layer0_tokens  [B, S]     int64  — the evolving token sequence
        logits         [B, S, V]  float  — raw model output
        probs          [B, S, V]  float  — softmax probabilities
        sampled        [B, S]     int64  — candidate tokens from sampling
        confidence     [B, S]     float  — max prob per position
        is_masked      [B, S]     bool   — True where token == MASK_TOKEN_ID
        sorted_idx     [B, S]     int64  — positions sorted by confidence ↑
        mask_positions [B, M]     int64  — M = num_masked positions to keep masked
    """
    B      = text_emb.size(0)
    device = text_emb.device

    # ── Initialise: all positions are [MASK] ──────────────────────────────
    layer0_tokens = torch.full(
        (B, seq_len), MASK_TOKEN_ID, dtype=torch.long, device=device
    )                                                         # [B, S]

    for step in range(1, num_iter + 1):

        # ── ① forward pass ────────────────────────────────────────────────
        logits = model.base_transformer(
            text_emb, layer0_tokens
        )                                                     # [B, S, V]

        # ── ② top-k + temperature → sample ───────────────────────────────
        filtered = apply_topk_temperature(logits, temperature, top_k)
        probs    = F.softmax(filtered, dim=-1)                # [B, S, V]

        if temperature < 1e-8:
            sampled = logits.argmax(dim=-1)                   # [B, S] greedy
        else:
            sampled = torch.multinomial(
                probs.view(B * seq_len, VOCAB_SIZE), num_samples=1
            ).view(B, seq_len)                                # [B, S]

        # ── ③ confidence = max probability (temperature-independent) ──────
        confidence = probs.max(dim=-1).values                 # [B, S]

        # ── ④ lock already-decoded positions ─────────────────────────────
        is_masked  = layer0_tokens.eq(MASK_TOKEN_ID)          # [B, S] bool
        confidence = confidence.masked_fill(~is_masked, float("inf"))

        # ── ⑤ cosine schedule → how many positions stay masked ────────────
        ratio      = cosine_mask_schedule(step, num_iter)
        num_masked = int(ratio * seq_len)                     # M

        # ── ⑥ update sequence ─────────────────────────────────────────────
        # Decode all currently-masked positions with sampled tokens …
        layer0_tokens = torch.where(is_masked, sampled, layer0_tokens)

        # … then re-mask the M least-confident positions.
        # sorted ascending → indices[:, :M] = least confident (still-masked ones,
        # since already-decoded positions have confidence = +∞ and sink to the tail).
        if num_masked > 0:
            _, sorted_idx  = confidence.sort(dim=-1)          # [B, S] ascending
            mask_positions = sorted_idx[:, :num_masked]       # [B, M]
            layer0_tokens.scatter_(1, mask_positions, MASK_TOKEN_ID)

    # ── Safety pass: greedily fill any residual [MASK] positions ──────────
    # (can occur when sequence length < num_iter due to tiny sequences)
    still_masked = layer0_tokens.eq(MASK_TOKEN_ID)            # [B, S] bool
    if still_masked.any():
        final_logits  = model.base_transformer(text_emb, layer0_tokens)
        layer0_tokens = torch.where(
            still_masked, final_logits.argmax(dim=-1), layer0_tokens
        )

    return layer0_tokens                                       # [B, S]


@torch.inference_mode()
def decode_residuals(
    text_emb:      torch.Tensor,
    layer0_tokens: torch.Tensor,
    model:         MoMaskWrapper,
) -> torch.Tensor:
    """
    Predict residual layers 1-5 in a single forward pass.

    The residual transformer is conditioned on the fully-decoded Layer-0
    tokens, so prediction is non-iterative — one shot is sufficient because
    the residuals are low-entropy refinements of a known base.

    Args:
        text_emb      : [B, T, D]
        layer0_tokens : [B, S]      finalised Layer-0 token ids
        model         : MoMaskWrapper

    Returns:
        residuals : [B, 5, S]  integer token ids for Layers 1-5 (greedy)

    Shapes::

        residual_input  [B, 5, S]   all MASK — model predicts from scratch
        res_logits      [B, 5, S, V]
        residuals       [B, 5, S]   argmax over V
    """
    B, S   = layer0_tokens.shape
    device = layer0_tokens.device

    # Feed all-MASK residuals; the model conditions purely on layer0 + text
    residual_input = torch.full(
        (B, model.num_residual_layers, S),
        MASK_TOKEN_ID, dtype=torch.long, device=device,
    )                                                          # [B, 5, S]

    res_logits = model.residual_transformer(
        text_emb, layer0_tokens, residual_input
    )                                                          # [B, 5, S, V]

    return res_logits.argmax(dim=-1)                           # [B, 5, S]


@torch.inference_mode()
def generate_motion(
    text_encoder:     KSLTextEncoder,
    model:            MoMaskWrapper,
    length_estimator: LengthEstimator,
    input_ids:        torch.Tensor,
    attention_mask:   torch.Tensor,
    num_iter:         int   = DEFAULT_NUM_ITER,
    temperature:      float = DEFAULT_TEMPERATURE,
    top_k:            int   = DEFAULT_TOP_K,
) -> torch.Tensor:
    """
    Full generation pipeline: text → motion tokens.

    Steps:
        A. Encode text with T5 → mean-pool → predict sequence length L.
        B. Iteratively decode Layer-0 tokens (MaskGIT, L steps).
        C. Predict Layers 1-5 residuals in one pass conditioned on Layer 0.

    Args:
        text_encoder     : KSLTextEncoder
        model            : MoMaskWrapper
        length_estimator : LengthEstimator
        input_ids        : [B, T]  tokenized text
        attention_mask   : [B, T]  padding mask
        num_iter         : MaskGIT decoding iterations for Layer 0
        temperature      : sampling temperature
        top_k            : top-k logit filter (0 = off)

    Returns:
        all_tokens : [B, 6, L_max]  integer token ids for all 6 RVQ layers.
                     Layer 0 at index 0, residuals at indices 1-5.
                     Each sample may have a different valid length; the
                     caller should trim using the returned lengths.
        lengths    : [B]  predicted integer sequence lengths per sample.
    """
    B      = input_ids.size(0)
    device = input_ids.device

    # ── A. Text encoding + length prediction ──────────────────────────────
    text_emb = text_encoder(input_ids, attention_mask)        # [B, T, D]
    pooled   = mean_pool_text(text_emb, attention_mask)       # [B, D]
    lengths  = length_estimator.predict(pooled)               # [B]  int64

    # Pad all sequences in this batch to the maximum predicted length.
    # Individual samples are trimmed to their own lengths by the caller.
    max_len  = int(lengths.max().item())

    # ── B. Project text once, reuse for all decoding steps ────────────────
    text_proj = model.text_proj(text_emb)                     # [B, T, hidden_dim]

    # ── B. Layer-0 iterative decoding ─────────────────────────────────────
    layer0 = iterative_decode_layer0(
        text_emb    = text_proj,
        model       = model,
        seq_len     = max_len,
        num_iter    = num_iter,
        temperature = temperature,
        top_k       = top_k,
    )                                                          # [B, max_len]

    # ── C. Single-pass residual prediction ────────────────────────────────
    residuals = decode_residuals(text_proj, layer0, model)    # [B, 5, max_len]

    # ── Assemble: [layer0 | res1 | … | res5] along dim 1 ─────────────────
    all_tokens = torch.cat(
        [layer0.unsqueeze(1), residuals], dim=1
    )                                                          # [B, 6, max_len]

    return all_tokens, lengths


# ─── Submission formatting ────────────────────────────────────────────────────

def tokens_to_submission_string(tokens: torch.Tensor) -> str:
    """
    Flatten a single sample's token matrix to a space-separated string.

    Layout (row-major by frame):
        frame_0_layer0, frame_0_layer1, …, frame_0_layer5,
        frame_1_layer0, …
        frame_L_layer5

    Args:
        tokens : [6, L]  integer token ids for one sample

    Returns:
        A space-separated string of 6·L integers.
    """
    # tokens: [6, L] → transpose → [L, 6] → flatten → [6L]
    flat = tokens.T.reshape(-1).cpu().numpy()
    return " ".join(map(str, flat.tolist()))


# ─── Main inference loop ──────────────────────────────────────────────────────

def run_inference(
    text_encoder:     KSLTextEncoder,
    model:            MoMaskWrapper,
    length_estimator: LengthEstimator,
    tokenizer:        T5TokenizerFast,
    test_csv:         str,
    output_csv:       str,
    batch_size:       int   = 16,
    max_text_len:     int   = 64,
    use_gloss:        bool  = False,
    num_iter:         int   = DEFAULT_NUM_ITER,
    temperature:      float = DEFAULT_TEMPERATURE,
    top_k:            int   = DEFAULT_TOP_K,
    device:           torch.device = torch.device("cpu"),
) -> None:
    """
    Iterate through test.csv, generate motion tokens for every row, and
    save the formatted predictions to ``output_csv``.

    Args:
        text_encoder     : KSLTextEncoder (eval mode)
        model            : MoMaskWrapper  (eval mode)
        length_estimator : LengthEstimator (eval mode)
        tokenizer        : T5 tokenizer
        test_csv         : path to test.csv  (columns: id, sentence, gloss)
        output_csv       : path to write submission.csv
        batch_size       : number of samples per GPU batch
        max_text_len     : max tokenized text length (must match training)
        use_gloss        : use 'gloss' column instead of 'sentence'
        num_iter         : MaskGIT iterations for Layer-0 decoding
        temperature      : sampling temperature
        top_k            : top-k logit filtering
        device           : inference device
    """
    test_df     = pd.read_csv(test_csv)
    text_column = "gloss" if use_gloss else "sentence"
    n_samples   = len(test_df)

    print(f"[inference] Test samples : {n_samples}")
    print(f"[inference] Text column  : '{text_column}'")
    print(f"[inference] Batch size   : {batch_size}")
    print(f"[inference] MaskGIT iter : {num_iter}")
    print(f"[inference] Temperature  : {temperature}")
    print(f"[inference] Top-k        : {top_k}\n")

    ids:     List[str] = []
    preds:   List[str] = []

    # process in batches, iterate with tqdm progress bar
    with tqdm(total=n_samples, desc="Generating", unit="sample") as pbar:
        for start in range(0, n_samples, batch_size):
            batch_rows = test_df.iloc[start : start + batch_size]
            texts      = batch_rows[text_column].fillna("").tolist()
            batch_ids  = batch_rows["id"].astype(str).tolist()

            # ── tokenise ──────────────────────────────────────────────────
            enc = tokenizer(
                texts,
                max_length   = max_text_len,
                padding      = "max_length",
                truncation   = True,
                return_tensors = "pt",
            )
            input_ids      = enc["input_ids"].to(device)       # [B, T]
            attention_mask = enc["attention_mask"].to(device)  # [B, T]

            # ── generate ──────────────────────────────────────────────────
            all_tokens, lengths = generate_motion(
                text_encoder     = text_encoder,
                model            = model,
                length_estimator = length_estimator,
                input_ids        = input_ids,
                attention_mask   = attention_mask,
                num_iter         = num_iter,
                temperature      = temperature,
                top_k            = top_k,
            )
            # all_tokens: [B, 6, max_len]
            # lengths:    [B]

            # ── format each sample ────────────────────────────────────────
            for i, (sid, L) in enumerate(zip(batch_ids, lengths.tolist())):
                # trim to the actual predicted length for this sample
                sample_tokens = all_tokens[i, :, :L]          # [6, L]
                pred_str      = tokens_to_submission_string(sample_tokens)
                ids.append(sid)
                preds.append(pred_str)

            pbar.update(len(batch_rows))

    # ── write submission.csv ──────────────────────────────────────────────
    submission = pd.DataFrame({"id": ids, "motion_tokens": preds})
    submission.to_csv(output_csv, index=False)
    print(f"\n[inference] Submission saved → {output_csv}  ({len(submission)} rows)")
    print(f"[inference] Preview:\n{submission.head(3).to_string()}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MoMask inference — generate Kaggle submission from test.csv"
    )

    # ── required ──────────────────────────────────────────────────────────
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to training checkpoint (.pth) produced by train.py",
    )
    p.add_argument(
        "--length_estimator",
        default="length_estimator.pth",
        help="Path to length_estimator.pth (default: %(default)s)",
    )
    p.add_argument(
        "--test_csv", required=True,
        help="Path to test.csv",
    )

    # ── output ────────────────────────────────────────────────────────────
    p.add_argument(
        "--output", default="submission.csv",
        help="Output submission CSV path (default: %(default)s)",
    )

    # ── text ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--use_gloss", action="store_true",
        help="Condition on gloss column instead of sentence",
    )
    p.add_argument(
        "--max_text_len", type=int, default=64,
        help="Max tokenised text length (must match training, default: %(default)s)",
    )

    # ── decoding ──────────────────────────────────────────────────────────
    p.add_argument(
        "--num_iter", type=int, default=DEFAULT_NUM_ITER,
        help=f"MaskGIT iterations for Layer-0 (default: %(default)s)",
    )
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="Sampling temperature; 0 = greedy (default: %(default)s)",
    )
    p.add_argument(
        "--top_k", type=int, default=DEFAULT_TOP_K,
        help="Top-k logit filter; 0 = disabled (default: %(default)s)",
    )

    # ── batching ──────────────────────────────────────────────────────────
    p.add_argument(
        "--batch_size", type=int, default=16,
        help="Inference batch size (default: %(default)s)",
    )

    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    device = get_device()

    print(f"[inference] Device: {device}\n")

    # ── load all components ───────────────────────────────────────────────
    text_encoder, model, length_estimator, tokenizer = load_inference_components(
        checkpoint_path       = args.checkpoint,
        length_estimator_path = args.length_estimator,
        device                = device,
    )

    # ── run inference and save submission ─────────────────────────────────
    run_inference(
        text_encoder     = text_encoder,
        model            = model,
        length_estimator = length_estimator,
        tokenizer        = tokenizer,
        test_csv         = args.test_csv,
        output_csv       = args.output,
        batch_size       = args.batch_size,
        max_text_len     = args.max_text_len,
        use_gloss        = args.use_gloss,
        num_iter         = args.num_iter,
        temperature      = args.temperature,
        top_k            = args.top_k,
        device           = device,
    )


if __name__ == "__main__":
    main()
