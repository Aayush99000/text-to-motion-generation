"""
model.py — MoMask: Generative Masked Motion Modelling for Sign Language Generation
====================================================================================

Architecture overview
─────────────────────
  TextProjection          : MLP that maps T5 embeddings → hidden_dim
  SinusoidalPositionalEnc : Fixed sine/cosine PE added to motion token embeddings
  BaseMotionTransformer   : Bidirectional encoder that predicts RVQ Layer-0 tokens
  ResidualMotionTransformer: Bidirectional encoder that predicts RVQ Layers 1-5
                             conditioned on Layer-0 and text
  MoMaskWrapper           : Top-level module; exposes forward() for training and
                             generate() for iterative masked decoding (MaskGIT)

RVQ layout
──────────
  Layer 0        : base motion     (vocab 0-511, MASK = 512)
  Layers 1 – 5   : fine residuals  (vocab 0-511, MASK = 512)
  VOCAB_SIZE = 512  →  embedding tables have 513 entries (+ MASK token)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Global constants ─────────────────────────────────────────────────────────

VOCAB_SIZE: int = 512          # RVQ codebook size per layer
MASK_TOKEN_ID: int = 512       # [MASK] id; valid token ids are 0 … 511
NUM_RVQ_LAYERS: int = 6        # total layers (0 = base, 1-5 = residuals)
NUM_RESIDUAL_LAYERS: int = 5   # layers 1 through 5


# ─── 1. Text Projection Layer ─────────────────────────────────────────────────

class TextProjection(nn.Module):
    """
    Two-layer MLP that projects T5 (or any encoder) text embeddings into the
    shared hidden dimension used by the motion transformers.

    Architecture:
        Linear(text_dim → hidden_dim) → GELU → Dropout → Linear(hidden_dim → hidden_dim)
        followed by a LayerNorm for training stability.

    Args:
        text_dim   : dimensionality of the incoming text embeddings
                     (768 for T5-base, 1024 for T5-large)
        hidden_dim : target hidden dimension of the motion transformers
        dropout    : dropout probability applied after the activation

    Forward
    ───────
        x      : [batch_size, text_len, text_dim]
        return : [batch_size, text_len, hidden_dim]
    """

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T, text_dim]   raw text embeddings
        Returns:
              : [B, T, hidden_dim]
        """
        return self.norm(self.net(x))


# ─── 2. Sinusoidal Positional Encoding ───────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learnable) sine/cosine positional encoding from Vaswani et al. (2017).

    A pre-computed buffer of shape [1, max_seq_len, hidden_dim] is sliced at
    runtime and added to the input.  Because the buffer is not a parameter,
    it is moved to the correct device automatically with .to(device).

    Args:
        hidden_dim  : model embedding dimension (must be even)
        max_seq_len : maximum sequence length to pre-compute (default 512)
        dropout     : dropout applied after adding the positional signal

    Forward
    ───────
        x      : [batch_size, seq_len, hidden_dim]
        return : [batch_size, seq_len, hidden_dim]  (PE added in-place semantics)
    """

    def __init__(
        self,
        hidden_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pre-compute the encoding table: [max_seq_len, hidden_dim]
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)   # [L, 1]
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_dim)
        )                                                                        # [D/2]
        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  dimensions

        # register as buffer: not a parameter, but part of state_dict
        self.register_buffer("pe", pe.unsqueeze(0))    # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, S, D]
        Returns:
              : [B, S, D]
        """
        # self.pe is [1, max_seq_len, D]; slice to [1, S, D] and broadcast over B
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─── 3. Base Motion Transformer (RVQ Layer 0) ────────────────────────────────

class BaseMotionTransformer(nn.Module):
    """
    Bidirectional Transformer Encoder for the base motion layer (RVQ Layer 0).

    The model is mask-prediction based (BERT/MaskGIT style): some Layer-0 tokens
    are replaced with MASK_TOKEN_ID = 512 during training, and the model must
    reconstruct the original token at every masked position.

    Architecture
    ─────────────
    1. Embed motion tokens (integer ids → dense vectors)   [B, S, D]
    2. Add sinusoidal positional encoding                  [B, S, D]
    3. Prepend projected text embeddings (no PE re-added;
       T5 already encodes position)                        [B, T+S, D]
    4. Bidirectional Transformer Encoder (no causal mask)  [B, T+S, D]
    5. Slice the motion portion                            [B, S, D]
    6. Linear classification head                          [B, S, vocab_size]

    Args:
        hidden_dim  : width of all embedding and attention layers
        vocab_size  : RVQ codebook size, default 512
        num_heads   : number of attention heads (hidden_dim must be divisible)
        num_layers  : number of TransformerEncoderLayer stacks
        ffn_dim     : inner dimension of the position-wise FFN
        max_seq_len : maximum motion sequence length (for PE pre-computation)
        dropout     : dropout probability

    Forward
    ───────
        text_emb      : [B, T, hidden_dim]  projected text embeddings
        motion_tokens : [B, S]              integer ids; masked positions = 512
        ──────────────────────────────────────────────────────────
        returns logits: [B, S, vocab_size]
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        vocab_size: int = VOCAB_SIZE,
        num_heads: int = 8,
        num_layers: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # vocab_size + 1 entries: 0…511 are real tokens, 512 is [MASK]
        self.token_emb = nn.Embedding(vocab_size + 1, hidden_dim)
        self.pos_enc   = SinusoidalPositionalEncoding(hidden_dim, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # tensors are [B, S, D] not [S, B, D]
            norm_first=True,    # pre-LN (more stable than post-LN for deep models)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm     = nn.LayerNorm(hidden_dim)
        self.head     = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        text_emb: torch.Tensor,
        motion_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_emb      : [B, T, hidden_dim]
            motion_tokens : [B, S]   int64, values in {0…511, 512=MASK}

        Returns:
            logits        : [B, S, vocab_size]
        """
        B, T, _ = text_emb.shape
        # S = motion_tokens.size(1)

        # ── embed motion tokens and encode position ────────────────────────
        m_emb = self.token_emb(motion_tokens)   # [B, S, D]
        m_emb = self.pos_enc(m_emb)             # [B, S, D]

        # ── build full context: [text_tokens | motion_tokens] ────────────
        seq = torch.cat([text_emb, m_emb], dim=1)  # [B, T+S, D]

        # ── bidirectional attention (no src_mask → full attention) ────────
        seq = self.encoder(seq)   # [B, T+S, D]
        seq = self.norm(seq)

        # ── extract only the motion positions ─────────────────────────────
        motion_out = seq[:, T:, :]        # [B, S, D]
        logits     = self.head(motion_out) # [B, S, vocab_size]
        return logits


# ─── 4. Residual Motion Transformer (RVQ Layers 1-5) ─────────────────────────

class ResidualMotionTransformer(nn.Module):
    """
    Bidirectional Transformer Encoder that predicts all five residual RVQ layers
    (Layers 1–5) in a **single** forward pass, conditioned on the decoded Layer-0
    tokens and the text description.

    Key design choices
    ──────────────────
    • All five residual token sequences are concatenated along the sequence
      dimension after adding a learnable *layer-index* embedding to each, so the
      model can distinguish which RVQ level it is attending to at every position.
    • The Layer-0 tokens are embedded (with positional encoding + layer-index 0)
      and prepended as context (not masked).
    • The full input sequence has layout:
          [text (T) | layer0 (S) | res_layer1 (S) | … | res_layer5 (S)]
      total length = T + 6·S

    Architecture
    ─────────────
    1. Embed Layer-0 tokens + PE + layer-index emb[0]          [B, S, D]
    2. For each residual layer i in {1…5}:
         Embed tokens + PE + layer-index emb[i]                [B, S, D]
    3. Concatenate: [text | layer0 | res1 | … | res5]          [B, T+6S, D]
    4. Bidirectional Transformer Encoder                        [B, T+6S, D]
    5. Slice residual portions                        5 × [B, S, D]
    6. Shared linear head → stack                     [B, 5, S, vocab_size]

    Args:
        hidden_dim           : model width
        vocab_size           : RVQ codebook size, default 512
        num_residual_layers  : number of residual RVQ levels (default 5)
        num_heads            : attention heads
        num_layers           : encoder depth
        ffn_dim              : FFN inner dimension
        max_seq_len          : maximum motion sequence length
        dropout              : dropout probability

    Forward
    ───────
        text_emb        : [B, T, hidden_dim]
        layer0_tokens   : [B, S]                no masking; predicted or ground-truth
        residual_tokens : [B, num_res_layers, S] masked during training
        ─────────────────────────────────────────────────────────────────────
        returns logits  : [B, num_res_layers, S, vocab_size]
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        vocab_size: int = VOCAB_SIZE,
        num_residual_layers: int = NUM_RESIDUAL_LAYERS,
        num_heads: int = 8,
        num_layers: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_residual_layers = num_residual_layers

        # shared embedding table: real tokens (0-511) + MASK (512)
        self.token_emb     = nn.Embedding(vocab_size + 1, hidden_dim)
        self.pos_enc       = SinusoidalPositionalEncoding(hidden_dim, max_seq_len, dropout)

        # learnable layer-index embeddings:
        #   index 0   → Layer-0 (base)
        #   index 1-5 → residual layers 1-5
        self.layer_idx_emb = nn.Embedding(num_residual_layers + 1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm     = nn.LayerNorm(hidden_dim)

        # single shared head: each position predicts one token from the same codebook
        self.head = nn.Linear(hidden_dim, vocab_size)

    def _embed_segment(
        self,
        token_ids: torch.Tensor,   # [B, S]
        layer_index: int,
    ) -> torch.Tensor:
        """
        Shared helper: embed a token sequence, add positional encoding and
        a learnable layer-index embedding.

        Args:
            token_ids   : [B, S]
            layer_index : which RVQ level (0 = base, 1-5 = residuals)

        Returns:
            emb : [B, S, hidden_dim]
        """
        emb = self.token_emb(token_ids)           # [B, S, D]
        emb = self.pos_enc(emb)                   # [B, S, D]

        idx = torch.tensor(
            [layer_index], dtype=torch.long, device=token_ids.device
        )
        emb = emb + self.layer_idx_emb(idx)       # broadcast [1, 1, D] over [B, S, D]
        return emb

    def forward(
        self,
        text_emb: torch.Tensor,
        layer0_tokens: torch.Tensor,
        residual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_emb        : [B, T, D]
            layer0_tokens   : [B, S]
            residual_tokens : [B, num_res, S]

        Returns:
            logits          : [B, num_res, S, vocab_size]
        """
        B, T, _ = text_emb.shape
        S = layer0_tokens.size(1)

        # ── embed Layer-0 as conditioning context ─────────────────────────
        l0_emb = self._embed_segment(layer0_tokens, layer_index=0)  # [B, S, D]

        # ── embed each residual layer's token sequence ────────────────────
        res_embs = [
            self._embed_segment(residual_tokens[:, i, :], layer_index=i + 1)
            for i in range(self.num_residual_layers)
        ]  # list of num_res × [B, S, D]

        # ── build full sequence ───────────────────────────────────────────
        # layout: [text (T) | layer0 (S) | res_1 (S) | … | res_5 (S)]
        seq = torch.cat([text_emb, l0_emb] + res_embs, dim=1)  # [B, T+6S, D]

        # ── bidirectional transformer ──────────────────────────────────────
        seq = self.encoder(seq)   # [B, T+6S, D]
        seq = self.norm(seq)

        # ── slice residual segments and project to logits ─────────────────
        # text occupies [0, T), layer0 occupies [T, T+S),
        # residual i occupies [T + S + i*S, T + S + (i+1)*S)
        res_logits: list[torch.Tensor] = []
        for i in range(self.num_residual_layers):
            start = T + S + i * S
            end   = start + S
            r_out = seq[:, start:end, :]        # [B, S, D]
            res_logits.append(self.head(r_out)) # [B, S, vocab_size]

        return torch.stack(res_logits, dim=1)   # [B, num_res, S, vocab_size]


# ─── 5. MoMask Wrapper ────────────────────────────────────────────────────────

class MoMaskWrapper(nn.Module):
    """
    Top-level MoMask model.

    Wires together TextProjection, BaseMotionTransformer, and
    ResidualMotionTransformer.  Exposes two entry points:

    forward()   — training mode: returns logits given pre-masked token sequences.
    generate()  — inference mode: iterative MaskGIT decoding of Layer 0, followed
                  by a single-pass prediction of Layers 1-5.

    Args:
        text_dim             : dimension of incoming text embeddings (T5-base = 768)
        hidden_dim           : shared hidden dimension for all sub-models
        vocab_size           : RVQ codebook entries per layer (default 512)
        num_residual_layers  : number of residual RVQ levels (default 5)
        base_num_heads       : attention heads for the base transformer
        base_num_layers      : encoder depth for the base transformer
        res_num_heads        : attention heads for the residual transformer
        res_num_layers       : encoder depth for the residual transformer
        ffn_dim              : FFN inner dimension (shared by both transformers)
        max_seq_len          : maximum motion sequence length
        dropout              : dropout probability
    """

    MASK_ID: int = MASK_TOKEN_ID  # 512

    def __init__(
        self,
        text_dim: int = 768,
        hidden_dim: int = 512,
        vocab_size: int = VOCAB_SIZE,
        num_residual_layers: int = NUM_RESIDUAL_LAYERS,
        base_num_heads: int = 8,
        base_num_layers: int = 8,
        res_num_heads: int = 8,
        res_num_layers: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vocab_size          = vocab_size
        self.num_residual_layers = num_residual_layers

        self.text_proj = TextProjection(text_dim, hidden_dim, dropout)

        self.base_transformer = BaseMotionTransformer(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_heads=base_num_heads,
            num_layers=base_num_layers,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.residual_transformer = ResidualMotionTransformer(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_residual_layers=num_residual_layers,
            num_heads=res_num_heads,
            num_layers=res_num_layers,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        text_emb: torch.Tensor,
        masked_layer0: torch.Tensor,
        masked_residuals: torch.Tensor,
        gt_layer0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        For Layer 0, we run the base transformer on the masked token sequence.
        For Layers 1-5, we condition the residual transformer on either the
        ground-truth Layer-0 tokens (teacher forcing, recommended) or the
        base model's own argmax predictions.

        Args:
            text_emb          : [B, T, text_dim]
                                Raw T5 embeddings; projected internally.
            masked_layer0     : [B, S]
                                Layer-0 token ids with a subset replaced by MASK_ID.
            masked_residuals  : [B, num_res, S]
                                Residual token ids (Layers 1-5) with a subset masked.
            gt_layer0         : [B, S] or None
                                Ground-truth Layer-0 ids used to condition the residual
                                transformer.  When None the base model's argmax is used
                                (straight-through, no gradient to base from residual loss).

        Returns:
            base_logits       : [B, S, vocab_size]           Layer-0 classification scores
            residual_logits   : [B, num_res, S, vocab_size]  Layers 1-5 scores
        """
        # ── 1. project text embeddings into hidden space ───────────────────
        text_proj = self.text_proj(text_emb)                # [B, T, D]

        # ── 2. predict Layer-0 ────────────────────────────────────────────
        base_logits = self.base_transformer(
            text_proj, masked_layer0
        )                                                   # [B, S, vocab_size]

        # ── 3. choose Layer-0 conditioning for the residual model ─────────
        if gt_layer0 is not None:
            # teacher forcing: use ground truth to avoid error propagation
            cond_layer0 = gt_layer0                         # [B, S]
        else:
            # use model's own argmax (detached; no gradient flows back)
            cond_layer0 = base_logits.detach().argmax(dim=-1)  # [B, S]

        # ── 4. predict residual Layers 1-5 ────────────────────────────────
        residual_logits = self.residual_transformer(
            text_proj, cond_layer0, masked_residuals
        )                                                   # [B, num_res, S, vocab_size]

        return base_logits, residual_logits

    # ── Inference: iterative masked decoding ──────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        text_emb: torch.Tensor,
        seq_len: int,
        num_iter: int = 10,
        temperature: float = 1.0,
        mask_schedule: str = "cosine",
    ) -> torch.Tensor:
        """
        Generate a full motion sequence from text embeddings.

        Stage 1 — Base layer (MaskGIT iterative decoding):
            • Initialise all S positions as [MASK].
            • Repeat for `num_iter` steps:
                1. Run the base transformer → logits [B, S, V].
                2. Sample a candidate token at every position.
                3. Compute per-position confidence (max softmax probability).
                4. Determine how many tokens must remain masked according to
                   the schedule; keep the LEAST confident positions masked.
                5. Unmask the remaining positions with their sampled tokens.
                   Already-decoded positions are never re-masked.
            • After the loop, greedily fill any lingering [MASK] positions.

        Stage 2 — Residual layers (single-pass prediction):
            • Initialise all residual positions as [MASK].
            • Run the residual transformer once → logits [B, num_res, S, V].
            • Greedily decode (argmax) for all positions.

        Args:
            text_emb      : [B, T, text_dim]   T5 embeddings (no grad required)
            seq_len       : number of motion tokens to generate (S)
            num_iter      : MaskGIT decoding iterations for Layer 0 (default 10)
            temperature   : softmax temperature; 0 or near-0 → greedy (default 1.0)
            mask_schedule : schedule for the masking ratio; "cosine" or "linear"

        Returns:
            all_tokens    : [B, NUM_RVQ_LAYERS, S]  integer token ids
                            index 0 → Layer 0, indices 1-5 → Layers 1-5
        """
        B      = text_emb.size(0)
        device = text_emb.device

        # project text once and reuse for every decoding step
        text_proj = self.text_proj(text_emb)   # [B, T, D]

        # ── Stage 1: iterative masked decoding of Layer 0 ─────────────────

        # start fully masked
        layer0_tokens = torch.full(
            (B, seq_len), self.MASK_ID, dtype=torch.long, device=device
        )                                                    # [B, S]

        for step in range(num_iter):
            # ── predict ───────────────────────────────────────────────────
            logits = self.base_transformer(
                text_proj, layer0_tokens
            )                                               # [B, S, vocab_size]

            # clamp temperature to avoid division by zero
            temp = max(temperature, 1e-8)
            probs = F.softmax(logits / temp, dim=-1)        # [B, S, vocab_size]

            # sample a candidate token for every position
            if temperature < 1e-8:
                # fully greedy
                sampled = logits.argmax(dim=-1)             # [B, S]
            else:
                sampled = torch.multinomial(
                    probs.view(B * seq_len, self.vocab_size),
                    num_samples=1,
                ).view(B, seq_len)                          # [B, S]

            # confidence = max probability (peak sharpness of the distribution)
            confidence = probs.max(dim=-1).values           # [B, S]

            # positions that are already decoded should never be re-masked;
            # give them infinite confidence so they sort to the end
            is_masked  = layer0_tokens.eq(self.MASK_ID)     # [B, S] bool
            confidence = confidence.masked_fill(~is_masked, float("inf"))

            # how many tokens should remain masked after this step
            ratio      = _mask_ratio(step + 1, num_iter, mask_schedule)
            num_masked = int(ratio * seq_len)               # target masked count

            # sort ascending: indices of least-confident positions come first
            _, sorted_idx = confidence.sort(dim=-1)         # [B, S]

            # unmask all masked positions (fill with sampled token)
            layer0_tokens = torch.where(is_masked, sampled, layer0_tokens)

            # re-mask the `num_masked` least-confident positions
            # (sorted_idx[:, :num_masked] are all originally-masked tokens
            #  because already-decoded tokens have confidence = +inf and
            #  appear at the tail of ascending sort)
            if num_masked > 0:
                mask_positions = sorted_idx[:, :num_masked]  # [B, num_masked]
                layer0_tokens.scatter_(1, mask_positions, self.MASK_ID)

        # safety net: greedily decode any positions that are still [MASK]
        still_masked = layer0_tokens.eq(self.MASK_ID)       # [B, S]
        if still_masked.any():
            final_logits  = self.base_transformer(text_proj, layer0_tokens)
            greedy_tokens = final_logits.argmax(dim=-1)     # [B, S]
            layer0_tokens = torch.where(still_masked, greedy_tokens, layer0_tokens)

        # ── Stage 2: single-pass prediction of residual Layers 1-5 ────────

        # start fully masked
        residual_tokens = torch.full(
            (B, self.num_residual_layers, seq_len),
            self.MASK_ID,
            dtype=torch.long,
            device=device,
        )                                                    # [B, num_res, S]

        res_logits = self.residual_transformer(
            text_proj, layer0_tokens, residual_tokens
        )                                                    # [B, num_res, S, vocab_size]

        # greedy decode all residual positions
        res_tokens = res_logits.argmax(dim=-1)               # [B, num_res, S]

        # ── assemble all 6 layers ──────────────────────────────────────────
        all_tokens = torch.cat(
            [layer0_tokens.unsqueeze(1), res_tokens], dim=1
        )                                                    # [B, NUM_RVQ_LAYERS, S]

        return all_tokens


# ─── Utility ──────────────────────────────────────────────────────────────────

def _mask_ratio(step: int, total_steps: int, schedule: str = "cosine") -> float:
    """
    Compute the fraction of tokens that should remain MASKED after `step` iterations.

    Values decrease from ~1.0 (fully masked) to 0.0 (fully decoded).

    Args:
        step        : current decoding step (1-indexed)
        total_steps : total number of decoding steps
        schedule    : "cosine" (smooth, recommended) or "linear"

    Returns:
        ratio in [0.0, 1.0]

    Examples:
        >>> [round(_mask_ratio(t, 10, "cosine"), 3) for t in range(1, 11)]
        [0.988, 0.951, 0.891, 0.809, 0.707, 0.588, 0.454, 0.309, 0.156, 0.0]
    """
    r = step / total_steps
    if schedule == "cosine":
        return math.cos(r * math.pi / 2)
    elif schedule == "linear":
        return 1.0 - r
    else:
        raise ValueError(f"Unknown schedule '{schedule}'. Use 'cosine' or 'linear'.")


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal forward + generate smoke test (CPU, tiny dims for speed)
    B, T, S = 2, 16, 32   # batch=2, text_len=16, motion_len=32

    model = MoMaskWrapper(
        text_dim=768,
        hidden_dim=128,
        vocab_size=512,
        num_residual_layers=5,
        base_num_heads=4,
        base_num_layers=2,
        res_num_heads=4,
        res_num_layers=2,
        ffn_dim=256,
        max_seq_len=64,
        dropout=0.0,   # disable dropout for deterministic test
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── forward (training) ────────────────────────────────────────────────
    text_emb         = torch.randn(B, T, 768)
    masked_layer0    = torch.randint(0, 512, (B, S))
    # mask ~15% of positions
    mask_idx         = torch.rand(B, S) < 0.15
    masked_layer0[mask_idx] = MASK_TOKEN_ID

    masked_residuals = torch.randint(0, 512, (B, 5, S))
    mask_idx_r       = torch.rand(B, 5, S) < 0.15
    masked_residuals[mask_idx_r] = MASK_TOKEN_ID

    gt_layer0 = torch.randint(0, 512, (B, S))

    base_logits, res_logits = model(
        text_emb, masked_layer0, masked_residuals, gt_layer0=gt_layer0
    )
    print(f"base_logits     : {tuple(base_logits.shape)}")   # (2, 32, 512)
    print(f"residual_logits : {tuple(res_logits.shape)}")    # (2, 5, 32, 512)

    # ── generate (inference) ──────────────────────────────────────────────
    tokens = model.generate(text_emb, seq_len=S, num_iter=5, temperature=1.0)
    print(f"generated tokens: {tuple(tokens.shape)}")        # (2, 6, 32)
    print("All checks passed.")
