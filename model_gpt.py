"""
model_gpt.py — T2M-GPT: Autoregressive Text-to-Motion Generation
=================================================================

Architecture
─────────────
  A GPT-style causal transformer decoder conditioned on T5 text embeddings
  via cross-attention at every layer.

  Motion tokens are flattened frame-by-frame across all 6 RVQ layers:
    [BOS, l0_t0, l1_t0, l2_t0, l3_t0, l4_t0, l5_t0,
          l0_t1, l1_t1, ..., l5_t1,
          ...
          l0_tL, l1_tL, ..., l5_tL, EOS]

  This lets the model learn temporal structure naturally via causal attention,
  and decide sequence length itself by predicting EOS.

Special tokens
──────────────
  0–511  : RVQ motion tokens
  512    : BOS (beginning of sequence)
  513    : EOS (end of sequence)
  514    : PAD (padding, ignored in loss)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Vocabulary ────────────────────────────────────────────────────────────────
VOCAB_SIZE     : int = 512   # RVQ token ids  0-511
BOS_ID         : int = 512
EOS_ID         : int = 513
PAD_ID         : int = 514
TOTAL_VOCAB    : int = 515   # 512 motion + BOS + EOS + PAD
NUM_RVQ_LAYERS : int = 6


# ── Attention blocks ──────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal (lower-triangular) mask."""

    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        causal = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale                      # [B, H, T, T]
        attn = attn.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from motion, keys/values from text."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv  = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        S = context.size(1)

        q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(context).split(C, dim=-1)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class GPTBlock(nn.Module):
    """Transformer block: causal self-attn → cross-attn → FFN (pre-norm)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.self_attn  = CausalSelfAttention(hidden_dim, num_heads, max_seq_len, dropout)
        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), context)
        x = x + self.ffn(self.ln3(x))
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class T2MGPT(nn.Module):
    """
    Text-to-Motion GPT.

    Args:
        text_dim    : dimension of T5 text embeddings (768 for t5-base)
        hidden_dim  : transformer hidden size
        num_heads   : attention heads
        num_layers  : number of GPT blocks
        ffn_dim     : feed-forward expansion dim
        max_seq_len : max motion token sequence length (max_frames * 6 + 2)
        dropout     : dropout probability
    """

    def __init__(
        self,
        text_dim   : int   = 768,
        hidden_dim : int   = 512,
        num_heads  : int   = 8,
        num_layers : int   = 8,
        ffn_dim    : int   = 2048,
        max_seq_len: int   = 770,   # 128 frames × 6 + BOS + EOS
        dropout    : float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(TOTAL_VOCAB, hidden_dim)
        self.pos_emb   = nn.Embedding(max_seq_len, hidden_dim)
        self.drop      = nn.Dropout(dropout)

        # Project T5 → hidden_dim
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(hidden_dim, num_heads, ffn_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, TOTAL_VOCAB, bias=False)

        # Weight tying: output projection shares token embedding weights
        self.head.weight = self.token_emb.weight

        self._init_weights()
        print(
            f"T2MGPT | params: {sum(p.numel() for p in self.parameters()):,} | "
            f"layers: {num_layers} | heads: {num_heads} | dim: {hidden_dim} | "
            f"max_seq: {max_seq_len}"
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (teacher-forced training).

        Args:
            tokens   : [B, T]      input token ids  (BOS + motion tokens)
            text_emb : [B, S, D]   T5 text embeddings (raw, before projection)

        Returns:
            logits   : [B, T, TOTAL_VOCAB]
        """
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)  # [1, T]

        x = self.drop(self.token_emb(tokens) + self.pos_emb(pos))

        context = self.text_proj(text_emb)  # [B, S, hidden_dim]

        for block in self.blocks:
            x = block(x, context)

        return self.head(self.ln_f(x))      # [B, T, TOTAL_VOCAB]

    @torch.inference_mode()
    def generate(
        self,
        text_emb    : torch.Tensor,
        max_new_tokens: int   = 768,
        temperature : float   = 1.0,
        top_k       : int     = 256,
    ) -> list[int]:
        """
        Autoregressively generate motion tokens for a single sample.

        Args:
            text_emb      : [1, S, text_dim]  T5 text embeddings
            max_new_tokens : max tokens to generate (max_frames * 6)
            temperature   : sampling temperature
            top_k         : top-k logit filter

        Returns:
            List of integer motion token ids (BOS/EOS stripped)
        """
        device  = text_emb.device
        context = self.text_proj(text_emb)  # [1, S, hidden_dim]

        tokens = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            T = tokens.size(1)
            if T >= self.max_seq_len:
                break

            pos = torch.arange(T, device=device).unsqueeze(0)
            x   = self.drop(self.token_emb(tokens) + self.pos_emb(pos))

            for block in self.blocks:
                x = block(x, context)

            logits = self.head(self.ln_f(x))[:, -1, :]   # [1, TOTAL_VOCAB]

            # Mask BOS and PAD — never generate them
            logits[:, BOS_ID] = float("-inf")
            logits[:, PAD_ID] = float("-inf")

            # Top-k filter (only over motion token range)
            if top_k > 0:
                k = min(top_k, VOCAB_SIZE)
                motion_logits = logits[:, :VOCAB_SIZE]
                topk_vals, _ = motion_logits.topk(k, dim=-1)
                threshold = topk_vals[:, -1:]
                logits[:, :VOCAB_SIZE] = logits[:, :VOCAB_SIZE].masked_fill(
                    logits[:, :VOCAB_SIZE] < threshold, float("-inf")
                )

            if temperature != 1.0 and temperature > 1e-8:
                logits = logits / temperature
            elif temperature <= 1e-8:
                next_token = logits.argmax(dim=-1, keepdim=True)
                if next_token.item() == EOS_ID:
                    break
                tokens = torch.cat([tokens, next_token], dim=1)
                continue

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == EOS_ID:
                break

            tokens = torch.cat([tokens, next_token], dim=1)

        # Return motion tokens only (strip BOS)
        return tokens[0, 1:].cpu().tolist()
