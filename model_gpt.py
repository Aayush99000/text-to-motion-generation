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

    @torch.inference_mode()
    def generate_batch(
        self,
        text_emb      : torch.Tensor,
        max_new_tokens: int   = 768,
        temperature   : float = 1.0,
        top_k         : int   = 256,
        guidance_scale: float = 1.0,
    ) -> list[list[int]]:
        """
        Autoregressively generate motion tokens for a batch using KV-cache.

        Each step processes only the ONE new token (O(T) per step) instead of
        recomputing the full growing sequence (O(T²) per step), reducing total
        complexity from O(T³) → O(T²).

        Args:
            text_emb       : [B, S, text_dim]  T5 text embeddings
            max_new_tokens : max tokens to generate (max_frames * 6)
            temperature    : sampling temperature
            top_k          : top-k logit filter

        Returns:
            List of B token lists (BOS/EOS stripped)
        """
        B        = text_emb.size(0)
        device   = text_emb.device
        C        = self.hidden_dim
        H        = self.blocks[0].self_attn.num_heads
        head_dim = C // H

        # ── CFG: double the batch with null (zero) embeddings ────────────────
        # Layout: [cond_0..cond_B-1, uncond_0..uncond_B-1]
        use_cfg = guidance_scale > 1.0
        if use_cfg:
            null_emb  = torch.zeros_like(text_emb)
            text_emb  = torch.cat([text_emb, null_emb], dim=0)  # [2B, S, D]
        BS = text_emb.size(0)  # B or 2B

        # ── Pre-compute cross-attention KV from text (never changes) ──────────
        context = self.text_proj(text_emb)  # [BS, S, C]
        S       = context.size(1)
        cross_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            k, v = block.cross_attn.kv(context).split(C, dim=-1)
            k = k.view(BS, S, H, head_dim).transpose(1, 2)   # [BS, H, S, head_dim]
            v = v.view(BS, S, H, head_dim).transpose(1, 2)
            cross_kv.append((k, v))

        # ── Self-attention KV cache (grows by 1 at every step) ───────────────
        empty = torch.zeros(BS, H, 0, head_dim, device=device)
        self_kv: list[tuple[torch.Tensor, torch.Tensor]] = [
            (empty.clone(), empty.clone()) for _ in self.blocks
        ]

        done      = torch.zeros(B, dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in range(B)]
        results:   list[list[int] | None] = [None] * B

        # Feed BOS at position 0, then up to max_new_tokens tokens
        # For CFG: same tokens fed to both cond and uncond paths
        cur_input = torch.full((B,), BOS_ID, dtype=torch.long, device=device)

        for pos_idx in range(max_new_tokens + 1):  # pos 0 = BOS
            if pos_idx >= self.max_seq_len:
                break

            # Embedding for the single new token
            # For CFG: replicate cur_input for both cond and uncond paths
            pos       = torch.tensor([pos_idx], device=device)
            inp       = torch.cat([cur_input, cur_input], dim=0) if use_cfg else cur_input
            x   = self.drop(
                self.token_emb(inp).unsqueeze(1) +          # [BS, 1, C]
                self.pos_emb(pos).unsqueeze(0)              # [1,  1, C]
            )   # [BS, 1, C]

            new_self_kv: list[tuple[torch.Tensor, torch.Tensor]] = []

            for i, block in enumerate(self.blocks):
                # ── Self-attention (KV-cached) ─────────────────────────────
                x_norm = block.ln1(x)
                q_new, k_new, v_new = block.self_attn.qkv(x_norm).split(C, dim=-1)
                q_new = q_new.view(BS, 1, H, head_dim).transpose(1, 2)  # [BS,H,1,d]
                k_new = k_new.view(BS, 1, H, head_dim).transpose(1, 2)
                v_new = v_new.view(BS, 1, H, head_dim).transpose(1, 2)

                k_past, v_past = self_kv[i]
                k_full = torch.cat([k_past, k_new], dim=2)  # [B,H,T,d]
                v_full = torch.cat([v_past, v_new], dim=2)
                new_self_kv.append((k_full, v_full))

                # q is always the LAST position → no causal mask needed
                scale = block.self_attn.scale
                attn  = F.softmax((q_new @ k_full.transpose(-2, -1)) * scale, dim=-1)
                sa_out = (attn @ v_full).transpose(1, 2).contiguous().view(BS, 1, C)
                x = x + block.self_attn.resid_drop(block.self_attn.proj(sa_out))

                # ── Cross-attention (pre-cached text KV) ───────────────────
                x_norm2 = block.ln2(x)
                q_c     = block.cross_attn.q(x_norm2).view(BS, 1, H, head_dim).transpose(1, 2)
                k_ctx, v_ctx = cross_kv[i]
                attn_c  = F.softmax(
                    (q_c @ k_ctx.transpose(-2, -1)) * block.cross_attn.scale, dim=-1
                )
                ca_out  = (attn_c @ v_ctx).transpose(1, 2).contiguous().view(BS, 1, C)
                x = x + block.cross_attn.resid_drop(block.cross_attn.proj(ca_out))

                # ── FFN ────────────────────────────────────────────────────
                x = x + block.ffn(block.ln3(x))

            self_kv = new_self_kv

            # Logits for next token
            logits_all = self.head(self.ln_f(x))[:, 0, :]   # [BS, TOTAL_VOCAB]

            # ── CFG blending ──────────────────────────────────────────────
            if use_cfg:
                logits_cond   = logits_all[:B]   # [B, TOTAL_VOCAB]
                logits_uncond = logits_all[B:]   # [B, TOTAL_VOCAB]
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_all              # [B, TOTAL_VOCAB]

            if pos_idx == max_new_tokens:
                break   # reached limit; results saved below

            # Mask BOS and PAD
            logits[:, BOS_ID] = float("-inf")
            logits[:, PAD_ID] = float("-inf")

            # Top-k filter over motion token range
            if top_k > 0:
                k_filter = min(top_k, VOCAB_SIZE)
                topk_vals, _ = logits[:, :VOCAB_SIZE].topk(k_filter, dim=-1)
                threshold    = topk_vals[:, -1:]
                logits[:, :VOCAB_SIZE] = logits[:, :VOCAB_SIZE].masked_fill(
                    logits[:, :VOCAB_SIZE] < threshold, float("-inf")
                )

            # Sample
            if temperature > 1e-8:
                probs       = F.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
            else:
                next_tokens = logits.argmax(dim=-1)   # [B]

            is_eos = next_tokens == EOS_ID

            for i in range(B):
                if is_eos[i] and not done[i]:
                    results[i] = list(generated[i])
                    done[i]    = True
                elif not done[i]:
                    generated[i].append(next_tokens[i].item())

            if done.all():
                break

            next_tokens[done] = PAD_ID   # finished samples emit PAD (discarded)
            cur_input = next_tokens

        # Sequences that never emitted EOS
        for i in range(B):
            if results[i] is None:
                results[i] = list(generated[i])

        return results  # type: ignore[return-value]
