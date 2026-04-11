"""
text_encoder.py — T5-based text encoder for the KSL text-to-motion model
==========================================================================

Only the *encoder* half of T5 is loaded (``T5EncoderModel``).  The decoder is
never needed because we are not generating text — we only need rich, contextual
semantic embeddings to condition the motion transformers defined in ``model.py``.

Typical usage
─────────────
    from text_encoder import KSLTextEncoder

    # Stage 1: freeze T5, train motion transformers only
    encoder = KSLTextEncoder(model_name="t5-base", freeze_base=True)

    # Stage 2: unfreeze the last 2 T5 blocks for joint fine-tuning
    encoder.unfreeze_weights(num_layers=2)

T5 hidden sizes (d_model)
──────────────────────────
    t5-small  : 512
    t5-base   : 768   ← recommended for prototyping
    t5-large  : 1024
    t5-xl     : 2048
    t5-xxl    : 4096
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Config


class KSLTextEncoder(nn.Module):
    """
    Wraps a HuggingFace ``T5EncoderModel`` to produce contextual text embeddings.

    The encoder accepts tokenized text (``input_ids`` + ``attention_mask``) and
    returns the final hidden states, one vector per input token.  These vectors
    are then passed to the ``TextProjection`` MLP inside ``MoMaskWrapper``
    (see ``model.py``) before being fed to the motion transformers.

    Freeze/unfreeze workflow
    ─────────────────────────
    During early training the large T5 backbone is kept frozen to avoid
    over-fitting on the small Kaggle dataset.  Once the motion transformers
    have converged, the last few T5 encoder blocks can be unfrozen for
    domain-specific fine-tuning on sign-language glosses.

        Stage 1 (default):  freeze_base=True  → only motion params update
        Stage 2 (optional): encoder.unfreeze_weights(num_layers=2)
                                               → last 2 T5 blocks + final norm

    Args:
        model_name  : HuggingFace model identifier, e.g. ``'t5-base'``
        freeze_base : freeze all T5 weights immediately after loading

    Attributes:
        hidden_size : int — d_model of the loaded T5 config (e.g. 768 for t5-base).
                      Use this to set ``text_dim`` when constructing
                      ``MoMaskWrapper``.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        freeze_base: bool = True,
    ) -> None:
        super().__init__()

        self.model_name = model_name

        # Load encoder-only variant — no decoder weights are instantiated
        self.t5: T5EncoderModel = T5EncoderModel.from_pretrained(model_name)

        if freeze_base:
            self.freeze_weights()

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def hidden_size(self) -> int:
        """
        Hidden dimension of the T5 encoder (``d_model`` in the T5 config).

        Pass this value as ``text_dim`` to ``MoMaskWrapper`` so the
        ``TextProjection`` MLP is sized correctly.

        Example::

            encoder = KSLTextEncoder("t5-base")
            model   = MoMaskWrapper(text_dim=encoder.hidden_size, ...)
        """
        return self.t5.config.d_model

    @property
    def num_encoder_blocks(self) -> int:
        """Total number of transformer blocks in the T5 encoder stack."""
        return len(self.t5.encoder.block)

    # ── Freeze / unfreeze API ─────────────────────────────────────────────────

    def freeze_weights(self) -> None:
        """
        Freeze **all** T5 encoder parameters.

        After calling this method, no T5 parameter appears in the optimizer's
        parameter group, which means the backbone is used purely as a feature
        extractor and its weights are not updated during the backward pass.
        """
        for param in self.t5.parameters():
            param.requires_grad = False

        n = self._count_params(only_trainable=False)
        print(
            f"[KSLTextEncoder] All {n:,} T5 parameters frozen "
            f"(model: {self.model_name})."
        )

    def unfreeze_weights(self, num_layers: Optional[int] = None) -> None:
        """
        Unfreeze T5 encoder parameters for fine-tuning.

        Unfreeze selectively (last N transformer blocks) rather than all at
        once to avoid catastrophic forgetting on the small Kaggle dataset.

        Args:
            num_layers : number of *trailing* encoder blocks to unfreeze.
                         ``None`` unfreezes every T5 parameter (full fine-tune).

        Raises:
            ValueError : if ``num_layers`` exceeds the encoder block count.

        Examples::

            encoder.unfreeze_weights()             # full fine-tune
            encoder.unfreeze_weights(num_layers=2) # last 2 blocks + final norm
        """
        total = self.num_encoder_blocks

        if num_layers is None:
            # Unfreeze the entire backbone
            for param in self.t5.parameters():
                param.requires_grad = True
            n = self._count_params(only_trainable=True)
            print(
                f"[KSLTextEncoder] All T5 parameters unfrozen "
                f"({n:,} trainable, model: {self.model_name})."
            )
            return

        if num_layers < 1 or num_layers > total:
            raise ValueError(
                f"num_layers must be in [1, {total}] for {self.model_name} "
                f"(which has {total} encoder blocks). "
                f"Pass None to unfreeze all weights."
            )

        # Unfreeze only the last `num_layers` transformer blocks
        for block in self.t5.encoder.block[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze the final layer norm — it sits after the last block
        # and is critical for the quality of output representations
        for param in self.t5.encoder.final_layer_norm.parameters():
            param.requires_grad = True

        n_trainable = self._count_params(only_trainable=True)
        print(
            f"[KSLTextEncoder] Unfrozen last {num_layers}/{total} encoder blocks "
            f"+ final_layer_norm → {n_trainable:,} trainable parameters "
            f"(model: {self.model_name})."
        )

    # ── Memory / compute helpers ──────────────────────────────────────────────

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing inside the T5 encoder.

        Trading compute for memory: activations are recomputed during the
        backward pass instead of being stored, which reduces peak VRAM usage
        by ~40-60% at the cost of a slower backward pass.  Recommended when
        fine-tuning T5-large or larger on a single GPU.

        Note:
            This must be called **after** ``unfreeze_weights()`` — gradient
            checkpointing has no effect (and will raise a warning) when
            ``requires_grad=False`` for all parameters.
        """
        self.t5.gradient_checkpointing_enable()
        print("[KSLTextEncoder] Gradient checkpointing enabled.")

    def count_parameters(self, only_trainable: bool = True) -> int:
        """
        Return the number of (trainable) parameters in the T5 encoder.

        Args:
            only_trainable : if ``True`` (default), count only params with
                             ``requires_grad=True``; otherwise count all params.

        Returns:
            Integer parameter count.

        Example::

            total     = encoder.count_parameters(only_trainable=False)
            trainable = encoder.count_parameters(only_trainable=True)
            print(f"Trainable: {trainable:,} / {total:,}")
        """
        return self._count_params(only_trainable)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract contextual token embeddings from the T5 encoder.

        The ``attention_mask`` is forwarded directly to the T5 attention layers
        so that padding tokens (mask value = 0) do not influence the hidden
        states of real tokens through the self-attention mechanism.

        When all T5 parameters are frozen the forward pass is automatically
        wrapped in ``torch.no_grad()`` to skip building the backward graph,
        saving both memory and compute.

        Args:
            input_ids      : [B, T]  integer token ids from the HuggingFace
                             tokenizer (values in ``[0, vocab_size)``)
            attention_mask : [B, T]  binary mask; ``1`` = real token,
                             ``0`` = padding token

        Returns:
            last_hidden_state : [B, T, hidden_size]
                Per-token contextual embeddings from the final encoder layer.
                Pass this directly to ``TextProjection`` inside
                ``MoMaskWrapper``.

        Note:
            The caller is responsible for passing ``attention_mask`` to the
            downstream motion transformer so that text padding positions are
            excluded from cross-attention.  In our concatenation-based
            architecture (``model.py``) this is handled by building a
            combined padding mask before the bidirectional encoder.
        """
        # Skip gradient tracking entirely when the backbone is frozen —
        # integer input tensors never carry grad, so this is always safe
        _frozen = not any(p.requires_grad for p in self.t5.parameters())
        ctx     = torch.no_grad() if _frozen else contextlib.nullcontext()

        with ctx:
            outputs = self.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        # last_hidden_state: [B, T, hidden_size]
        return outputs.last_hidden_state

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        total     = self._count_params(only_trainable=False)
        trainable = self._count_params(only_trainable=True)
        return (
            f"KSLTextEncoder("
            f"model='{self.model_name}', "
            f"hidden_size={self.hidden_size}, "
            f"encoder_blocks={self.num_encoder_blocks}, "
            f"params={total:,}, "
            f"trainable={trainable:,})"
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _count_params(self, only_trainable: bool) -> int:
        return sum(
            p.numel()
            for p in self.t5.parameters()
            if (not only_trainable or p.requires_grad)
        )


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from transformers import T5TokenizerFast

    MODEL = "t5-small"   # use t5-small for a fast smoke test
    print(f"Running text_encoder smoke test with '{MODEL}' …\n")

    tokenizer = T5TokenizerFast.from_pretrained(MODEL)
    encoder   = KSLTextEncoder(model_name=MODEL, freeze_base=True)
    print(encoder, "\n")

    # ── basic forward pass (frozen) ───────────────────────────────────────
    sentences = [
        "a person waves their right hand",
        "she signs the word hello",   # different length — tokenizer will pad
    ]
    encoding = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"]       # [2, 32]
    attention_mask = encoding["attention_mask"]  # [2, 32]

    with torch.no_grad():
        hidden = encoder(input_ids, attention_mask)

    print(f"input_ids shape      : {tuple(input_ids.shape)}")
    print(f"attention_mask shape : {tuple(attention_mask.shape)}")
    print(f"hidden_states shape  : {tuple(hidden.shape)}")   # (2, 32, 512) for t5-small
    assert hidden.shape == (2, 32, encoder.hidden_size), "Shape mismatch!"
    print(f"hidden_size          : {encoder.hidden_size}")

    # ── verify frozen: no gradients should exist ──────────────────────────
    assert not any(p.requires_grad for p in encoder.t5.parameters()), \
        "Expected all T5 params to be frozen!"
    print(f"\nAll T5 params frozen  : ✓")
    print(f"Trainable params      : {encoder.count_parameters():,}")
    print(f"Total params          : {encoder.count_parameters(only_trainable=False):,}")

    # ── unfreeze last 2 blocks ────────────────────────────────────────────
    print()
    encoder.unfreeze_weights(num_layers=2)
    trainable_after = encoder.count_parameters(only_trainable=True)
    assert trainable_after > 0, "Expected some trainable params after unfreeze!"
    print(f"Trainable after unfreeze(2): {trainable_after:,}")

    # ── re-freeze, then full unfreeze ─────────────────────────────────────
    print()
    encoder.freeze_weights()
    encoder.unfreeze_weights(num_layers=None)
    assert encoder.count_parameters() == encoder.count_parameters(only_trainable=False)
    print(f"Full unfreeze verified : ✓")

    print("\nAll checks passed.")
