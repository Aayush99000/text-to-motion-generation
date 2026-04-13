"""
train.py — Training loop for the KSL text-to-motion MoMask model
==================================================================

Ties together:
  text_encoder.py  →  KSLTextEncoder   (T5 backbone, frozen or partially fine-tuned)
  model.py         →  MoMaskWrapper    (BaseMotionTransformer + ResidualMotionTransformer)
  dataset.py       →  KSLMotionDataset / MotionCollator / build_dataloader

Training strategy (MoMask, two-stage masked prediction)
─────────────────────────────────────────────────────────
  Stage 1 — Base model (Layer 0):
    • Sample a per-sample masking ratio r ~ Uniform(mask_ratio_low, mask_ratio_high).
    • Randomly replace r% of the ground-truth Layer-0 tokens with [MASK] (id = 512).
    • The base transformer sees [text | masked_layer0] and predicts the original tokens
      at MASKED positions only.  Unmasked and padded positions are excluded from loss
      via ignore_index=512 in CrossEntropyLoss.

  Stage 2 — Residual model (Layers 1-5):
    • The FULL ground-truth Layer-0 sequence is provided as context (teacher forcing).
      This avoids error propagation from base-model mistakes during early training.
    • A separate per-sample masking ratio is sampled; the SAME time positions are
      masked across all five residual layers so the model attends to coherent frames.
    • The residual transformer predicts the original tokens at masked positions only.

  Total loss = 0.5 * base_loss + 0.5 * residual_loss
  (CrossEntropyLoss, ignore_index=512 for both heads)

AMP & optimiser
────────────────
  Automatic Mixed Precision (float16) is used on CUDA via torch.autocast +
  GradScaler.  On MPS / CPU, AMP is silently disabled.
  Optimiser: AdamW with weight decay.
  Scheduler: linear warmup → CosineAnnealingLR.

Run
───
  python train.py                       # default config
  python train.py --csv data/train.csv  # override paths via CLI flags
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import T5TokenizerFast

from dataset import MOTION_PAD_ID, NUM_RVQ_LAYERS, build_dataloader
from model import MASK_TOKEN_ID, VOCAB_SIZE, MoMaskWrapper
from text_encoder import KSLTextEncoder


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """
    Central configuration object.  Every hyper-parameter lives here so the
    training script is fully reproducible from a single config snapshot.
    """

    # ── Paths ─────────────────────────────────────────────────────────────
    csv_path:    str = "data/train.csv"
    ckpt_dir:    str = "checkpoints"
    resume_from: Optional[str] = None   # path to a .pth checkpoint to resume

    # ── Text encoder ──────────────────────────────────────────────────────
    t5_model:    str  = "t5-base"       # HuggingFace model id
    freeze_t5:   bool = True            # freeze T5 for Stage-1 training
    max_text_len: int = 64
    use_gloss:   bool = False           # use 'sentence' col if False

    # ── Motion model ──────────────────────────────────────────────────────
    hidden_dim:      int   = 512
    base_num_heads:  int   = 8
    base_num_layers: int   = 8
    res_num_heads:   int   = 8
    res_num_layers:  int   = 8
    ffn_dim:         int   = 2048
    max_seq_len:     int   = 512
    dropout:         float = 0.1

    # ── Masking ───────────────────────────────────────────────────────────
    mask_ratio_low:  float = 0.1        # minimum fraction of tokens masked per sample
    mask_ratio_high: float = 1.0        # maximum fraction of tokens masked per sample

    # ── Optimiser ─────────────────────────────────────────────────────────
    lr:              float = 1e-4
    weight_decay:    float = 0.01
    grad_clip:       float = 1.0        # max gradient norm (0 to disable)
    betas:           Tuple = (0.9, 0.98)

    # ── Scheduler ─────────────────────────────────────────────────────────
    num_epochs:    int = 100
    warmup_epochs: int = 5              # linear warmup before cosine decay
    lr_min:        float = 1e-6         # cosine floor

    # ── DataLoader ────────────────────────────────────────────────────────
    batch_size:  int = 32
    num_workers: int = 4
    pin_memory:  bool = True

    # ── Misc ──────────────────────────────────────────────────────────────
    seed:       int  = 42
    use_amp:    bool = True             # AMP; auto-disabled on non-CUDA
    log_every:  int  = 10              # log every N batches within an epoch
    save_every: int  = 1               # save checkpoint every N epochs


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Masking helpers ──────────────────────────────────────────────────────────

def mask_layer0(
    layer0_gt: torch.Tensor,
    padding_mask: torch.Tensor,
    mask_id: int = MASK_TOKEN_ID,
    ratio_low: float = 0.1,
    ratio_high: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random per-sample masking to the base RVQ layer (Layer 0).

    For each sample ``b`` in the batch a masking ratio ``r_b`` is drawn
    uniformly from ``[ratio_low, ratio_high]``.  Each non-padded position is
    then independently masked with probability ``r_b``.

    Detailed tensor transformations
    ────────────────────────────────
    Inputs
        layer0_gt    : [B, S]        Ground-truth Layer-0 token ids (0–511).
                                     Padded positions already hold ``mask_id``
                                     (set by MotionCollator with MOTION_PAD_ID=512).
        padding_mask : [B, S] bool   True at positions that are batch-padding
                                     (set by MotionCollator).

    Internal
        ratios       : [B]           Per-sample masking ratios ~ Uniform(low, high).
        rand_mat     : [B, S]        Independent uniform draws per position.
        token_mask   : [B, S] bool   True where we want to replace with [MASK].
                                     Padded positions are EXCLUDED from masking
                                     (they are already 512 and have no real token
                                     to recover).

    Outputs
        masked_layer0 : [B, S]       layer0_gt with token_mask positions → mask_id.
        targets       : [B, S]       Original token id at masked positions;
                                     ``mask_id`` (= ignore_index) everywhere else.
                                     CrossEntropyLoss(ignore_index=512) will only
                                     update on the masked positions.

    Shape summary::

        layer0_gt     [B, S]
        padding_mask  [B, S]  bool
        ──────────────────────────────────────────
        masked_layer0 [B, S]  long  (input to base transformer)
        targets       [B, S]  long  (supervision target; 512 = ignore)
    """
    B, S = layer0_gt.shape
    device = layer0_gt.device

    # ── sample per-sample masking ratios ──────────────────────────────────
    ratios  = torch.rand(B, device=device) * (ratio_high - ratio_low) + ratio_low
    # ratios: [B]

    # ── random binary mask — True where we APPLY [MASK] ──────────────────
    rand_mat   = torch.rand(B, S, device=device)          # [B, S]
    token_mask = rand_mat < ratios.unsqueeze(1)           # [B, S] bool

    # ── never mask padded positions (they hold no real token) ─────────────
    token_mask = token_mask & ~padding_mask               # [B, S] bool

    # ── build masked input ────────────────────────────────────────────────
    masked_layer0 = layer0_gt.clone()
    masked_layer0[token_mask] = mask_id                   # [B, S]

    # ── build targets: original token at masked positions, ignore elsewhere ──
    # ignore_index = mask_id = 512; CrossEntropyLoss skips these positions
    targets = torch.full_like(layer0_gt, fill_value=mask_id)  # [B, S]
    targets[token_mask] = layer0_gt[token_mask]               # only at masked positions

    return masked_layer0, targets
    # masked_layer0: [B, S]  |  targets: [B, S]


def mask_residuals(
    residuals_gt: torch.Tensor,
    padding_mask: torch.Tensor,
    mask_id: int = MASK_TOKEN_ID,
    ratio_low: float = 0.1,
    ratio_high: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random per-sample masking to the residual RVQ layers (Layers 1–5).

    A single masking pattern is shared across all five residual layers so that
    the model masks the *same time positions* in every layer.  This is more
    coherent than masking each layer independently because the layers are
    tightly coupled (higher layers refine the same underlying motion frame).

    Detailed tensor transformations
    ────────────────────────────────
    Inputs
        residuals_gt : [B, R, S]  Ground-truth residual token ids (R = 5 layers,
                                  0–511 for valid positions, 512 for padding).
        padding_mask : [B, S]     bool — True at padded time positions.
                                  Broadcast to [B, R, S] before applying.

    Internal
        ratios       : [B]        Per-sample masking ratios ~ Uniform(low, high).
        rand_mat     : [B, S]     Random draws — shared across all R layers.
        frame_mask   : [B, S]     bool — which time positions are masked.
        token_mask   : [B, R, S]  bool — frame_mask broadcast over residual layers.

    Outputs
        masked_residuals : [B, R, S]  residuals_gt with token_mask → mask_id.
        targets          : [B, R, S]  Original token at masked positions;
                                      mask_id elsewhere (ignored by loss).

    Shape summary::

        residuals_gt     [B, R, S]
        padding_mask     [B, S]    bool
        ───────────────────────────────────────────────
        masked_residuals [B, R, S] long  (input to residual transformer)
        targets          [B, R, S] long  (supervision target; 512 = ignore)
    """
    B, R, S = residuals_gt.shape
    device  = residuals_gt.device

    # ── sample per-sample masking ratios ──────────────────────────────────
    ratios     = torch.rand(B, device=device) * (ratio_high - ratio_low) + ratio_low
    # ratios: [B]

    # ── one random mask per sample per time-step (shared across R layers) ──
    rand_mat   = torch.rand(B, S, device=device)          # [B, S]
    frame_mask = rand_mat < ratios.unsqueeze(1)           # [B, S] bool

    # ── exclude padded positions ───────────────────────────────────────────
    frame_mask = frame_mask & ~padding_mask               # [B, S] bool

    # ── broadcast the time mask across all R residual layers ──────────────
    # frame_mask.unsqueeze(1): [B, 1, S]  →  expand to [B, R, S]
    token_mask = frame_mask.unsqueeze(1).expand(B, R, S)  # [B, R, S] bool

    # ── build masked residuals and targets ────────────────────────────────
    masked_residuals = residuals_gt.clone()
    masked_residuals[token_mask] = mask_id                # [B, R, S]

    targets = torch.full_like(residuals_gt, fill_value=mask_id)   # [B, R, S]
    targets[token_mask] = residuals_gt[token_mask]

    return masked_residuals, targets
    # masked_residuals: [B, R, S]  |  targets: [B, R, S]


# ─── Training epoch ───────────────────────────────────────────────────────────

def train_one_epoch(
    text_encoder: KSLTextEncoder,
    model:        MoMaskWrapper,
    loader:       torch.utils.data.DataLoader,
    criterion:    nn.CrossEntropyLoss,
    optimizer:    torch.optim.Optimizer,
    scaler:       torch.cuda.amp.GradScaler,
    device:       torch.device,
    epoch:        int,
    cfg:          TrainConfig,
    use_amp:      bool,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Args:
        text_encoder : KSLTextEncoder (T5 backbone)
        model        : MoMaskWrapper  (base + residual motion transformers)
        loader       : DataLoader yielding batches from KSLMotionDataset
        criterion    : CrossEntropyLoss(ignore_index=512)
        optimizer    : AdamW
        scaler       : GradScaler (no-op when AMP disabled)
        device       : target device
        epoch        : current epoch index (1-indexed for display)
        cfg          : TrainConfig
        use_amp      : whether to run forward pass under torch.autocast

    Returns:
        dict with keys ``total``, ``base``, ``residual`` holding average losses
        over the epoch.
    """
    text_encoder.train()
    model.train()

    total_loss    = 0.0
    base_loss_sum = 0.0
    res_loss_sum  = 0.0
    n_batches     = 0
    t0            = time.perf_counter()

    for batch_idx, batch in enumerate(loader):

        # ── move batch to device ──────────────────────────────────────────
        input_ids      = batch["input_ids"].to(device)            # [B, T]
        attention_mask = batch["attention_mask"].to(device)       # [B, T]
        motion_tokens  = batch["motion_tokens"].to(device)        # [B, S, 6]
        pad_mask       = batch["motion_padding_mask"].to(device)  # [B, S] bool

        # ── split motion into Layer 0 and Layers 1-5 ─────────────────────
        # motion_tokens layout: last dim = [layer0, res1, res2, res3, res4, res5]
        layer0_gt    = motion_tokens[:, :, 0]                     # [B, S]
        residuals_gt = motion_tokens[:, :, 1:].permute(0, 2, 1)  # [B, 5, S]
        # permute: [B, S, 5] → [B, 5, S] so residual layers index as dim-1

        # ── MoMask masking strategy ───────────────────────────────────────
        #
        # Layer 0 — random masking, per-sample ratio ~ Uniform(low, high)
        #   masked_layer0 : [B, S]    input with some tokens replaced by 512
        #   layer0_targets: [B, S]    true token at masked positions, 512 elsewhere
        #
        # Layers 1-5 — teacher forcing: GT Layer-0 is given as context (no masking)
        #   separate random masking applied to the residual layers
        #   masked_residuals  : [B, 5, S]
        #   residual_targets  : [B, 5, S]
        masked_layer0,   layer0_targets   = mask_layer0(
            layer0_gt, pad_mask,
            ratio_low=cfg.mask_ratio_low, ratio_high=cfg.mask_ratio_high,
        )
        masked_residuals, residual_targets = mask_residuals(
            residuals_gt, pad_mask,
            ratio_low=cfg.mask_ratio_low, ratio_high=cfg.mask_ratio_high,
        )

        # ── forward pass under AMP context ───────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16,
                                  enabled=use_amp)

        with amp_ctx:

            # ── text embeddings (T5 encoder) ──────────────────────────────
            # text_emb: [B, T, text_encoder.hidden_size]
            text_emb = text_encoder(input_ids, attention_mask)

            # ── motion forward pass ───────────────────────────────────────
            # forward() signature (from model.py):
            #   forward(text_emb, masked_layer0, masked_residuals, gt_layer0)
            #
            # gt_layer0 (teacher forcing): provides the UNMASKED Layer-0 tokens
            # to the residual transformer so it can perfectly condition on Layer 0.
            #
            # base_logits  : [B, S, 512]      Layer-0 predictions
            # res_logits   : [B, 5, S, 512]   Layers 1-5 predictions
            base_logits, res_logits = model(
                text_emb       = text_emb,
                masked_layer0  = masked_layer0,
                masked_residuals = masked_residuals,
                gt_layer0      = layer0_gt,   # teacher forcing for residual model
            )

            # ── loss — Layer 0 ────────────────────────────────────────────
            # CrossEntropyLoss expects [N, C] logits and [N] targets.
            # ignore_index=512 skips positions where target == 512
            # (i.e. unmasked positions and padded positions).
            #
            # base_logits  [B, S, 512]  →  view(-1, 512)  →  [B*S, 512]
            # layer0_targets [B, S]     →  view(-1)        →  [B*S]
            base_loss = criterion(
                base_logits.view(-1, VOCAB_SIZE),   # [B*S, 512]
                layer0_targets.view(-1),            # [B*S]
            )

            # ── loss — Layers 1-5 ─────────────────────────────────────────
            # res_logits    [B, 5, S, 512]  →  view(-1, 512)  →  [B*5*S, 512]
            # residual_targets [B, 5, S]   →  view(-1)        →  [B*5*S]
            res_loss = criterion(
                res_logits.reshape(-1, VOCAB_SIZE),    # [B*5*S, 512]
                residual_targets.reshape(-1),          # [B*5*S]
            )

            # ── total loss: equal weighting of base and residual ──────────
            loss = 0.5 * base_loss + 0.5 * res_loss

        # ── backward + gradient clipping + optimiser step ────────────────
        scaler.scale(loss).backward()

        if cfg.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            all_params = (
                list(text_encoder.parameters()) + list(model.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # ── bookkeeping (skip NaN batches from epoch average) ────────────
        loss_val = loss.item()
        if math.isfinite(loss_val):
            total_loss    += loss_val
            base_loss_sum += base_loss.item()
            res_loss_sum  += res_loss.item()
            n_batches     += 1

        if (batch_idx + 1) % cfg.log_every == 0:
            elapsed     = time.perf_counter() - t0
            its         = n_batches / elapsed
            current_lr  = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:03d} | "
                f"Batch {batch_idx+1:04d}/{len(loader):04d} | "
                f"Loss {loss.item():.4f} "
                f"(base {base_loss.item():.4f}, res {res_loss.item():.4f}) | "
                f"LR {current_lr:.2e} | "
                f"{its:.2f} it/s"
            )

    avg_total = total_loss    / max(n_batches, 1)
    avg_base  = base_loss_sum / max(n_batches, 1)
    avg_res   = res_loss_sum  / max(n_batches, 1)
    return {"total": avg_total, "base": avg_base, "residual": avg_res}


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(
    text_encoder: KSLTextEncoder,
    model:        MoMaskWrapper,
    optimizer:    torch.optim.Optimizer,
    scheduler,
    scaler:       torch.cuda.amp.GradScaler,
    epoch:        int,
    loss:         float,
    cfg:          TrainConfig,
) -> Path:
    """
    Save a full training checkpoint to ``cfg.ckpt_dir``.

    The checkpoint contains every object needed to resume training or run
    inference: model weights, optimiser state, scheduler state, scaler state,
    epoch index, and the config dict.

    Args:
        All training objects (see train_one_epoch for descriptions).
        epoch : completed epoch index (1-indexed).
        loss  : average total loss for this epoch.

    Returns:
        Path to the saved ``.pth`` file.
    """
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.ckpt_dir) / f"checkpoint_epoch_{epoch:03d}.pth"

    torch.save(
        {
            "epoch"              : epoch,
            "loss"               : loss,
            "text_encoder_state" : text_encoder.state_dict(),
            "model_state"        : model.state_dict(),
            "optimizer_state"    : optimizer.state_dict(),
            "scheduler_state"    : scheduler.state_dict(),
            "scaler_state"       : scaler.state_dict(),
            "config"             : cfg.__dict__,
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(
    path:         str,
    text_encoder: KSLTextEncoder,
    model:        MoMaskWrapper,
    optimizer:    torch.optim.Optimizer,
    scheduler,
    scaler:       torch.cuda.amp.GradScaler,
    device:       torch.device,
) -> int:
    """
    Load a checkpoint saved by ``save_checkpoint`` and restore all states.

    Args:
        path         : path to the ``.pth`` file
        All training objects whose states will be restored in-place.
        device       : map_location for the tensors.

    Returns:
        The epoch number stored in the checkpoint (resume from next epoch).
    """
    ckpt = torch.load(path, map_location=device)
    text_encoder.load_state_dict(ckpt["text_encoder_state"])
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    epoch = ckpt["epoch"]
    print(f"[train] Resumed from checkpoint '{path}' (epoch {epoch}, loss {ckpt['loss']:.4f})")
    return epoch


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: TrainConfig) -> None:
    """
    Build all components, run the training loop, and save checkpoints.

    Args:
        cfg : fully populated TrainConfig dataclass.
    """
    torch.manual_seed(cfg.seed)
    device  = get_device()
    use_amp = cfg.use_amp and device.type == "cuda"

    print(f"[train] Device : {device}")
    print(f"[train] AMP    : {use_amp}")
    print(f"[train] Seed   : {cfg.seed}\n")

    # ── tokeniser ─────────────────────────────────────────────────────────
    tokenizer = T5TokenizerFast.from_pretrained(cfg.t5_model)

    # ── data loader ───────────────────────────────────────────────────────
    loader = build_dataloader(
        csv_path    = cfg.csv_path,
        tokenizer   = tokenizer,
        batch_size  = cfg.batch_size,
        max_text_len= cfg.max_text_len,
        use_gloss   = cfg.use_gloss,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = cfg.pin_memory and device.type == "cuda",
    )

    # ── text encoder ──────────────────────────────────────────────────────
    text_encoder = KSLTextEncoder(
        model_name  = cfg.t5_model,
        freeze_base = cfg.freeze_t5,
    ).to(device)
    print(text_encoder)

    # ── motion model ──────────────────────────────────────────────────────
    model = MoMaskWrapper(
        text_dim        = text_encoder.hidden_size,
        hidden_dim      = cfg.hidden_dim,
        base_num_heads  = cfg.base_num_heads,
        base_num_layers = cfg.base_num_layers,
        res_num_heads   = cfg.res_num_heads,
        res_num_layers  = cfg.res_num_layers,
        ffn_dim         = cfg.ffn_dim,
        max_seq_len     = cfg.max_seq_len,
        dropout         = cfg.dropout,
    ).to(device)
    total_motion_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Motion model params : {total_motion_params:,}\n")

    # ── loss ──────────────────────────────────────────────────────────────
    # ignore_index=512: positions where target==512 are excluded from loss.
    # This covers (a) unmasked real positions and (b) padded positions,
    # both of which are assigned target=512 in the masking helpers above.
    criterion = nn.CrossEntropyLoss(ignore_index=MASK_TOKEN_ID)

    # ── optimiser ─────────────────────────────────────────────────────────
    # Collect only parameters that require gradients — T5 params are frozen
    # by default (freeze_t5=True) so they don't appear in the optimiser.
    # If you later call text_encoder.unfreeze_weights(num_layers=N), you will
    # need to rebuild or patch the optimiser to include the new param groups.
    trainable_params = [
        p for p in list(text_encoder.parameters()) + list(model.parameters())
        if p.requires_grad
    ]
    optimizer = AdamW(trainable_params, lr=cfg.lr,
                      weight_decay=cfg.weight_decay, betas=cfg.betas)

    # ── LR schedule: linear warmup → cosine decay ─────────────────────────
    warmup_epochs = min(cfg.warmup_epochs, cfg.num_epochs)
    cosine_epochs = max(cfg.num_epochs - warmup_epochs, 1)

    warmup_sched = LinearLR(
        optimizer,
        start_factor = 1e-2,          # begin at 1% of base LR
        total_iters  = warmup_epochs,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max   = cosine_epochs,
        eta_min = cfg.lr_min,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup_sched, cosine_sched],
        milestones = [warmup_epochs],
    )

    # ── AMP GradScaler ────────────────────────────────────────────────────
    # enabled=False → all scaler calls become no-ops (safe on CPU/MPS)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── optional resume ───────────────────────────────────────────────────
    start_epoch = 0
    if cfg.resume_from:
        start_epoch = load_checkpoint(
            cfg.resume_from, text_encoder, model,
            optimizer, scheduler, scaler, device,
        )

    # ── training loop ─────────────────────────────────────────────────────
    print(f"[train] Starting training for {cfg.num_epochs} epochs "
          f"({len(loader)} batches/epoch)\n")

    for epoch in range(start_epoch + 1, cfg.num_epochs + 1):
        epoch_t0 = time.perf_counter()
        metrics  = train_one_epoch(
            text_encoder = text_encoder,
            model        = model,
            loader       = loader,
            criterion    = criterion,
            optimizer    = optimizer,
            scaler       = scaler,
            device       = device,
            epoch        = epoch,
            cfg          = cfg,
            use_amp      = use_amp,
        )
        scheduler.step()

        elapsed = time.perf_counter() - epoch_t0
        print(
            f"Epoch {epoch:03d}/{cfg.num_epochs:03d} | "
            f"Loss {metrics['total']:.4f} "
            f"(base {metrics['base']:.4f}, res {metrics['residual']:.4f}) | "
            f"{elapsed:.1f}s\n"
        )

        if epoch % cfg.save_every == 0:
            ckpt_path = save_checkpoint(
                text_encoder, model, optimizer, scheduler, scaler,
                epoch, metrics["total"], cfg,
            )
            print(f"[train] Checkpoint saved → {ckpt_path}\n")

    print("[train] Training complete.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> TrainConfig:
    """
    Parse CLI arguments and return a populated ``TrainConfig``.
    Defaults are taken from the dataclass; only the most commonly
    overridden fields are exposed as flags.
    """
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train the KSL text-to-motion MoMask model")

    # paths
    p.add_argument("--csv",        default=defaults.csv_path,    help="path to train.csv")
    p.add_argument("--ckpt_dir",   default=defaults.ckpt_dir,    help="checkpoint output dir")
    p.add_argument("--resume",     default=defaults.resume_from, help="checkpoint to resume from")

    # model
    p.add_argument("--t5",         default=defaults.t5_model,    help="T5 model name")
    p.add_argument("--hidden_dim", default=defaults.hidden_dim,  type=int)
    p.add_argument("--use_gloss",  action="store_true",          help="use gloss column")

    # training
    p.add_argument("--epochs",     default=defaults.num_epochs,  type=int)
    p.add_argument("--batch_size", default=defaults.batch_size,  type=int)
    p.add_argument("--lr",         default=defaults.lr,          type=float)
    p.add_argument("--seed",       default=defaults.seed,        type=int)
    p.add_argument("--no_amp",     action="store_true",          help="disable AMP")
    p.add_argument("--no_freeze",  action="store_true",          help="don't freeze T5 weights")

    args = p.parse_args()

    return TrainConfig(
        csv_path    = args.csv,
        ckpt_dir    = args.ckpt_dir,
        resume_from = args.resume,
        t5_model    = args.t5,
        hidden_dim  = args.hidden_dim,
        use_gloss   = args.use_gloss,
        num_epochs  = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        seed        = args.seed,
        use_amp     = not args.no_amp,
        freeze_t5   = not args.no_freeze,
    )


if __name__ == "__main__":
    main(_parse_args())
