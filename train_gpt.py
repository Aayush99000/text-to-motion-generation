"""
train_gpt.py — T2M-GPT training loop
=====================================

Trains a causal GPT decoder to autoregressively generate motion tokens
conditioned on T5 text embeddings via cross-attention.

Run
───
  python train_gpt.py \
    --csv        dataset/train.csv \
    --ckpt_dir   checkpoints_gpt \
    --epochs     100 \
    --batch_size 8 \
    --lr         3e-4
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from transformers import T5TokenizerFast

from dataset_gpt import MAX_FRAMES, build_gpt_dataloader
from model_gpt import PAD_ID, T2MGPT
from text_encoder import KSLTextEncoder


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    csv       : str   = "dataset/train.csv"
    ckpt_dir  : str   = "checkpoints_gpt"
    epochs    : int   = 100
    batch_size: int   = 8
    lr        : float = 3e-4
    t5        : str   = "t5-base"
    max_frames: int   = MAX_FRAMES
    grad_clip : float = 1.0
    warmup    : int   = 500
    seed      : int   = 42
    use_gloss : bool  = False
    resume    : str   = ""


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device : {device}")

    # ── Text encoder (frozen) ─────────────────────────────────────────────
    tokenizer    = T5TokenizerFast.from_pretrained(cfg.t5)
    text_encoder = KSLTextEncoder(model_name=cfg.t5, freeze_base=True).to(device)
    text_encoder.eval()

    # ── Data ──────────────────────────────────────────────────────────────
    loader = build_gpt_dataloader(
        cfg.csv,
        tokenizer,
        batch_size  = cfg.batch_size,
        max_frames  = cfg.max_frames,
        use_gloss   = cfg.use_gloss,
        num_workers = 4,
    )
    print(f"[train] Batches/epoch : {len(loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    max_seq_len = cfg.max_frames * 6 + 2   # BOS + max_frames*6 tokens + EOS
    model = T2MGPT(
        text_dim    = text_encoder.hidden_size,
        hidden_dim  = 512,
        num_heads   = 8,
        num_layers  = 8,
        ffn_dim     = 2048,
        max_seq_len = max_seq_len,
        dropout     = 0.1,
    ).to(device)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"]
        print(f"[train] Resumed from epoch {start_epoch} | loss {ckpt.get('loss', '?'):.4f}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1, betas=(0.9, 0.95))

    total_steps  = cfg.epochs * len(loader)
    warmup_sched = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=cfg.warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(total_steps - cfg.warmup, 1), eta_min=1e-6)
    scheduler    = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[cfg.warmup])

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        for batch in tqdm(loader, desc=f"Epoch {epoch:03d}/{cfg.epochs}", leave=False, ncols=100):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            motion_input   = batch["motion_input"].to(device)
            motion_target  = batch["motion_target"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():
                    text_emb = text_encoder(input_ids, attention_mask)  # [B, T, D]

                logits = model(motion_input, text_emb)   # [B, T, TOTAL_VOCAB]
                B, T, V = logits.shape
                loss = criterion(logits.view(B * T, V), motion_target.view(B * T))

            loss_val = loss.item()
            if math.isfinite(loss_val):
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            total_loss += loss_val
            n_batches  += 1
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"Loss {avg_loss:.4f} | "
            f"LR {lr_now:.2e} | "
            f"{elapsed:.1f}s"
        )

        # ── Save checkpoint ───────────────────────────────────────────────
        ckpt_path = Path(cfg.ckpt_dir) / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save({
            "epoch"              : epoch,
            "model_state"        : model.state_dict(),
            "text_encoder_state" : text_encoder.state_dict(),
            "config"             : cfg.__dict__,
            "loss"               : avg_loss,
        }, ckpt_path)
        print(f"[train] Checkpoint saved → {ckpt_path}")

    print("[train] Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train T2M-GPT")
    p.add_argument("--csv",        default="dataset/train.csv")
    p.add_argument("--ckpt_dir",   default="checkpoints_gpt")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--t5",         default="t5-base")
    p.add_argument("--max_frames", type=int,   default=MAX_FRAMES)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--warmup",     type=int,   default=500)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--use_gloss",  action="store_true")
    p.add_argument("--resume",     default="")
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
