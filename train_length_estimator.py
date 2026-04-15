"""
train_length_estimator.py
=========================
Quickly trains a LengthEstimator MLP (768 → 1) that predicts motion sequence
length from mean-pooled T5 text embeddings, using the ground-truth lengths
from train.csv.

The T5 encoder weights are loaded from the training checkpoint and kept frozen
throughout. Only the LengthEstimator MLP parameters are optimised.

Run
───
  python train_length_estimator.py \
    --checkpoint checkpoints/checkpoint_epoch_100.pth \
    --csv        dataset/train.csv \
    --output     length_estimator.pth \
    --epochs     30 \
    --batch_size 128
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5TokenizerFast

from text_encoder import KSLTextEncoder
from inference import LengthEstimator, mean_pool_text


# ── Dataset ───────────────────────────────────────────────────────────────────

class LengthDataset(Dataset):
    """Pairs of (sentence, actual_seq_len) from train.csv."""

    def __init__(
        self,
        csv_path: str,
        tokenizer: T5TokenizerFast,
        max_text_len: int = 64,
        use_gloss: bool = False,
    ) -> None:
        df = pd.read_csv(csv_path)
        text_col = "gloss" if use_gloss else "sentence"

        self.texts = df[text_col].fillna("").tolist()
        # sequence length = number of space-separated tokens in base_tokens column
        self.lengths = df["base_tokens"].apply(
            lambda s: len(str(s).split())
        ).tolist()

        self.tokenizer   = tokenizer
        self.max_text_len = max_text_len

        print(
            f"[LengthDataset] {len(self.texts)} samples | "
            f"len range [{min(self.lengths)}, {max(self.lengths)}] | "
            f"mean {sum(self.lengths)/len(self.lengths):.1f}"
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            max_length     = self.max_text_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return (
            enc["input_ids"].squeeze(0),       # [T]
            enc["attention_mask"].squeeze(0),  # [T]
            torch.tensor(self.lengths[idx], dtype=torch.float32),
        )


# ── Training ──────────────────────────────────────────────────────────────────

def train_length_estimator(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[length_estimator] Device: {device}")

    # ── Load T5 encoder from training checkpoint ──────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get("config", {})
    t5_name = cfg.get("t5_model", "t5-base")

    tokenizer = T5TokenizerFast.from_pretrained(t5_name)

    text_encoder = KSLTextEncoder(model_name=t5_name, freeze_base=True)
    text_encoder.load_state_dict(ckpt["text_encoder_state"])
    text_encoder.to(device).eval()
    print(f"[length_estimator] T5 encoder loaded ({t5_name}), frozen.")

    # ── Dataset / DataLoader ──────────────────────────────────────────────
    dataset = LengthDataset(
        csv_path     = args.csv,
        tokenizer    = tokenizer,
        max_text_len = 64,
        use_gloss    = args.use_gloss,
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    le = LengthEstimator(text_dim=text_encoder.hidden_size).to(device)
    print(f"[length_estimator] LengthEstimator params: "
          f"{sum(p.numel() for p in le.parameters()):,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = AdamW(le.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.HuberLoss(delta=10.0)   # robust to outlier lengths

    # ── Training loop ─────────────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        le.train()
        total_loss = 0.0
        total_mae  = 0.0
        n_batches  = 0

        for input_ids, attention_mask, target_len in tqdm(
            loader, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False
        ):
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_len     = target_len.to(device)

            # encode text (no grad for T5)
            with torch.no_grad():
                text_emb = text_encoder(input_ids, attention_mask)  # [B, T, D]
                pooled   = mean_pool_text(text_emb, attention_mask) # [B, D]

            pred = le(pooled)                                        # [B]
            loss = criterion(pred, target_len)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(le.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                mae = (pred.round() - target_len).abs().mean().item()

            total_loss += loss.item()
            total_mae  += mae
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        avg_mae  = total_mae  / max(n_batches, 1)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"HuberLoss {avg_loss:.4f} | MAE {avg_mae:.2f} frames"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(le.state_dict(), args.output)
            print(f"  → Best model saved → {args.output}")

    print(f"\n[length_estimator] Done. Best HuberLoss: {best_loss:.4f}")
    print(f"[length_estimator] Saved → {args.output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LengthEstimator from T5 embeddings → seq len"
    )
    p.add_argument("--checkpoint",  required=True,
                   help="Path to training checkpoint (checkpoint_epoch_100.pth)")
    p.add_argument("--csv",         required=True,
                   help="Path to train.csv")
    p.add_argument("--output",      default="length_estimator.pth",
                   help="Output path for length_estimator.pth")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--use_gloss",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train_length_estimator(_parse_args())
