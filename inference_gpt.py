"""
inference_gpt.py — T2M-GPT inference and Kaggle submission generation
======================================================================

Run
───
  python inference_gpt.py \
    --checkpoint checkpoints_gpt/checkpoint_epoch_100.pth \
    --test_csv   dataset/test.csv \
    --output     submission_gpt.csv \
    --batch_size 32
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5TokenizerFast

from model_gpt import NUM_RVQ_LAYERS, T2MGPT
from text_encoder import KSLTextEncoder


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("config", {})

    t5_name    = cfg.get("t5",         "t5-base")
    max_frames = cfg.get("max_frames", 128)

    tokenizer = T5TokenizerFast.from_pretrained(t5_name)

    text_encoder = KSLTextEncoder(model_name=t5_name, freeze_base=True)
    text_encoder.load_state_dict(ckpt["text_encoder_state"])
    text_encoder.to(device).eval()

    model = T2MGPT(
        text_dim    = text_encoder.hidden_size,
        max_seq_len = max_frames * NUM_RVQ_LAYERS + 2,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print(f"[inference] Loaded  : {checkpoint_path}")
    print(f"[inference] Epoch   : {ckpt.get('epoch', '?')} | Loss: {ckpt.get('loss', '?'):.4f}")
    print(f"[inference] T5      : {t5_name}")
    print(f"[inference] MaxFrame: {max_frames}")

    return model, text_encoder, tokenizer, max_frames


# ── Inference loop ────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Device  : {device}\n")

    model, text_encoder, tokenizer, max_frames = load_model(args.checkpoint, device)

    # Read test CSV
    with open(args.test_csv, newline="") as f:
        test_rows = list(csv.DictReader(f))

    if args.use_both:
        text_col = "both"
    elif args.use_gloss:
        text_col = "gloss"
    else:
        text_col = "sentence"
    max_new_tokens = max_frames * NUM_RVQ_LAYERS
    batch_size     = args.batch_size
    n_batches      = math.ceil(len(test_rows) / batch_size)

    print(f"[inference] Samples    : {len(test_rows)}")
    print(f"[inference] Batch size : {batch_size}  ({n_batches} batches)")
    print(f"[inference] Text col   : '{text_col}'")
    print(f"[inference] Max tokens : {max_new_tokens}")
    print(f"[inference] Temperature: {args.temperature}")
    print(f"[inference] Top-k      : {args.top_k}\n")

    out_rows = []

    for batch_idx in tqdm(range(n_batches), desc="Generating", unit="batch"):
        batch_rows = test_rows[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        if text_col == "both":
            texts = [
                (str(row.get("gloss", "") or "").strip() + " " +
                 str(row.get("sentence", "") or "").strip()).strip()
                for row in batch_rows
            ]
        else:
            texts = [str(row.get(text_col, "") or "") for row in batch_rows]

        enc = tokenizer(
            texts,
            max_length     = 128,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        text_emb = text_encoder(input_ids, attention_mask)  # [B, T, D]

        batch_tokens = model.generate_batch(
            text_emb,
            max_new_tokens = max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
        )

        for row, tokens in zip(batch_rows, batch_tokens):
            # Ensure length is divisible by NUM_RVQ_LAYERS (6)
            n = (len(tokens) // NUM_RVQ_LAYERS) * NUM_RVQ_LAYERS
            if n == 0:
                tokens = [0] * NUM_RVQ_LAYERS
                n = NUM_RVQ_LAYERS
            else:
                tokens = tokens[:n]

            # Reshape [n] → [L, 6]
            L   = n // NUM_RVQ_LAYERS
            arr = [tokens[i * NUM_RVQ_LAYERS:(i + 1) * NUM_RVQ_LAYERS] for i in range(L)]

            out_rows.append({
                "id"         : row["id"],
                "base_tokens": " ".join(str(arr[i][0]) for i in range(L)),
                "residual_1" : " ".join(str(arr[i][1]) for i in range(L)),
                "residual_2" : " ".join(str(arr[i][2]) for i in range(L)),
                "residual_3" : " ".join(str(arr[i][3]) for i in range(L)),
                "residual_4" : " ".join(str(arr[i][4]) for i in range(L)),
                "residual_5" : " ".join(str(arr[i][5]) for i in range(L)),
            })

    # Write submission CSV
    fieldnames = ["id", "base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n[inference] Saved → {args.output}  ({len(out_rows)} rows)")
    print(f"[inference] Preview:")
    for row in out_rows[:2]:
        tokens_preview = " ".join(row["base_tokens"].split()[:10])
        print(f"  id={row['id']}  base_tokens={tokens_preview}...")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T2M-GPT inference")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to checkpoint_epoch_100.pth")
    p.add_argument("--test_csv",    required=True,
                   help="Path to test.csv")
    p.add_argument("--output",      default="submission_gpt.csv",
                   help="Output submission CSV path")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature (lower = sharper)")
    p.add_argument("--top_k",       type=int,   default=256,
                   help="Top-k logit filter")
    p.add_argument("--batch_size",  type=int,   default=32,
                   help="Number of samples to generate in parallel")
    p.add_argument("--use_gloss",   action="store_true",
                   help="Use gloss column instead of sentence")
    p.add_argument("--use_both",    action="store_true",
                   help="Concatenate gloss + sentence as input text")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
