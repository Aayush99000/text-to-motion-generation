"""
inference_gpt.py — T2M-GPT inference and Kaggle submission generation
======================================================================

Run
───
  python inference_gpt.py \
    --checkpoint checkpoints_gpt/checkpoint_epoch_100.pth \
    --test_csv   dataset/test.csv \
    --output     submission_gpt.csv
"""

from __future__ import annotations

import argparse
import csv
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

    text_col       = "gloss" if args.use_gloss else "sentence"
    max_new_tokens = max_frames * NUM_RVQ_LAYERS

    print(f"[inference] Samples    : {len(test_rows)}")
    print(f"[inference] Text col   : '{text_col}'")
    print(f"[inference] Max tokens : {max_new_tokens}")
    print(f"[inference] Temperature: {args.temperature}")
    print(f"[inference] Top-k      : {args.top_k}\n")

    out_rows = []

    for row in tqdm(test_rows, desc="Generating", unit="sample"):
        text = str(row.get(text_col, "") or "")

        enc = tokenizer(
            text,
            max_length      = 64,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        text_emb = text_encoder(input_ids, attention_mask)  # [1, T, D]

        tokens = model.generate(
            text_emb,
            max_new_tokens = max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
        )

        # Ensure length is divisible by NUM_RVQ_LAYERS (6)
        n = (len(tokens) // NUM_RVQ_LAYERS) * NUM_RVQ_LAYERS
        if n == 0:
            # Fallback: at least 1 frame of zeros
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
    p.add_argument("--use_gloss",   action="store_true",
                   help="Use gloss column instead of sentence")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
