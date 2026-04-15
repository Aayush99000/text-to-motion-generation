"""
dataset_gpt.py — Dataset for T2M-GPT autoregressive training
=============================================================

Each sample is a pair of:
  - Tokenized text (T5 input)
  - Flattened motion token sequence: [BOS, l0_t0, l1_t0, ..., l5_t0, l0_t1, ..., EOS]

The model is trained to predict the next token at every position (causal LM).
PAD positions are excluded from the loss via ignore_index.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5TokenizerFast

from model_gpt import BOS_ID, EOS_ID, PAD_ID, NUM_RVQ_LAYERS

MAX_FRAMES: int = 128   # truncate sequences longer than this


class T2MGPTDataset(Dataset):
    """
    Loads train.csv and builds flattened interleaved token sequences.

    Motion token layout per frame:
        layer0_t, layer1_t, layer2_t, layer3_t, layer4_t, layer5_t

    Full sequence:
        [BOS, l0_t0, l1_t0, ..., l5_t0, l0_t1, ..., l5_tL, EOS, PAD, ...]

    Args:
        csv_path    : path to train.csv
        tokenizer   : T5 tokenizer
        max_text_len: max tokenized text length
        max_frames  : truncate motion to this many frames
        use_gloss   : use gloss column instead of sentence
    """

    RVQ_COLS = ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]

    def __init__(
        self,
        csv_path    : str,
        tokenizer   : T5TokenizerFast,
        max_text_len: int  = 64,
        max_frames  : int  = MAX_FRAMES,
        use_gloss   : bool = False,
    ) -> None:
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len
        self.max_frames   = max_frames
        # Maximum motion tokens per sequence = max_frames * 6
        # Input length = 1 (BOS) + max_frames*6, Target length = max_frames*6 + 1 (EOS)
        self.max_seq_len  = max_frames * NUM_RVQ_LAYERS + 1

        text_col = "gloss" if use_gloss else "sentence"
        df = pd.read_csv(csv_path)

        self.samples: List[Dict] = []
        skipped = 0

        for _, row in df.iterrows():
            try:
                layers = [
                    list(map(int, str(row[col]).split()))
                    for col in self.RVQ_COLS
                ]
            except Exception:
                skipped += 1
                continue

            L = min(min(len(l) for l in layers), max_frames)
            if L == 0:
                skipped += 1
                continue

            # Interleave: frame-first → [l0_t0, l1_t0, ..., l5_t0, l0_t1, ...]
            flat: List[int] = []
            for t in range(L):
                for layer in range(NUM_RVQ_LAYERS):
                    flat.append(layers[layer][t])

            self.samples.append({
                "text"  : str(row[text_col]),
                "tokens": flat,   # L * 6 ints
            })

        lens = [len(s["tokens"]) // NUM_RVQ_LAYERS for s in self.samples]
        print(
            f"[T2MGPTDataset] {len(self.samples)} samples "
            f"(skipped {skipped}) | "
            f"frames: min={min(lens)} mean={sum(lens)//len(lens)} max={max(lens)} | "
            f"max_frames={max_frames}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Tokenize text
        enc = self.tokenizer(
            sample["text"],
            max_length      = self.max_text_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )

        motion = sample["tokens"]   # L*6 ints

        # Build input and target sequences
        # input : [BOS, m0, m1, ..., mN]          length = L*6 + 1
        # target: [m0,  m1, ..., mN, EOS]          length = L*6 + 1
        inp = [BOS_ID] + motion
        tgt = motion + [EOS_ID]

        # Truncate to max_seq_len
        inp = inp[:self.max_seq_len]
        tgt = tgt[:self.max_seq_len]

        # Pad to fixed length
        pad_len = self.max_seq_len - len(inp)
        inp = inp + [PAD_ID] * pad_len
        tgt = tgt + [PAD_ID] * pad_len

        return {
            "input_ids"     : enc["input_ids"].squeeze(0),       # [T_text]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [T_text]
            "motion_input"  : torch.tensor(inp, dtype=torch.long),  # [max_seq_len]
            "motion_target" : torch.tensor(tgt, dtype=torch.long),  # [max_seq_len]
        }


def build_gpt_dataloader(
    csv_path    : str,
    tokenizer   : T5TokenizerFast,
    batch_size  : int  = 8,
    max_text_len: int  = 64,
    max_frames  : int  = MAX_FRAMES,
    use_gloss   : bool = False,
    num_workers : int  = 4,
) -> DataLoader:
    dataset = T2MGPTDataset(
        csv_path, tokenizer,
        max_text_len=max_text_len,
        max_frames=max_frames,
        use_gloss=use_gloss,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
