"""
dataset.py — Data loading and batching for the KSL Text-to-Motion model
=========================================================================

Components
──────────
  KSLMotionDataset : torch.utils.data.Dataset that reads train.csv and parses
                     pre-tokenized RVQ motion tokens from CSV columns.
  MotionCollator   : Callable collate function that dynamically pads variable-
                     length motion token sequences within a batch.

CSV token columns
─────────────────
  The competition CSV contains six space-separated token columns per row:
      base_tokens  : Layer-0 (base) token ids
      residual_1   : Residual layer 1 token ids
      residual_2   : Residual layer 2 token ids
      residual_3   : Residual layer 3 token ids
      residual_4   : Residual layer 4 token ids
      residual_5   : Residual layer 5 token ids

  These are parsed and stacked into a [seq_len, 6] tensor per sample.

Token ID conventions (must match model.py)
──────────────────────────────────────────
  Valid token ids : 0 … 511
  MASK token id  : 512   (used during training by the masking strategy)
  PAD  token id  : 512   (same value; padded positions are excluded from loss
                          via ``motion_padding_mask``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


# ─── Constants ────────────────────────────────────────────────────────────────

MOTION_PAD_ID: int  = 512   # padding value for motion sequences (= MASK_TOKEN_ID)
NUM_RVQ_LAYERS: int = 6     # number of RVQ layers per motion frame

# Ordered column names for the 6 RVQ layers in the CSV
_TOKEN_COLUMNS = ["base_tokens", "residual_1", "residual_2",
                  "residual_3",  "residual_4", "residual_5"]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class KSLMotionDataset(Dataset):
    """
    PyTorch Dataset for Korean Sign Language (KSL) text-to-motion generation.

    Reads sample metadata and pre-computed RVQ motion tokens directly from
    the competition CSV file.  No separate token files are required.

    CSV schema
    ──────────
    Required columns:
        id          : unique sample identifier
        sentence    : natural English description of the sign
        gloss       : sign-language grammar (space-separated gloss words)
        base_tokens : space-separated Layer-0 token ids
        residual_1 … residual_5 : space-separated residual layer token ids

    Args:
        csv_path     : path to ``train.csv`` (or ``test.csv``)
        tokenizer    : HuggingFace tokenizer (e.g. ``T5TokenizerFast``)
        max_text_len : maximum number of text tokens; longer sequences are
                       truncated, shorter ones are padded to this length
        use_gloss    : if ``True``, tokenise the ``gloss`` column instead of
                       ``sentence``
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        max_text_len: int = 64,
        max_motion_len: int = 512,
        use_gloss: bool = False,
    ) -> None:
        super().__init__()

        self.tokenizer     = tokenizer
        self.max_text_len  = max_text_len
        self.max_motion_len = max_motion_len
        self.text_column   = "gloss" if use_gloss else "sentence"

        # ── load and validate the CSV ──────────────────────────────────────
        df = pd.read_csv(csv_path)

        required = {"id", "sentence"} | set(_TOKEN_COLUMNS)
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        df["id"] = df["id"].astype(str)

        # Drop rows where any token column is NaN (test set has no tokens)
        token_present = df[_TOKEN_COLUMNS].notna().all(axis=1)
        n_dropped = (~token_present).sum()
        if n_dropped:
            print(
                f"[KSLMotionDataset] Warning: dropping {n_dropped} rows "
                f"with missing token columns (test-set rows?)."
            )
        df = df[token_present].reset_index(drop=True)

        self.records: pd.DataFrame = df
        print(
            f"[KSLMotionDataset] Loaded {len(self.records)} samples "
            f"from '{csv_path}' | text column: '{self.text_column}'"
        )

    # ── Core interface ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Return one training sample parsed from the CSV row.

        Returns a dictionary with the following keys:

        +-----------------+-----------------------------+-------------------+
        | key             | shape                       | dtype             |
        +=================+=============================+===================+
        | input_ids       | [max_text_len]              | torch.long        |
        | attention_mask  | [max_text_len]              | torch.long        |
        | motion_tokens   | [seq_len, NUM_RVQ_LAYERS]   | torch.long        |
        | motion_length   | scalar int                  | Python int        |
        +-----------------+-----------------------------+-------------------+

        ``motion_length`` is the un-padded sequence length and is used by
        ``MotionCollator`` to build the padding mask after batching.
        """
        row = self.records.iloc[idx]

        # ── tokenise text ──────────────────────────────────────────────────
        text = str(row[self.text_column])
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",   # fixed length → simple stacking in collate
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)       # [max_text_len]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [max_text_len]

        # ── parse motion tokens from CSV columns ───────────────────────────
        motion_tokens = self._parse_tokens(row)  # [seq_len, 6]

        return {
            "input_ids"     : input_ids,
            "attention_mask": attention_mask,
            "motion_tokens" : motion_tokens,
            "motion_length" : motion_tokens.size(0),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_tokens(self, row: pd.Series) -> torch.Tensor:
        """
        Parse the six token columns from a CSV row into a [seq_len, 6] tensor.

        All six columns must have the same number of space-separated integers
        (one per motion frame).  Token ids are clipped to [0, 511] as a safety
        measure against any out-of-range values in the source data.
        """
        layers = []
        for col in _TOKEN_COLUMNS:
            ids = [int(x) for x in str(row[col]).split()]
            layers.append(ids)

        # All layers must have the same sequence length
        seq_len = len(layers[0])
        for i, layer in enumerate(layers[1:], 1):
            if len(layer) != seq_len:
                raise ValueError(
                    f"Token column length mismatch for id='{row['id']}': "
                    f"base_tokens has {seq_len} tokens but "
                    f"'{_TOKEN_COLUMNS[i]}' has {len(layer)} tokens."
                )

        # Stack into [6, seq_len] then transpose to [seq_len, 6]
        tensor = torch.tensor(layers, dtype=torch.long).T  # [seq_len, 6]

        # Truncate to max_motion_len to keep memory bounded
        if tensor.size(0) > self.max_motion_len:
            tensor = tensor[:self.max_motion_len, :]

        return tensor

    def __repr__(self) -> str:
        return (
            f"KSLMotionDataset("
            f"n_samples={len(self)}, "
            f"text_column='{self.text_column}', "
            f"max_text_len={self.max_text_len})"
        )


# ─── Collate Function ─────────────────────────────────────────────────────────

class MotionCollator:
    """
    Callable collate object for use with ``torch.utils.data.DataLoader``.

    Handles the critical task of dynamically padding variable-length motion
    token sequences to the **maximum length within each batch**.  Text
    tensors (``input_ids``, ``attention_mask``) are already fixed-length
    from the tokenizer and are simply stacked.

    Args:
        motion_pad_id : token id used to fill padding positions in the motion
                        tensor (default ``512``, matching ``MOTION_PAD_ID``)

    Usage::

        collator   = MotionCollator(motion_pad_id=512)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)

    Batched output dictionary
    ─────────────────────────
    +----------------------+----------------------------------+-----------+
    | key                  | shape                            | dtype     |
    +======================+==================================+===========+
    | input_ids            | [B, max_text_len]                | long      |
    | attention_mask       | [B, max_text_len]                | long      |
    | motion_tokens        | [B, max_motion_len, 6]           | long      |
    | motion_lengths       | [B]                              | long      |
    | motion_padding_mask  | [B, max_motion_len]              | bool      |
    +----------------------+----------------------------------+-----------+

    ``motion_padding_mask[b, t]`` is ``True`` where position ``t`` of sample
    ``b`` is padding and should be **excluded** from the loss.
    """

    def __init__(self, motion_pad_id: int = MOTION_PAD_ID) -> None:
        self.motion_pad_id = motion_pad_id

    def __call__(
        self,
        batch: List[Dict[str, Union[torch.Tensor, int]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples returned by ``KSLMotionDataset.__getitem__``.

        Args:
            batch : list of per-sample dicts, each with keys
                    ``input_ids``, ``attention_mask``, ``motion_tokens``,
                    ``motion_length``

        Returns:
            A single dict with batched tensors; see class docstring for shapes.
        """
        # ── fixed-length text tensors: simple stack ────────────────────────
        input_ids      = torch.stack([s["input_ids"]      for s in batch], dim=0)
        attention_mask = torch.stack([s["attention_mask"] for s in batch], dim=0)
        # shapes: [B, max_text_len]

        # ── variable-length motion tensors: pad to batch maximum ───────────
        motion_lengths = torch.tensor(
            [s["motion_length"] for s in batch], dtype=torch.long
        )  # [B]

        max_motion_len = int(motion_lengths.max().item())
        B              = len(batch)

        # allocate the padded buffer: [B, max_motion_len, 6]
        motion_tokens = torch.full(
            (B, max_motion_len, NUM_RVQ_LAYERS),
            fill_value=self.motion_pad_id,
            dtype=torch.long,
        )

        for i, sample in enumerate(batch):
            seq: torch.Tensor = sample["motion_tokens"]  # [seq_len_i, 6]
            seq_len = seq.size(0)
            motion_tokens[i, :seq_len, :] = seq          # copy into padded buffer

        # ── padding mask: True where the position is PAD ──────────────────
        # arange trick: compare position indices against each sample's true length
        # positions[t] >= motion_lengths[b]  →  padding
        positions = torch.arange(max_motion_len, dtype=torch.long).unsqueeze(0)
        # positions : [1, max_motion_len]
        # motion_lengths.unsqueeze(1) : [B, 1]
        motion_padding_mask = positions >= motion_lengths.unsqueeze(1)
        # result: [B, max_motion_len], dtype=bool

        return {
            "input_ids"           : input_ids,           # [B, max_text_len]      long
            "attention_mask"      : attention_mask,       # [B, max_text_len]      long
            "motion_tokens"       : motion_tokens,        # [B, max_motion_len, 6] long
            "motion_lengths"      : motion_lengths,       # [B]                    long
            "motion_padding_mask" : motion_padding_mask,  # [B, max_motion_len]    bool
        }


# ─── Convenience factory ──────────────────────────────────────────────────────

def build_dataloader(
    csv_path: Union[str, Path],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    max_text_len: int = 64,
    max_motion_len: int = 512,
    use_gloss: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Convenience function that constructs a ``KSLMotionDataset`` and wraps it
    in a ``DataLoader`` with the custom ``MotionCollator``.

    Tokens are read directly from the CSV columns — no separate token
    directory is needed.

    Args:
        csv_path     : path to the CSV file (must contain token columns)
        tokenizer    : HuggingFace tokenizer
        batch_size   : number of samples per batch
        max_text_len : maximum tokenised text length
        use_gloss    : use gloss column instead of sentence
        shuffle      : shuffle the dataset each epoch (set False for val/test)
        num_workers  : DataLoader worker processes
        pin_memory   : pin memory for faster GPU transfer

    Returns:
        A ``DataLoader`` whose batches are dicts matching the schema described
        in ``MotionCollator``.
    """
    dataset = KSLMotionDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_text_len=max_text_len,
        max_motion_len=max_motion_len,
        use_gloss=use_gloss,
    )
    collator = MotionCollator(motion_pad_id=MOTION_PAD_ID)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("Running dataset smoke test …")

    def _make_token_str(length: int) -> str:
        return " ".join(str(torch.randint(0, 512, (1,)).item()) for _ in range(length))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write a fake CSV with inline token columns
        rows = []
        for sid, length in [("001", 20), ("002", 35), ("003", 12)]:
            rows.append({
                "id"        : sid,
                "sentence"  : f"sentence for {sid}",
                "gloss"     : f"GLOSS {sid}",
                "base_tokens": _make_token_str(length),
                "residual_1" : _make_token_str(length),
                "residual_2" : _make_token_str(length),
                "residual_3" : _make_token_str(length),
                "residual_4" : _make_token_str(length),
                "residual_5" : _make_token_str(length),
            })
        csv_path = tmpdir / "train.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        # use a real T5 tokenizer if available, else fall back to a mock
        try:
            from transformers import T5TokenizerFast
            tokenizer = T5TokenizerFast.from_pretrained("t5-base")
            print("  Using real T5TokenizerFast")
        except Exception:
            class _MockTokenizer:
                pad_token_id = 0
                def __call__(self, text, max_length, padding, truncation, return_tensors):
                    ids  = torch.zeros(1, max_length, dtype=torch.long)
                    mask = torch.ones(1, max_length, dtype=torch.long)
                    return {"input_ids": ids, "attention_mask": mask}
            tokenizer = _MockTokenizer()
            print("  Using mock tokenizer (transformers not available)")

        # ── dataset ───────────────────────────────────────────────────────
        dataset = KSLMotionDataset(
            csv_path=csv_path,
            tokenizer=tokenizer,
            max_text_len=32,
            use_gloss=False,
        )
        print(f"  {dataset}")
        assert len(dataset) == 3

        sample = dataset[0]
        print(f"  sample keys        : {list(sample.keys())}")
        print(f"  input_ids shape    : {tuple(sample['input_ids'].shape)}")
        print(f"  attention_mask     : {tuple(sample['attention_mask'].shape)}")
        print(f"  motion_tokens shape: {tuple(sample['motion_tokens'].shape)}")
        print(f"  motion_length      : {sample['motion_length']}")

        # ── collation ─────────────────────────────────────────────────────
        collator = MotionCollator(motion_pad_id=MOTION_PAD_ID)
        batch    = collator([dataset[0], dataset[1], dataset[2]])

        print(f"  input_ids (batched)          : {tuple(batch['input_ids'].shape)}")
        print(f"  attention_mask (batched)     : {tuple(batch['attention_mask'].shape)}")
        print(f"  motion_tokens (batched)      : {tuple(batch['motion_tokens'].shape)}")
        print(f"  motion_lengths               : {batch['motion_lengths'].tolist()}")
        print(f"  motion_padding_mask (batched): {tuple(batch['motion_padding_mask'].shape)}")

        B, max_len = 3, 35
        assert batch["motion_tokens"].shape       == (B, max_len, NUM_RVQ_LAYERS)
        assert batch["motion_padding_mask"].shape == (B, max_len)
        assert batch["motion_padding_mask"][2, 12:].all()
        assert not batch["motion_padding_mask"][2, :12].any()
        assert (batch["motion_tokens"][2, 12:, :] == MOTION_PAD_ID).all()

    print("All checks passed.")
