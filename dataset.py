"""
dataset.py — Data loading and batching for the KSL Text-to-Motion model
=========================================================================

Components
──────────
  KSLMotionDataset : torch.utils.data.Dataset that reads train.csv and loads
                     pre-tokenized motion tensors from disk.
  MotionCollator   : Callable collate function that dynamically pads variable-
                     length motion token sequences within a batch.

Motion token file format
────────────────────────
  Each sample has a corresponding file in the token directory named
  ``{id}.pt`` (PyTorch) or ``{id}.npy`` (NumPy).  The tensor shape is
  ``[seq_len, 6]`` where 6 corresponds to the six RVQ layers
  (layer 0 = base motion, layers 1-5 = fine residuals).

Token ID conventions (must match model.py)
──────────────────────────────────────────
  Valid token ids : 0 … 511
  MASK token id  : 512   (used during training by the masking strategy)
  PAD  token id  : 512   (same value; padded positions are excluded from loss
                          via ``motion_padding_mask``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


# ─── Constants ────────────────────────────────────────────────────────────────

MOTION_PAD_ID: int  = 512   # padding value for motion sequences (= MASK_TOKEN_ID)
NUM_RVQ_LAYERS: int = 6     # number of RVQ layers per motion frame


# ─── Dataset ──────────────────────────────────────────────────────────────────

class KSLMotionDataset(Dataset):
    """
    PyTorch Dataset for Korean Sign Language (KSL) text-to-motion generation.

    Reads sample metadata from a CSV file and lazily loads pre-computed RVQ
    motion tokens from disk on each ``__getitem__`` call.

    CSV schema
    ──────────
    Required columns:
        id       : unique sample identifier (used to locate the motion file)
        sentence : natural English description of the sign
        gloss    : sign-language grammar (space-separated gloss words)

    Args:
        csv_path     : path to ``train.csv`` (or ``val.csv`` / ``test.csv``)
        token_dir    : directory that contains ``{id}.pt`` or ``{id}.npy`` files
        tokenizer    : HuggingFace tokenizer (e.g. ``T5TokenizerFast``)
        max_text_len : maximum number of text tokens; longer sequences are
                       truncated, shorter ones are padded to this length
        use_gloss    : if ``True``, tokenise the ``gloss`` column instead of
                       ``sentence``
        skip_missing : if ``True``, silently drop rows whose motion file cannot
                       be found on disk; if ``False``, raise ``FileNotFoundError``
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        token_dir: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        max_text_len: int = 64,
        use_gloss: bool = False,
        skip_missing: bool = True,
    ) -> None:
        super().__init__()

        self.token_dir   = Path(token_dir)
        self.tokenizer   = tokenizer
        self.max_text_len = max_text_len
        self.text_column = "gloss" if use_gloss else "sentence"

        # ── load and validate the CSV ──────────────────────────────────────
        df = pd.read_csv(csv_path)

        required = {"id", "sentence", "gloss"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        df["id"] = df["id"].astype(str)

        # ── optionally filter rows with missing motion files ───────────────
        if skip_missing:
            present = df["id"].apply(lambda sid: self._resolve_path(sid) is not None)
            n_dropped = (~present).sum()
            if n_dropped:
                print(
                    f"[KSLMotionDataset] Warning: dropping {n_dropped} samples "
                    f"with no motion file in '{self.token_dir}'."
                )
            df = df[present].reset_index(drop=True)
        else:
            # validate upfront so the error is caught at construction time
            for sid in df["id"]:
                if self._resolve_path(sid) is None:
                    raise FileNotFoundError(
                        f"Motion file not found for id='{sid}' in '{self.token_dir}'. "
                        f"Expected '{sid}.pt' or '{sid}.npy'."
                    )

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
        Load and return one training sample.

        Args:
            idx : integer index into the dataset

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
        # tokenizer returns [1, L]; squeeze the batch dim
        input_ids      = encoding["input_ids"].squeeze(0)       # [max_text_len]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [max_text_len]

        # ── load motion tokens ─────────────────────────────────────────────
        motion_tokens = self._load_motion(str(row["id"]))  # [seq_len, 6]

        return {
            "input_ids"     : input_ids,           # [max_text_len]     long
            "attention_mask": attention_mask,       # [max_text_len]     long
            "motion_tokens" : motion_tokens,        # [seq_len, 6]       long
            "motion_length" : motion_tokens.size(0),# scalar             int
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_path(self, sample_id: str) -> Optional[Path]:
        """
        Return the first existing motion file path for ``sample_id``, or
        ``None`` if neither ``.pt`` nor ``.npy`` exists.

        Lookup order: ``{id}.pt`` → ``{id}.npy``
        """
        for ext in (".pt", ".npy"):
            p = self.token_dir / f"{sample_id}{ext}"
            if p.exists():
                return p
        return None

    def _load_motion(self, sample_id: str) -> torch.Tensor:
        """
        Load a motion token tensor from disk.

        Supports both PyTorch (``.pt``) and NumPy (``.npy``) formats.
        The loaded tensor is cast to ``torch.long`` and validated to have
        exactly ``NUM_RVQ_LAYERS`` (6) columns.

        Args:
            sample_id : the string id of the sample (not the full path)

        Returns:
            motion_tokens : [seq_len, NUM_RVQ_LAYERS]  dtype=torch.long

        Raises:
            FileNotFoundError : if neither ``.pt`` nor ``.npy`` exists
            ValueError        : if the loaded tensor does not have 6 columns
        """
        path = self._resolve_path(sample_id)
        if path is None:
            raise FileNotFoundError(
                f"No motion file found for id='{sample_id}' in '{self.token_dir}'."
            )

        if path.suffix == ".pt":
            tokens = torch.load(path, map_location="cpu")
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens)
        else:  # .npy
            tokens = torch.from_numpy(np.load(str(path)))

        tokens = tokens.long()  # ensure dtype is torch.int64

        if tokens.ndim == 1:
            # some files might store a single-layer [seq_len] tensor → unsqueeze
            tokens = tokens.unsqueeze(-1)

        if tokens.shape[-1] != NUM_RVQ_LAYERS:
            raise ValueError(
                f"Expected motion tensor with {NUM_RVQ_LAYERS} RVQ layers "
                f"(columns), got shape {tuple(tokens.shape)} for id='{sample_id}'."
            )

        return tokens  # [seq_len, 6]

    def __repr__(self) -> str:
        return (
            f"KSLMotionDataset("
            f"n_samples={len(self)}, "
            f"text_column='{self.text_column}', "
            f"max_text_len={self.max_text_len}, "
            f"token_dir='{self.token_dir}')"
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
    token_dir: Union[str, Path],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    max_text_len: int = 64,
    use_gloss: bool = False,
    skip_missing: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Convenience function that constructs a ``KSLMotionDataset`` and wraps it
    in a ``DataLoader`` with the custom ``MotionCollator``.

    Args:
        csv_path     : path to the CSV file
        token_dir    : directory containing ``{id}.pt`` / ``{id}.npy`` files
        tokenizer    : HuggingFace tokenizer
        batch_size   : number of samples per batch
        max_text_len : maximum tokenised text length
        use_gloss    : use gloss column instead of sentence
        skip_missing : drop samples whose motion file is absent
        shuffle      : shuffle the dataset each epoch (set False for val/test)
        num_workers  : DataLoader worker processes
        pin_memory   : pin memory for faster GPU transfer

    Returns:
        A ``DataLoader`` whose batches are dicts matching the schema described
        in ``MotionCollator``.
    """
    dataset = KSLMotionDataset(
        csv_path=csv_path,
        token_dir=token_dir,
        tokenizer=tokenizer,
        max_text_len=max_text_len,
        use_gloss=use_gloss,
        skip_missing=skip_missing,
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
    import tempfile, os

    print("Running dataset smoke test …")

    # ── build a tiny synthetic dataset on disk ────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # write a fake CSV
        csv_path = tmpdir / "train.csv"
        pd.DataFrame({
            "id"      : ["001", "002", "003"],
            "sentence": ["a man is running", "she is waving", "they are dancing"],
            "gloss"   : ["MAN RUN", "SHE WAVE", "THEY DANCE"],
        }).to_csv(csv_path, index=False)

        # write fake motion tensors of different lengths
        for sid, length in [("001", 20), ("002", 35), ("003", 12)]:
            tokens = torch.randint(0, 512, (length, NUM_RVQ_LAYERS))
            torch.save(tokens, tmpdir / f"{sid}.pt")

        # use a real T5 tokenizer if available, else fall back to a mock
        try:
            from transformers import T5TokenizerFast
            tokenizer = T5TokenizerFast.from_pretrained("t5-base")
            print("  Using real T5TokenizerFast")
        except Exception:
            # minimal mock tokenizer for environments without model weights
            class _MockTokenizer:
                pad_token_id = 0
                def __call__(self, text, max_length, padding, truncation, return_tensors):
                    ids = torch.zeros(1, max_length, dtype=torch.long)
                    mask = torch.ones(1, max_length, dtype=torch.long)
                    return {"input_ids": ids, "attention_mask": mask}
            tokenizer = _MockTokenizer()
            print("  Using mock tokenizer (transformers not available)")

        # ── dataset ───────────────────────────────────────────────────────
        dataset = KSLMotionDataset(
            csv_path=csv_path,
            token_dir=tmpdir,
            tokenizer=tokenizer,
            max_text_len=32,
            use_gloss=False,
            skip_missing=False,
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

        # verify shapes
        B, max_len = 3, 35   # longest sequence in this batch
        assert batch["motion_tokens"].shape       == (B, max_len, NUM_RVQ_LAYERS)
        assert batch["motion_padding_mask"].shape == (B, max_len)

        # verify that padded positions are correctly flagged
        # sample 003 has length 12; positions 12..34 should be True (padding)
        assert batch["motion_padding_mask"][2, 12:].all()
        assert not batch["motion_padding_mask"][2, :12].any()

        # verify padded positions hold MOTION_PAD_ID
        assert (batch["motion_tokens"][2, 12:, :] == MOTION_PAD_ID).all()

    print("All checks passed.")
