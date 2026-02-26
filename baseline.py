import numpy as np
import pandas as pd
from pathlib import Path
import os

DATA_PATH = "/kaggle/input/motion-s-hierarchical-text-to-motion-generation-for-sign-language"

train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
test_df = pd.read_csv(f"{DATA_PATH}/test.csv")
motion_root = DATA_PATH+ "/Motion-Features"

print("Train:", train_df.shape)
print("Test:", test_df.shape)
train_df.head()

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

#Helper function
def load_metadata(sample_dir: Path) -> dict:
    metadata_path = sample_dir / "metadata.txt"
    if not metadata_path.exists():
        return {}
    result = {}

    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith("SENTENCE:"):
                result["sentence"] =line.split(":",1)[1].strip()
            elif line.startswith("GLOSS:"):
                result["gloss"] = line.split(":",1)[1].strip()

    return result

from pathlib import Path

def count_bvh_files(sample_dir: Path) -> int:
    return len(list(sample_dir.glob("*.bvh")))

def parse_glosses(gloss_str: str) -> list:
    cleaned = gloss_str.replace("//", "").strip()
    return [g.strip() for g in cleaned.split() if g.strip()]

def is_fingerspelling(gloss: str) -> bool:
    """Check if a gloss is a fingerspelled letter (single uppercase letter)."""
    return len(gloss) == 1 and gloss.isupper()

import torch
from torch.nn.utils.rnn import pad_sequence 

#Collate function
"""Collate functions will pad shorter sequences (for variable motion length)"""
def collate_fn(batch):
    sentences = [b[0] for b in batch]

    motions= [torch.tensor(b[1], dtype=torch.float32) for b in batch ]
    lengths= torch.tensor([m.shape[0] for m in motions])

    motions = pad_sequence(motions, batch_first =True)

    return sentences,motions,lenghts
    

'''DATASET CLASS'''
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self, df, motion_root):
        self.df = df.reset_index(drop=True)
        self.motion_root = Path(motion_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sid = str(row["id"])
        sentence = row["sentence"]

        npy_path = self.motion_root / f"{sid}.npy"
        motion = np.load(npy_path)

        return sentence, motion


dataset = MotionDataset(train_df, motion_root)
print(len(dataset))

"""Dataset sample"""
s, m = dataset[0]

print(type(s))
print(m.shape)