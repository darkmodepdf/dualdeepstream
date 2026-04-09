"""
dataset.py — PyTorch Dataset for antibody-antigen affinity prediction.

Tokenizes antibody heavy/light chains (separately) for AntiBERTa2-CSSP
and antigen sequences for ESM-2. Returns tensors ready for the DualEncoderModel.

This is the ONLY standalone .py file. Justification: reused across training,
validation, testing, and the NN baseline computation. Keeping it separate
avoids code duplication and keeps the notebook clean.
"""

import torch
from torch.utils.data import Dataset


class AbAgAffinityDataset(Dataset):
    """
    Dataset for antibody-antigen binding affinity (pKd) prediction.

    Each sample produces:
      - heavy_input_ids, heavy_attention_mask   (AntiBERTa2-CSSP tokenized heavy chain)
      - light_input_ids, light_attention_mask   (AntiBERTa2-CSSP tokenized light chain)
      - ag_input_ids, ag_attention_mask          (ESM-2 tokenized antigen)
      - target                                   (pKd float scalar)
      - cluster_id                               (antigen cluster label, for stratification)
    """

    def __init__(self, df, ab_tokenizer, ag_tokenizer, max_ab_len=512, max_ag_len=512):
        """
        Args:
            df: DataFrame with columns Ab_heavy_chain_seq, Ab_light_chain_seq,
                Ag_seq, pKd, ag_cluster_40
            ab_tokenizer: HuggingFace tokenizer for AntiBERTa2-CSSP
            ag_tokenizer: HuggingFace tokenizer for ESM-2
            max_ab_len: max token length for each antibody chain
            max_ag_len: max token length for antigen sequence
        """
        self.df = df.reset_index(drop=True)
        self.ab_tokenizer = ab_tokenizer
        self.ag_tokenizer = ag_tokenizer
        self.max_ab_len = max_ab_len
        self.max_ag_len = max_ag_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Antibody: encode heavy and light chains SEPARATELY ---
        heavy_tokens = self.ab_tokenizer(
            row["Ab_heavy_chain_seq"],
            max_length=self.max_ab_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        light_tokens = self.ab_tokenizer(
            row["Ab_light_chain_seq"],
            max_length=self.max_ab_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # --- Antigen ---
        ag_tokens = self.ag_tokenizer(
            row["Ag_seq"],
            max_length=self.max_ag_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = torch.tensor(row["pKd"], dtype=torch.float32)
        cluster_id = torch.tensor(row["ag_cluster_40"], dtype=torch.long)

        return {
            "heavy_input_ids": heavy_tokens["input_ids"].squeeze(0),
            "heavy_attention_mask": heavy_tokens["attention_mask"].squeeze(0),
            "light_input_ids": light_tokens["input_ids"].squeeze(0),
            "light_attention_mask": light_tokens["attention_mask"].squeeze(0),
            "ag_input_ids": ag_tokens["input_ids"].squeeze(0),
            "ag_attention_mask": ag_tokens["attention_mask"].squeeze(0),
            "target": target,
            "cluster_id": cluster_id,
        }
