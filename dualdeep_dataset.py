"""
dualdeep_dataset.py — PyTorch Dataset for antibody-antigen affinity prediction.

Tokenizes antibody heavy/light chains (separately) for AntiBERTa2-CSSP
and antigen sequences for ESM-2. Uses z-score standardized pKd as target.
"""

import torch
from torch.utils.data import Dataset


class AbAgAffinityDataset(Dataset):
    """
    Dataset for antibody-antigen binding affinity (pKd) prediction.

    Each sample produces:
      - heavy_input_ids, heavy_attention_mask
      - light_input_ids, light_attention_mask
      - ag_input_ids, ag_attention_mask
      - target (pKd_scaled: z-score standardized)
      - cluster_id (antigen cluster label)
    """

    def __init__(self, df, ab_tokenizer, ag_tokenizer,
                 max_ab_len=256, max_ag_len=512, target_col="pKd_scaled"):
        self.df = df.reset_index(drop=True)
        self.ab_tokenizer = ab_tokenizer
        self.ag_tokenizer = ag_tokenizer
        self.max_ab_len = max_ab_len
        self.max_ag_len = max_ag_len
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        heavy_tokens = self.ab_tokenizer(
            str(row["Ab_heavy_chain_seq"]),
            max_length=self.max_ab_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        light_tokens = self.ab_tokenizer(
            str(row["Ab_light_chain_seq"]),
            max_length=self.max_ab_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ag_tokens = self.ag_tokenizer(
            str(row["Ag_seq"]),
            max_length=self.max_ag_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = torch.tensor(row[self.target_col], dtype=torch.float32)
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
