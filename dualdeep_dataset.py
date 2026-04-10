import torch
from torch.utils.data import Dataset

class AbAgAffinityDataset(Dataset):
    def __init__(self, df, ab_tokenizer, ag_tokenizer, max_ab_len=256, max_ag_len=512):
        self.df = df.reset_index(drop=True)
        self.ab_tokenizer = ab_tokenizer
        self.ag_tokenizer = ag_tokenizer
        self.max_ab_len = max_ab_len
        self.max_ag_len = max_ag_len

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
