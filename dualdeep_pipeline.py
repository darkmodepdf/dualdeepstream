# %% [markdown]
# # DuaDeep-SeqAffinity: Antibody-Antigen pKd Prediction Pipeline
#
# **Architecture:** AntiBERTa2-CSSP (antibody H+L, separate) + ESM-2 150M (antigen) → 256-d projections → MLP → pKd
#
# **Dataset:** AbRank (~342K rows)
#
# **Hardware:** Single A100, mixed precision (torch.cuda.amp)

# %% [markdown]
# ## §0 — Environment Verification

# %%
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer
import pandas as pd
import numpy as np
import os, sys, re, csv, glob, subprocess, json
from pathlib import Path
from datetime import datetime

print(f"PyTorch:      {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA:         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
assert torch.cuda.is_available(), "❌ No GPU detected — aborting"
print("✓ Environment OK")

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Paths
DATA_DIR = Path(".")
DATA_FILE = DATA_DIR / "AbRank_dataset.csv"
CKPT_DIR = DATA_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "training_log.csv"

# %% [markdown]
# ## §1 — Data Loading & Cleaning

# %%
print("Loading dataset...")
df_raw = pd.read_csv(DATA_FILE, sep="\t")
print(f"Raw dataset: {len(df_raw):,} rows, {df_raw.shape[1]} columns")
print(f"Columns: {list(df_raw.columns)}")

# %%
# --- Filtering ---

# 1. Keep only exact affinity measurements (not censored > or <)
df = df_raw[df_raw["Aff_op"] == "="].copy()
print(f"After Aff_op='=' filter: {len(df):,} rows")

# 2. Drop rows with missing critical fields
df = df.dropna(subset=["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq", "Affinity_Kd [nM]"])
print(f"After dropping NaN: {len(df):,} rows")

# Convert Affinity_Kd [nM] to numeric, turning non-parseable values to NaN
df["Affinity_Kd [nM]"] = pd.to_numeric(df["Affinity_Kd [nM]"], errors='coerce')
df = df.dropna(subset=["Affinity_Kd [nM]"])

# 3. Filter biologically meaningful Kd range: 1e-3 < Kd < 1e9 nM
df = df[(df["Affinity_Kd [nM]"] > 1e-3) & (df["Affinity_Kd [nM]"] < 1e9)]
print(f"After Kd range filter: {len(df):,} rows")

# 4. Clean sequences — keep only standard amino acids, replace others with X
def clean_seq(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', s.upper().strip())

for col in ["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"]:
    df[col] = df[col].apply(clean_seq)

# Remove rows where any sequence is empty
df = df[(df["Ab_heavy_chain_seq"].str.len() > 0) &
        (df["Ab_light_chain_seq"].str.len() > 0) &
        (df["Ag_seq"].str.len() > 0)]
print(f"After sequence cleaning: {len(df):,} rows")

# 5. Compute pKd target (positively correlated with binding strength)
df["pKd"] = 9 - np.log10(df["Affinity_Kd [nM]"])

print(f"\n--- Dataset Summary ---")
print(f"Final samples:      {len(df):,}")
print(f"pKd range:          [{df['pKd'].min():.2f}, {df['pKd'].max():.2f}]")
print(f"pKd mean ± std:     {df['pKd'].mean():.2f} ± {df['pKd'].std():.2f}")
print(f"Unique antigens:    {df['Ag_name'].nunique():,}")
print(f"Unique antibodies:  {df['Ab_name'].nunique():,}")

# Sequence length stats
for col, name in [("Ab_heavy_chain_seq", "Heavy"), ("Ab_light_chain_seq", "Light"), ("Ag_seq", "Antigen")]:
    lengths = df[col].str.len()
    print(f"{name} chain lengths: mean={lengths.mean():.0f}, max={lengths.max()}, p95={lengths.quantile(0.95):.0f}")

# %% [markdown]
# ## §2 — Antigen Clustering (MMseqs2, 40% Identity)

# %%
# Extract unique antigen sequences to FASTA
unique_ags = df[["Ag_name", "Ag_seq"]].drop_duplicates(subset="Ag_name")
print(f"Unique antigen sequences for clustering: {len(unique_ags)}")

fasta_file = DATA_DIR / "antigens.fasta"
with open(fasta_file, "w") as f:
    for _, row in unique_ags.iterrows():
        f.write(f">{row['Ag_name']}\n{row['Ag_seq']}\n")
print(f"✓ Wrote {fasta_file}")

# %%
# Run MMseqs2 clustering at 40% sequence identity
cluster_prefix = str(DATA_DIR / "ag_clusters")
tmp_dir = str(DATA_DIR / "mmseqs_tmp")
os.makedirs(tmp_dir, exist_ok=True)

import shutil

# Attempt to locate mmseqs binary in PATH or Conda env
mmseqs_path = shutil.which("mmseqs")
if mmseqs_path is None:
    # Explicitly check the Conda environment's bin folder
    mmseqs_path = os.path.join(sys.prefix, "bin", "mmseqs")
    if not os.path.exists(mmseqs_path):
        raise FileNotFoundError(
            f"mmseqs not found in PATH or at {mmseqs_path}. "
            "Please ensure you installed it via `conda install bioconda::mmseqs2`"
        )

cmd = [
    mmseqs_path, "easy-cluster",
    str(fasta_file),
    cluster_prefix,
    tmp_dir,
    "--min-seq-id", "0.4",
    "-c", "0.8",
    "--cov-mode", "0",
    "--threads", "8",
]
print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"⚠ MMseqs2 stderr:\n{result.stderr}")
    raise RuntimeError("MMseqs2 clustering failed")
print("✓ MMseqs2 clustering complete")

# %%
# Parse cluster results
cluster_tsv = f"{cluster_prefix}_cluster.tsv"
cluster_df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["representative", "member"])

# Assign integer cluster IDs
ag_to_cluster = {}
for cluster_id, (rep, group) in enumerate(cluster_df.groupby("representative")):
    for member in group["member"]:
        ag_to_cluster[member] = cluster_id

df["ag_cluster_40"] = df["Ag_name"].map(ag_to_cluster)

# Check for unmapped antigens
n_unmapped = df["ag_cluster_40"].isna().sum()
if n_unmapped > 0:
    print(f"⚠ {n_unmapped} antigens not found in cluster output — assigning to singleton clusters")
    max_cluster = df["ag_cluster_40"].max()
    unmapped_mask = df["ag_cluster_40"].isna()
    df.loc[unmapped_mask, "ag_cluster_40"] = range(int(max_cluster) + 1, int(max_cluster) + 1 + n_unmapped)

df["ag_cluster_40"] = df["ag_cluster_40"].astype(int)

# %%
# --- Distribution Analysis ---
cluster_sample_counts = df.groupby("ag_cluster_40").size().sort_values(ascending=False)
n_clusters = len(cluster_sample_counts)
n_singletons = (cluster_sample_counts == 1).sum()
imbalance_ratio = cluster_sample_counts.max() / cluster_sample_counts.median()

print(f"\n--- Antigen Clustering Report (40% Identity) ---")
print(f"Total clusters:       {n_clusters}")
print(f"Singleton clusters:   {n_singletons}")
print(f"Largest cluster:      {cluster_sample_counts.max():,} samples")
print(f"Smallest cluster:     {cluster_sample_counts.min():,} samples")
print(f"Median cluster size:  {cluster_sample_counts.median():.0f} samples")
print(f"Imbalance ratio:      {imbalance_ratio:.1f}x")

# Flag dominant clusters (>15% of total)
total_samples = len(df)
dominant_threshold = 0.15 * total_samples
dominant = cluster_sample_counts[cluster_sample_counts > dominant_threshold]
if len(dominant) > 0:
    print(f"\n🚨 DOMINANT CLUSTERS (>{dominant_threshold:.0f} samples, >15% of dataset):")
    for cid, count in dominant.items():
        print(f"   Cluster {cid}: {count:,} samples ({100*count/total_samples:.1f}%)")
else:
    print("✓ No single cluster exceeds 15% of the dataset")

# %%
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Top-20 clusters
cluster_sample_counts.head(20).plot.bar(ax=axes[0], color='steelblue')
axes[0].set_title("Top-20 Antigen Clusters by Sample Count")
axes[0].set_xlabel("Cluster ID")
axes[0].set_ylabel("Sample Count")

# Cluster size distribution (log scale)
axes[1].hist(cluster_sample_counts.values, bins=50, color='steelblue', edgecolor='white')
axes[1].set_yscale('log')
axes[1].set_title("Distribution of Cluster Sizes")
axes[1].set_xlabel("Samples per Cluster")
axes[1].set_ylabel("Number of Clusters (log)")
plt.tight_layout()
plt.savefig("cluster_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## §3 — Bias Analysis & Group-Aware Splits

# %%
MIN_CLUSTER_SAMPLES = 5  # threshold for flagging small clusters

small_clusters = cluster_sample_counts[cluster_sample_counts < MIN_CLUSTER_SAMPLES]
print(f"Clusters with < {MIN_CLUSTER_SAMPLES} samples: {len(small_clusters)}")
print("Decision: KEEP them (rare antigen families are valuable for generalization testing)")

# %%
# --- Group-aware train/val/test split ---
# No antigen family may appear in more than one split.

cluster_counts = df.groupby("ag_cluster_40").size().sort_values(ascending=False)
clusters = cluster_counts.index.tolist()

total = len(df)
target_train = 0.80 * total
target_val   = 0.10 * total
target_test  = 0.10 * total

test_clusters, val_clusters, train_clusters = [], [], []
test_count, val_count, train_count = 0, 0, 0

for c in clusters:
    count = cluster_counts[c]
    
    # Calculate how far each split is from its target relative to its target size
    scores = [
        ((target_train - train_count) / target_train, 0, 'train'),
        ((target_val - val_count) / target_val, 1, 'val'),
        ((target_test - test_count) / target_test, 2, 'test')
    ]
    # Sort descending by score, tie-break by priority (0=train, 1=val, 2=test)
    scores.sort(reverse=True, key=lambda x: (x[0], -x[1]))
    best_split = scores[0][2]
    
    if best_split == 'train':
        train_clusters.append(c)
        train_count += count
    elif best_split == 'val':
        val_clusters.append(c)
        val_count += count
    else:
        test_clusters.append(c)
        test_count += count

train_df = df[df["ag_cluster_40"].isin(train_clusters)].copy()
val_df   = df[df["ag_cluster_40"].isin(val_clusters)].copy()
test_df  = df[df["ag_cluster_40"].isin(test_clusters)].copy()

# Lock test families
TEST_FAMILIES = set(test_clusters)
VAL_FAMILIES = set(val_clusters)

print(f"\n--- Split Summary ---")
for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    n_fam = split_df["ag_cluster_40"].nunique()
    print(f"{name:5s}: {len(split_df):>7,} samples ({100*len(split_df)/total:5.1f}%), "
          f"{n_fam} antigen families, "
          f"pKd mean={split_df['pKd'].mean():.2f}±{split_df['pKd'].std():.2f}")

# %%
# Verify no leakage
train_fams = set(train_df["ag_cluster_40"].unique())
val_fams   = set(val_df["ag_cluster_40"].unique())
test_fams  = set(test_df["ag_cluster_40"].unique())

assert len(train_fams & test_fams) == 0, "❌ Leakage: train ∩ test"
assert len(train_fams & val_fams) == 0,  "❌ Leakage: train ∩ val"
assert len(val_fams & test_fams) == 0,   "❌ Leakage: val ∩ test"
print("✓ No antigen family leakage between splits")

# %%
# Per-cluster sample counts after splitting (visible, not hidden)
print("\n--- Per-Cluster Sample Counts (Train) ---")
train_cluster_counts = train_df.groupby("ag_cluster_40").size().sort_values(ascending=False)
print(train_cluster_counts.describe())

print("\n--- Per-Cluster Sample Counts (Val) ---")
val_cluster_counts = val_df.groupby("ag_cluster_40").size().sort_values(ascending=False)
print(val_cluster_counts.describe())

print("\n--- Per-Cluster Sample Counts (Test) ---")
test_cluster_counts = test_df.groupby("ag_cluster_40").size().sort_values(ascending=False)
print(test_cluster_counts.describe())

# %%
# --- WeightedRandomSampler for training ---
from torch.utils.data import WeightedRandomSampler

train_cluster_freq = train_df["ag_cluster_40"].value_counts().to_dict()
train_weights = train_df["ag_cluster_40"].map(lambda c: 1.0 / train_cluster_freq[c]).values
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(train_weights),
    num_samples=len(train_weights),
    replacement=True,
)
print(f"✓ WeightedRandomSampler: {len(train_weights):,} samples, upweighting {len(train_cluster_freq)} families")

# %% [markdown]
# ## §4 — Model: Dual Encoder + Projection + MLP Head

# %%
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer
import torch.nn as nn
import torch.nn.functional as F

class DualEncoderModel(nn.Module):
    """
    Dual-encoder model for antibody-antigen pKd prediction.

    Architecture:
      Heavy chain  → AntiBERTa2-CSSP (frozen) → mean pool → (768,)  ─┐
                                                                       ├─ concat → (1536,) → proj → (256,)
      Light chain  → AntiBERTa2-CSSP (frozen) → mean pool → (768,)  ─┘
      Antigen      → ESM-2 150M (frozen)      → mean pool → (640,)    → proj → (256,)

      Concat(ab_proj, ag_proj) = (512,) → MLP → scalar pKd
    """

    def __init__(self, ab_model_name="alchemab/antiberta2-cssp",
                 ag_model_name="facebook/esm2_t30_150M_UR50D",
                 proj_dim=256, mlp_hidden=None, dropout=0.1):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [512, 256, 128]

        # --- Frozen antibody encoder (shared for H and L) ---
        self.ab_encoder = AutoModel.from_pretrained(ab_model_name)
        for p in self.ab_encoder.parameters():
            p.requires_grad = False
        self.ab_encoder.eval()
        ab_hidden = self.ab_encoder.config.hidden_size  # 768

        # --- Frozen antigen encoder ---
        self.ag_encoder = EsmModel.from_pretrained(ag_model_name)
        for p in self.ag_encoder.parameters():
            p.requires_grad = False
        self.ag_encoder.eval()
        ag_hidden = self.ag_encoder.config.hidden_size  # 640

        # --- Learned projection layers → 256-d shared space ---
        self.ab_proj = nn.Sequential(
            nn.Linear(ab_hidden * 2, proj_dim),  # 1536 → 256
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )
        self.ag_proj = nn.Sequential(
            nn.Linear(ag_hidden, proj_dim),  # 640 → 256
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )

        # --- MLP Head ---
        layers = []
        in_dim = proj_dim * 2  # 512
        for h in mlp_hidden:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp_head = nn.Sequential(*layers)

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, heavy_input_ids, heavy_attention_mask,
                light_input_ids, light_attention_mask,
                ag_input_ids, ag_attention_mask):
        # Frozen encoders — no gradient
        with torch.no_grad():
            h_out = self.ab_encoder(input_ids=heavy_input_ids,
                                    attention_mask=heavy_attention_mask).last_hidden_state
            l_out = self.ab_encoder(input_ids=light_input_ids,
                                    attention_mask=light_attention_mask).last_hidden_state
            ag_out = self.ag_encoder(input_ids=ag_input_ids,
                                     attention_mask=ag_attention_mask).last_hidden_state

        # Mean pooling
        h_pooled = self.mean_pool(h_out, heavy_attention_mask)   # (B, 768)
        l_pooled = self.mean_pool(l_out, light_attention_mask)   # (B, 768)
        ab_pooled = torch.cat([h_pooled, l_pooled], dim=-1)      # (B, 1536)
        ag_pooled = self.mean_pool(ag_out, ag_attention_mask)     # (B, 640)

        # Projection
        ab_proj = self.ab_proj(ab_pooled)  # (B, 256)
        ag_proj = self.ag_proj(ag_pooled)  # (B, 256)

        # Fusion + regression
        fused = torch.cat([ab_proj, ag_proj], dim=-1)  # (B, 512)
        return self.mlp_head(fused).squeeze(-1)  # (B,)


# %%
# Load tokenizers
print("Loading tokenizers...")
ab_tokenizer = AutoTokenizer.from_pretrained("alchemab/antiberta2-cssp")
ag_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
print("✓ Tokenizers loaded")

# %% [markdown]
# ## §5 — Create DataLoaders

# %%
from dataset import AbAgAffinityDataset
from torch.utils.data import DataLoader

train_dataset = AbAgAffinityDataset(train_df, ab_tokenizer, ag_tokenizer, max_ab_len=256, max_ag_len=512)
val_dataset   = AbAgAffinityDataset(val_df,   ab_tokenizer, ag_tokenizer, max_ab_len=256, max_ag_len=512)
test_dataset  = AbAgAffinityDataset(test_df,  ab_tokenizer, ag_tokenizer, max_ab_len=256, max_ag_len=512)

train_loader = DataLoader(
    train_dataset, batch_size=32, sampler=sampler,
    num_workers=4, pin_memory=True, persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True,
)

print(f"Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
print(f"Val:   {len(val_dataset):,} samples, {len(val_loader)} batches")
print(f"Test:  {len(test_dataset):,} samples, {len(test_loader)} batches")

# %% [markdown]
# ## §6 — NN Baseline (Mandatory)

# %%
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


def compute_embeddings(model, loader, device):
    """Extract mean-pooled embeddings from frozen encoders for NN baseline."""
    all_ab_embs, all_ag_embs, all_targets, all_clusters = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing embeddings"):
            h_ids = batch["heavy_input_ids"].to(device)
            h_mask = batch["heavy_attention_mask"].to(device)
            l_ids = batch["light_input_ids"].to(device)
            l_mask = batch["light_attention_mask"].to(device)
            ag_ids = batch["ag_input_ids"].to(device)
            ag_mask = batch["ag_attention_mask"].to(device)

            h_out = model.ab_encoder(input_ids=h_ids, attention_mask=h_mask).last_hidden_state
            l_out = model.ab_encoder(input_ids=l_ids, attention_mask=l_mask).last_hidden_state
            ag_out = model.ag_encoder(input_ids=ag_ids, attention_mask=ag_mask).last_hidden_state

            h_pool = model.mean_pool(h_out, h_mask)
            l_pool = model.mean_pool(l_out, l_mask)
            ag_pool = model.mean_pool(ag_out, ag_mask)

            ab_emb = torch.cat([h_pool, l_pool], dim=-1).cpu().numpy()
            ag_emb = ag_pool.cpu().numpy()

            # Concatenate ab+ag for similarity
            combined = np.concatenate([ab_emb, ag_emb], axis=1)
            all_ab_embs.append(combined)
            all_targets.append(batch["target"].numpy())
            all_clusters.append(batch["cluster_id"].numpy())

    return (np.concatenate(all_ab_embs),
            np.concatenate(all_targets),
            np.concatenate(all_clusters))


# This cell runs AFTER model instantiation (§7 preamble)
# See "NN Baseline Execution" cell below

# %% [markdown]
# ## §7 — Training (Fault-Tolerant, AMP, CSV Logging)

# %%
# --- Instantiate model ---
device = torch.device("cuda")
model = DualEncoderModel().to(device)

# Only projection + MLP params are trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
frozen_params = total_params - trainable_params

# %%
# --- Pre-Training Summary ---
print("=" * 65)
print("              PRE-TRAINING SUMMARY")
print("=" * 65)
print(f"  Dataset:")
print(f"    Train:  {len(train_df):>8,} samples, {train_df['ag_cluster_40'].nunique():>4} antigen families")
print(f"    Val:    {len(val_df):>8,} samples, {val_df['ag_cluster_40'].nunique():>4} antigen families")
print(f"    Test:   {len(test_df):>8,} samples, {test_df['ag_cluster_40'].nunique():>4} antigen families")
print(f"\n  Cluster distribution (40% identity):")
print(f"    Total clusters:   {n_clusters}")
print(f"    Largest cluster:  {cluster_sample_counts.max():,} samples")
print(f"    Smallest cluster: {cluster_sample_counts.min():,} samples")
print(f"    Median cluster:   {int(cluster_sample_counts.median()):,} samples")
print(f"    Singletons:       {n_singletons}")
print(f"\n  Model parameters:")
print(f"    Total:     {total_params:>12,}")
print(f"    Trainable: {trainable_params:>12,} ({100*trainable_params/total_params:.2f}%)")
print(f"    Frozen:    {frozen_params:>12,}")
print(f"\n  Hardware:    {torch.cuda.get_device_name(0)}")
print(f"  Precision:   torch.cuda.amp (float16)")
print(f"  Optimizer:   AdamW, lr=1e-4, weight_decay=1e-4")
print("=" * 65)

# %%
# --- NN Baseline Execution ---
print("\n--- Computing NN Baseline ---")
train_embs, train_targets_np, train_clusters_np = compute_embeddings(model, train_loader, device)
test_embs, test_targets_np, test_clusters_np = compute_embeddings(model, test_loader, device)

# For each test sample, find most similar training sample
print("Computing cosine similarities...")
# Process in chunks to avoid OOM
CHUNK = 1000
nn_preds = np.zeros(len(test_embs))
for i in tqdm(range(0, len(test_embs), CHUNK), desc="NN search"):
    chunk = test_embs[i:i+CHUNK]
    sims = cosine_similarity(chunk, train_embs)
    nn_idx = sims.argmax(axis=1)
    nn_preds[i:i+CHUNK] = train_targets_np[nn_idx]

nn_rho, _ = spearmanr(test_targets_np, nn_preds)
print(f"🔑 NN Baseline Spearman ρ: {nn_rho:.4f}")
print("   (Model MUST beat this to be considered successful)")

# %%
# --- Training setup ---
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY,
)
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True,
)

# %%
# --- Checkpoint management ---

def save_checkpoint(model, optimizer, scaler, epoch, metric, ckpt_dir=CKPT_DIR, max_ckpts=2):
    ckpt_path = ckpt_dir / f"ckpt_epoch{epoch:03d}_spearman{metric:.4f}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_metric": metric,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }, ckpt_path)
    # Enforce max checkpoints on disk
    existing = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(existing) > max_ckpts:
        existing[0].unlink()
        existing.pop(0)
    print(f"  💾 Saved checkpoint: {ckpt_path.name}")


def load_latest_checkpoint(model, optimizer, scaler, ckpt_dir=CKPT_DIR):
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        print("  No checkpoint found — starting from scratch")
        return 0, -float("inf")
    ckpt_path = ckpts[-1]
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
    print(f"  ✓ Resumed from {ckpt_path.name} (epoch {ckpt['epoch']}, "
          f"best Spearman: {ckpt['best_metric']:.4f})")
    return ckpt["epoch"], ckpt["best_metric"]

# %%
# --- Evaluation function ---
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lifelines.utils import concordance_index


def evaluate(model, loader, device):
    """Evaluate on a DataLoader, return dict of all metrics."""
    model.eval()
    all_preds, all_targets, all_clusters = [], [], []
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast():
                preds = model(
                    heavy_input_ids=batch["heavy_input_ids"].to(device),
                    heavy_attention_mask=batch["heavy_attention_mask"].to(device),
                    light_input_ids=batch["light_input_ids"].to(device),
                    light_attention_mask=batch["light_attention_mask"].to(device),
                    ag_input_ids=batch["ag_input_ids"].to(device),
                    ag_attention_mask=batch["ag_attention_mask"].to(device),
                )
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["target"].numpy())
            all_clusters.append(batch["cluster_id"].numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    clusters = np.concatenate(all_clusters)

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    pearson_r, _ = pearsonr(targets, preds)
    spearman_rho, _ = spearmanr(targets, preds)
    kendall, _ = kendalltau(targets, preds)
    ci = concordance_index(targets, preds)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson_r,
        "spearman": spearman_rho,
        "kendall_tau": kendall,
        "ci": ci,
    }, preds, targets, clusters

# %%
# --- Training loop ---
start_epoch, best_spearman = load_latest_checkpoint(model, optimizer, scaler)

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    # Keep frozen encoders in eval mode
    model.ab_encoder.eval()
    model.ag_encoder.eval()

    epoch_loss = 0.0
    n_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            preds = model(
                heavy_input_ids=batch["heavy_input_ids"].to(device),
                heavy_attention_mask=batch["heavy_attention_mask"].to(device),
                light_input_ids=batch["light_input_ids"].to(device),
                light_attention_mask=batch["light_attention_mask"].to(device),
                ag_input_ids=batch["ag_input_ids"].to(device),
                ag_attention_mask=batch["ag_attention_mask"].to(device),
            )
            loss = F.mse_loss(preds, batch["target"].to(device))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = epoch_loss / n_batches

    # --- Validation ---
    val_metrics, _, _, _ = evaluate(model, val_loader, device)
    scheduler.step(val_metrics["spearman"])

    # --- Log to CSV ---
    log_row = {"epoch": epoch + 1, "train_loss": avg_train_loss, **val_metrics}
    write_header = not LOG_FILE.exists() or epoch == start_epoch
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in log_row.items()})

    # --- Print epoch summary ---
    print(f"  Epoch {epoch+1}: loss={avg_train_loss:.4f} | "
          f"val_RMSE={val_metrics['rmse']:.4f} | "
          f"val_Pearson={val_metrics['pearson']:.4f} | "
          f"val_Spearman={val_metrics['spearman']:.4f} | "
          f"val_CI={val_metrics['ci']:.4f}")

    # --- Checkpoint if improved ---
    if val_metrics["spearman"] > best_spearman:
        best_spearman = val_metrics["spearman"]
        save_checkpoint(model, optimizer, scaler, epoch + 1, best_spearman)

print(f"\n✓ Training complete. Best validation Spearman ρ: {best_spearman:.4f}")

# %% [markdown]
# ## §8 — Full Test Evaluation & Per-Family Breakdown

# %%
# Load best checkpoint for final evaluation
_, _ = load_latest_checkpoint(model, optimizer, scaler)

test_metrics, test_preds, test_targets, test_clusters = evaluate(model, test_loader, device)

print("\n" + "=" * 65)
print("              FINAL TEST METRICS")
print("=" * 65)
for k, v in test_metrics.items():
    print(f"  {k:15s}: {v:.4f}")

# Compare with NN baseline
print(f"\n  NN Baseline Spearman ρ:  {nn_rho:.4f}")
print(f"  Model Spearman ρ:       {test_metrics['spearman']:.4f}")
if test_metrics["spearman"] > nn_rho:
    print("  ✓ Model beats NN baseline")
else:
    print("  🚨 FAILURE: Model does NOT beat NN baseline on Spearman ρ!")
    print("     Investigate: model may be underfitting or data leakage in NN baseline")
print("=" * 65)

# %%
# --- Per-family breakdown ---
def per_family_metrics(predictions, targets, cluster_ids):
    results = {}
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        if mask.sum() < 2:
            continue
        y_true = targets[mask]
        y_pred = predictions[mask]
        rho_val, _ = spearmanr(y_true, y_pred)
        rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
        results[int(c)] = {"spearman": rho_val, "rmse": rmse_val, "n_samples": int(mask.sum())}

    valid = {k: v for k, v in results.items() if not np.isnan(v["spearman"])}
    spear_vals = [v["spearman"] for v in valid.values()]
    weights = [v["n_samples"] for v in valid.values()]
    macro_spearman = np.mean(spear_vals) if spear_vals else 0.0
    weighted_spearman = np.average(spear_vals, weights=weights) if spear_vals else 0.0
    macro_rmse = np.mean([v["rmse"] for v in valid.values()]) if valid else 0.0
    weighted_rmse = np.average([v["rmse"] for v in valid.values()], weights=weights) if valid else 0.0

    return results, {
        "macro_spearman": macro_spearman,
        "weighted_spearman": weighted_spearman,
        "macro_rmse": macro_rmse,
        "weighted_rmse": weighted_rmse,
    }


family_results, agg_metrics = per_family_metrics(test_preds, test_targets, test_clusters)

print("\n--- Per-Family Aggregates ---")
for k, v in agg_metrics.items():
    print(f"  {k:25s}: {v:.4f}")

print(f"\n--- Top-10 Families by Sample Count ---")
sorted_fams = sorted(family_results.items(), key=lambda x: x[1]["n_samples"], reverse=True)[:10]
print(f"  {'Family':>8s}  {'Samples':>8s}  {'Spearman':>10s}  {'RMSE':>8s}")
for fam_id, fam_data in sorted_fams:
    print(f"  {fam_id:>8d}  {fam_data['n_samples']:>8d}  {fam_data['spearman']:>10.4f}  {fam_data['rmse']:>8.4f}")

# %% [markdown]
# ## §9 — Calibration Plots

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Predicted vs Actual ---
axes[0].scatter(test_targets, test_preds, alpha=0.2, s=8, color='steelblue')
lims = [min(test_targets.min(), test_preds.min()) - 0.5,
        max(test_targets.max(), test_preds.max()) + 0.5]
axes[0].plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y=x')
m, b = np.polyfit(test_targets, test_preds, 1)
x_line = np.linspace(lims[0], lims[1], 100)
axes[0].plot(x_line, m * x_line + b, 'r-', linewidth=2, label=f'Fit: y={m:.2f}x+{b:.2f}')
axes[0].set_xlabel("Actual pKd", fontsize=12)
axes[0].set_ylabel("Predicted pKd", fontsize=12)
axes[0].set_title(f"Predicted vs Actual (Pearson r={test_metrics['pearson']:.3f}, R²={test_metrics['r2']:.3f})",
                  fontsize=13)
axes[0].legend()
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)

# --- Plot 2: Residuals ---
residuals = test_preds - test_targets
axes[1].scatter(test_targets, residuals, alpha=0.2, s=8, color='coral')
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[1].set_xlabel("Actual pKd", fontsize=12)
axes[1].set_ylabel("Residual (Pred − Actual)", fontsize=12)
axes[1].set_title(f"Residual Plot (MAE={test_metrics['mae']:.3f})", fontsize=13)

plt.tight_layout()
plt.savefig("calibration_plots.png", dpi=200, bbox_inches='tight')
plt.show()
print("✓ Saved calibration_plots.png")

# %% [markdown]
# ## §10 — Training History Visualization

# %%
if LOG_FILE.exists():
    log_df = pd.read_csv(LOG_FILE)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(log_df["epoch"], log_df["train_loss"], 'b-', label="Train Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(log_df["epoch"], log_df["rmse"], 'r-', label="Val RMSE")
    axes[0, 1].set_title("Validation RMSE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(log_df["epoch"], log_df["pearson"], 'g-', label="Val Pearson r")
    axes[1, 0].plot(log_df["epoch"], log_df["spearman"], 'm-', label="Val Spearman ρ")
    axes[1, 0].set_title("Correlation Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    axes[1, 1].plot(log_df["epoch"], log_df["ci"], 'c-', label="Val CI")
    axes[1, 1].set_title("Concordance Index")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved training_history.png")
