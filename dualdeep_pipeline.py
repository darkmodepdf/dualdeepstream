# %% [markdown]
# # DuaDeep-SeqAffinity: Antibody-Antigen pKd Prediction Pipeline
#
# **Architecture:** Dual-stream (Transformer + CNN) on AntiBERTa2-CSSP (H+L) + ESM-2 150M (Ag)
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
import os, sys, re, csv, glob, subprocess, json, joblib
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
# ## §3 — Bias Analysis & Group-Aware Splits + pKd Standardization

# %%
MIN_CLUSTER_SAMPLES = 5

small_clusters = cluster_sample_counts[cluster_sample_counts < MIN_CLUSTER_SAMPLES]
print(f"Clusters with < {MIN_CLUSTER_SAMPLES} samples: {len(small_clusters)}")
print("Decision: KEEP them (rare antigen families are valuable for generalization testing)")

# %%
# --- Group-aware train/val/test split ---
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
    scores = [
        ((target_train - train_count) / target_train, 0, 'train'),
        ((target_val - val_count) / target_val, 1, 'val'),
        ((target_test - test_count) / target_test, 2, 'test')
    ]
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

# Verify no leakage
train_fams = set(train_df["ag_cluster_40"].unique())
val_fams   = set(val_df["ag_cluster_40"].unique())
test_fams  = set(test_df["ag_cluster_40"].unique())

assert len(train_fams & test_fams) == 0, "❌ Leakage: train ∩ test"
assert len(train_fams & val_fams) == 0,  "❌ Leakage: train ∩ val"
assert len(val_fams & test_fams) == 0,   "❌ Leakage: val ∩ test"
print("✓ No antigen family leakage between splits")

# %%
# --- pKd z-score standardization (paper §4.1) ---
from sklearn.preprocessing import StandardScaler

pKd_scaler = StandardScaler()
train_df["pKd_scaled"] = pKd_scaler.fit_transform(train_df[["pKd"]]).flatten()
val_df["pKd_scaled"]   = pKd_scaler.transform(val_df[["pKd"]]).flatten()
test_df["pKd_scaled"]  = pKd_scaler.transform(test_df[["pKd"]]).flatten()

scaler_path = DATA_DIR / "pKd_scaler.pkl"
joblib.dump(pKd_scaler, scaler_path)
print(f"✓ pKd StandardScaler: mean={pKd_scaler.mean_[0]:.4f}, std={pKd_scaler.scale_[0]:.4f}")

# %%
# Print split summary
print(f"\n--- Split Summary ---")
for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    n_fam = split_df["ag_cluster_40"].nunique()
    print(f"{name:5s}: {len(split_df):>7,} samples ({100*len(split_df)/total:5.1f}%), "
          f"{n_fam} antigen families, "
          f"pKd mean={split_df['pKd'].mean():.2f}±{split_df['pKd'].std():.2f}")

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
# ## §4 — Model: Dual-Stream (Transformer + CNN) Architecture

# %%
import torch.nn as nn
import torch.nn.functional as F
from dualdeep_model import DualEncoderModel

# %%
# Load tokenizers
print("Loading tokenizers...")
ab_tokenizer = AutoTokenizer.from_pretrained("alchemab/antiberta2-cssp")
ag_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
print("✓ Tokenizers loaded")

# %% [markdown]
# ## §5 — Create DataLoaders

# %%
from dualdeep_dataset import AbAgAffinityDataset
from torch.utils.data import DataLoader

train_dataset = AbAgAffinityDataset(train_df, ab_tokenizer, ag_tokenizer, target_col="pKd_scaled")
val_dataset   = AbAgAffinityDataset(val_df,   ab_tokenizer, ag_tokenizer, target_col="pKd_scaled")
test_dataset  = AbAgAffinityDataset(test_df,  ab_tokenizer, ag_tokenizer, target_col="pKd_scaled")

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
from dualdeep_utils import evaluate, save_checkpoint, load_latest_checkpoint


def compute_embeddings(model, loader, device):
    """Extract mean-pooled embeddings from frozen encoders for NN baseline."""
    all_embs, all_targets, all_clusters = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing embeddings", mininterval=2.0):
            h_ids = batch["heavy_input_ids"].to(device)
            h_mask = batch["heavy_attention_mask"].to(device)
            l_ids = batch["light_input_ids"].to(device)
            l_mask = batch["light_attention_mask"].to(device)
            ag_ids = batch["ag_input_ids"].to(device)
            ag_mask = batch["ag_attention_mask"].to(device)

            with torch.cuda.amp.autocast():
                h_out = model.ab_encoder(input_ids=h_ids, attention_mask=h_mask).last_hidden_state
                l_out = model.ab_encoder(input_ids=l_ids, attention_mask=l_mask).last_hidden_state
                ag_out = model.ag_encoder(input_ids=ag_ids, attention_mask=ag_mask).last_hidden_state

                h_pool = model.mean_pool(h_out, h_mask)
                l_pool = model.mean_pool(l_out, l_mask)
                ag_pool = model.mean_pool(ag_out, ag_mask)

            ab_emb = torch.cat([h_pool, l_pool], dim=-1).cpu().numpy()
            ag_emb = ag_pool.cpu().numpy()

            combined = np.concatenate([ab_emb, ag_emb], axis=1)
            all_embs.append(combined)
            all_targets.append(batch["target"].numpy())
            all_clusters.append(batch["cluster_id"].numpy())

    return (np.concatenate(all_embs),
            np.concatenate(all_targets),
            np.concatenate(all_clusters))


# %% [markdown]
# ## §7 — Training (Fault-Tolerant, AMP, Gradient Clipping, CSV Logging)

# %%
# --- Instantiate model ---
device = torch.device("cuda")
model = DualEncoderModel().to(device)

# Only dual-stream + projection + MLP params are trainable
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
print(f"\n  pKd normalization:")
print(f"    Scaler mean:  {pKd_scaler.mean_[0]:.4f}")
print(f"    Scaler scale: {pKd_scaler.scale_[0]:.4f}")
print(f"\n  Model parameters:")
print(f"    Total:     {total_params:>12,}")
print(f"    Trainable: {trainable_params:>12,} ({100*trainable_params/total_params:.2f}%)")
print(f"    Frozen:    {frozen_params:>12,}")
print(f"\n  Architecture: Dual-Stream (Transformer + CNN)")
print(f"    Ab streams: Heavy (768+128=896) + Light (768+128=896)")
print(f"    Ag stream:  640+128=768")
print(f"    Fused dim:  2560 → MLP [512, 256, 128] → 1")
print(f"\n  Hardware:    {torch.cuda.get_device_name(0)}")
print(f"  Precision:   torch.cuda.amp (float16)")
print(f"  Optimizer:   AdamW, lr=1e-4, weight_decay=1e-4, grad_clip=1.0")
print("=" * 65)

# %%
# --- NN Baseline Execution ---
print("\n--- Computing NN Baseline ---")
train_embs, train_targets_scaled, train_clusters_np = compute_embeddings(model, train_loader, device)
test_embs, test_targets_scaled, test_clusters_np = compute_embeddings(model, test_loader, device)

# Inverse-transform for NN baseline comparison on original pKd scale
train_targets_raw = pKd_scaler.inverse_transform(train_targets_scaled.reshape(-1, 1)).flatten()
test_targets_raw = pKd_scaler.inverse_transform(test_targets_scaled.reshape(-1, 1)).flatten()

print("Computing cosine similarities...")
CHUNK = 1000
nn_preds = np.zeros(len(test_embs))
for i in tqdm(range(0, len(test_embs), CHUNK), desc="NN search", mininterval=2.0):
    chunk = test_embs[i:i+CHUNK]
    sims = cosine_similarity(chunk, train_embs)
    nn_idx = sims.argmax(axis=1)
    nn_preds[i:i+CHUNK] = train_targets_raw[nn_idx]

nn_rho, _ = spearmanr(test_targets_raw, nn_preds)
print(f"🔑 NN Baseline Spearman ρ: {nn_rho:.4f}")
print("   (Model MUST beat this to be considered successful)")

# %%
# --- Training setup ---
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY,
)
amp_scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True,
)

# %%
# --- Checkpoint management (imported from dualdeep_utils) ---

# %%
# --- Training loop ---
start_epoch, best_spearman = load_latest_checkpoint(model, optimizer, amp_scaler)

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    # Keep frozen encoders in eval mode
    model.ab_encoder.eval()
    model.ag_encoder.eval()

    epoch_loss = 0.0
    n_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", mininterval=2.0)
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

        amp_scaler.scale(loss).backward()

        # Gradient clipping for stability
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=MAX_GRAD_NORM,
        )

        amp_scaler.step(optimizer)
        amp_scaler.update()

        epoch_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = epoch_loss / n_batches

    # --- Validation (metrics on original pKd scale) ---
    val_metrics, _, _, _ = evaluate(model, val_loader, device, scaler=pKd_scaler)
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
        save_checkpoint(model, optimizer, amp_scaler, epoch + 1, best_spearman, CKPT_DIR)

print(f"\n✓ Training complete. Best validation Spearman ρ: {best_spearman:.4f}")

# %% [markdown]
# ## §8 — Full Test Evaluation & Per-Family Breakdown

# %%
# Load best checkpoint for final evaluation
_, _ = load_latest_checkpoint(model, optimizer, amp_scaler)

test_metrics, test_preds, test_targets, test_clusters = evaluate(
    model, test_loader, device, scaler=pKd_scaler
)

print("\n" + "=" * 65)
print("              FINAL TEST METRICS (original pKd scale)")
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
