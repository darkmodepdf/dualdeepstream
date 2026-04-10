import pandas as pd
import numpy as np
import os, sys, shutil, subprocess
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    print("=== Step 2: Antigen Clustering ===")
    DATA_DIR = Path(".")
    INPUT_FILE = DATA_DIR / "step1_cleaned_dataset.csv"
    OUTPUT_FILE = DATA_DIR / "step2_clustered_dataset.csv"

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run step 1 first.")

    df = pd.read_csv(INPUT_FILE)

    unique_ags = df[["Ag_name", "Ag_seq"]].drop_duplicates(subset="Ag_name")
    print(f"Unique antigen sequences for clustering: {len(unique_ags)}")

    fasta_file = DATA_DIR / "antigens.fasta"
    with open(fasta_file, "w") as f:
        for _, row in unique_ags.iterrows():
            f.write(f">{row['Ag_name']}\n{row['Ag_seq']}\n")
    print(f"✓ Wrote {fasta_file}")

    cluster_prefix = str(DATA_DIR / "ag_clusters")
    tmp_dir = str(DATA_DIR / "mmseqs_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    mmseqs_path = shutil.which("mmseqs")
    if mmseqs_path is None:
        mmseqs_path = os.path.join(sys.prefix, "bin", "mmseqs")
        if not os.path.exists(mmseqs_path):
            raise FileNotFoundError("mmseqs not found in PATH")

    cmd = [
        mmseqs_path, "easy-cluster",
        str(fasta_file), cluster_prefix, tmp_dir,
        "--min-seq-id", "0.4", "-c", "0.8", "--cov-mode", "0", "--threads", "8"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("MMseqs2 clustering failed\n" + result.stderr)

    cluster_tsv = f"{cluster_prefix}_cluster.tsv"
    cluster_df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["representative", "member"])

    ag_to_cluster = {}
    for cluster_id, (rep, group) in enumerate(cluster_df.groupby("representative")):
        for member in group["member"]:
            ag_to_cluster[member] = cluster_id

    df["ag_cluster_40"] = df["Ag_name"].map(ag_to_cluster)

    n_unmapped = df["ag_cluster_40"].isna().sum()
    if n_unmapped > 0:
        max_cluster = df["ag_cluster_40"].max()
        unmapped_mask = df["ag_cluster_40"].isna()
        df.loc[unmapped_mask, "ag_cluster_40"] = range(int(max_cluster) + 1, int(max_cluster) + 1 + n_unmapped)

    df["ag_cluster_40"] = df["ag_cluster_40"].astype(int)

    # Visualization
    cluster_sample_counts = df.groupby("ag_cluster_40").size().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    cluster_sample_counts.head(20).plot.bar(ax=axes[0], color='steelblue')
    axes[0].set_title("Top-20 Antigen Clusters by Sample Count")
    axes[1].hist(cluster_sample_counts.values, bins=50, color='steelblue', edgecolor='white')
    axes[1].set_yscale('log')
    axes[1].set_title("Distribution of Cluster Sizes")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "cluster_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved clustered data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
