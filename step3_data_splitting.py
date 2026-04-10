import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def main():
    print("=== Step 3: Bias Analysis & Group-Aware Splits ===")
    DATA_DIR = Path(".")
    INPUT_FILE = DATA_DIR / "step2_clustered_dataset.csv"
    
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run step 2 first.")

    df = pd.read_csv(INPUT_FILE)
    
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

    assert len(train_fams & test_fams) == 0, "Leakage: train ∩ test"
    assert len(train_fams & val_fams) == 0,  "Leakage: train ∩ val"
    assert len(val_fams & test_fams) == 0,   "Leakage: val ∩ test"
    print("✓ No antigen family leakage between splits")

    # --- pKd z-score standardization (paper §4.1) ---
    # Fit on TRAINING set only, transform all splits
    scaler = StandardScaler()
    train_df["pKd_scaled"] = scaler.fit_transform(train_df[["pKd"]]).flatten()
    val_df["pKd_scaled"]   = scaler.transform(val_df[["pKd"]]).flatten()
    test_df["pKd_scaled"]  = scaler.transform(test_df[["pKd"]]).flatten()

    # Save scaler for inference-time inverse transform
    scaler_path = DATA_DIR / "pKd_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved StandardScaler to {scaler_path}")
    print(f"  pKd train mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")

    # Print split summary
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_fam = split_df["ag_cluster_40"].nunique()
        print(f"{name:5s}: {len(split_df):>7,} samples ({100*len(split_df)/total:5.1f}%), "
              f"{n_fam} antigen families, "
              f"pKd mean={split_df['pKd'].mean():.2f}±{split_df['pKd'].std():.2f}, "
              f"pKd_scaled mean={split_df['pKd_scaled'].mean():.4f}±{split_df['pKd_scaled'].std():.4f}")

    train_df.to_csv(DATA_DIR / "train_split.csv", index=False)
    val_df.to_csv(DATA_DIR / "val_split.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_split.csv", index=False)
    print("✓ Saved train_split.csv, val_split.csv, test_split.csv")

if __name__ == "__main__":
    main()
