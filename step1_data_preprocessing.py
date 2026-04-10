import pandas as pd
import numpy as np
import os, re
from pathlib import Path

def main():
    print("=== Step 1: Data Preprocessing ===")
    DATA_DIR = Path(".")
    DATA_FILE = DATA_DIR / "AbRank_dataset.csv"
    OUTPUT_FILE = DATA_DIR / "step1_cleaned_dataset.csv"

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found!")

    print("Loading dataset...")
    df_raw = pd.read_csv(DATA_FILE, sep="\t")
    print(f"Raw dataset: {len(df_raw):,} rows, {df_raw.shape[1]} columns")

    # 1. Keep only exact affinity measurements
    df = df_raw[df_raw["Aff_op"] == "="].copy()
    print(f"After Aff_op='=' filter: {len(df):,} rows")

    # 2. Drop rows with missing critical fields
    df = df.dropna(subset=["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq", "Affinity_Kd [nM]"])
    print(f"After dropping NaN: {len(df):,} rows")

    # Convert Affinity_Kd [nM] to numeric
    df["Affinity_Kd [nM]"] = pd.to_numeric(df["Affinity_Kd [nM]"], errors='coerce')
    df = df.dropna(subset=["Affinity_Kd [nM]"])

    # 3. Filter bandwidth
    df = df[(df["Affinity_Kd [nM]"] > 1e-3) & (df["Affinity_Kd [nM]"] < 1e9)]
    print(f"After Kd range filter: {len(df):,} rows")

    # 4. Clean sequences
    def clean_seq(s):
        if not isinstance(s, str):
            return ""
        return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', s.upper().strip())

    for col in ["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"]:
        df[col] = df[col].apply(clean_seq)

    df = df[(df["Ab_heavy_chain_seq"].str.len() > 0) &
            (df["Ab_light_chain_seq"].str.len() > 0) &
            (df["Ag_seq"].str.len() > 0)]
    print(f"After sequence cleaning: {len(df):,} rows")

    # 5. Compute pKd target
    df["pKd"] = 9 - np.log10(df["Affinity_Kd [nM]"])

    print(f"\n--- Dataset Summary ---")
    print(f"Final samples:      {len(df):,}")
    print(f"pKd range:          [{df['pKd'].min():.2f}, {df['pKd'].max():.2f}]")
    print(f"pKd mean ± std:     {df['pKd'].mean():.2f} ± {df['pKd'].std():.2f}")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved cleaned data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
