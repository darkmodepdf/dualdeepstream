import pandas as pd
import numpy as np

def analyze():
    df_raw = pd.read_csv("AbRank_dataset.csv", sep="\t", low_memory=False)
    print(f"Raw shape: {df_raw.shape}")
    
    # 1. Filter
    df = df_raw[df_raw["Aff_op"] == "="].copy()
    
    # 2. Drop NA
    df = df.dropna(subset=["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq", "Affinity_Kd [nM]"])
    
    # Convert to numeric
    df["Kd"] = pd.to_numeric(df["Affinity_Kd [nM]"], errors='coerce')
    df = df.dropna(subset=["Kd"])
    
    print("\nKd raw stats (before range filter):")
    print(df["Kd"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    
    # Check max and min
    print(f"Min Kd: {df['Kd'].min()} nM")
    print(f"Max Kd: {df['Kd'].max()} nM")
    
    df_filtered = df[(df["Kd"] > 1e-3) & (df["Kd"] < 1e9)].copy()
    print(f"\nRemoved {(len(df) - len(df_filtered))} samples due to limit range [1e-3, 1e9]")
    
    df_filtered["pKd"] = 9 - np.log10(df_filtered["Kd"])
    
    print("\npKd stats (after range filter):")
    print(df_filtered["pKd"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    
if __name__ == "__main__":
    analyze()
