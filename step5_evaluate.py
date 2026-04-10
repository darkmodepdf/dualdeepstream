import torch
import pandas as pd
import numpy as np
import joblib
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from transformers import AutoTokenizer, EsmTokenizer
from torch.utils.data import DataLoader

from dualdeep_model import DualEncoderModel
from dualdeep_dataset import AbAgAffinityDataset
from dualdeep_utils import evaluate, load_latest_checkpoint

def per_family_metrics(predictions, targets, cluster_ids):
    results = {}
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        if mask.sum() < 2: continue
        y_true, y_pred = targets[mask], predictions[mask]
        rho_val, _ = spearmanr(y_true, y_pred)
        rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
        results[int(c)] = {"spearman": rho_val, "rmse": rmse_val, "n_samples": int(mask.sum())}

    valid = {k: v for k, v in results.items() if not np.isnan(v["spearman"])}
    spear_vals = [v["spearman"] for v in valid.values()]
    weights = [v["n_samples"] for v in valid.values()]
    return results, {
        "macro_spearman": np.mean(spear_vals) if spear_vals else 0.0,
        "weighted_spearman": np.average(spear_vals, weights=weights) if spear_vals else 0.0,
        "macro_rmse": np.mean([v["rmse"] for v in valid.values()]) if valid else 0.0,
        "weighted_rmse": np.average([v["rmse"] for v in valid.values()], weights=weights) if valid else 0.0,
    }

def main():
    print("=== Step 5: Full Test Evaluation ===")
    DATA_DIR = Path(".")
    CKPT_DIR = DATA_DIR / "checkpoints"
    LOG_FILE = DATA_DIR / "training_log.csv"

    # Load pKd scaler for inverse-transform
    pKd_scaler = joblib.load(DATA_DIR / "pKd_scaler.pkl")
    print(f"✓ Loaded pKd scaler (mean={pKd_scaler.mean_[0]:.4f}, std={pKd_scaler.scale_[0]:.4f})")

    test_df = pd.read_csv(DATA_DIR / "test_split.csv")
    ab_tokenizer = AutoTokenizer.from_pretrained("alchemab/antiberta2-cssp")
    ag_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    test_dataset = AbAgAffinityDataset(test_df, ab_tokenizer, ag_tokenizer, target_col="pKd_scaled")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoderModel().to(device)
    
    load_latest_checkpoint(model, None, None, CKPT_DIR)

    # Evaluate with inverse-transform to original pKd scale
    test_metrics, test_preds, test_targets, test_clusters = evaluate(
        model, test_loader, device, scaler=pKd_scaler
    )

    print("\n" + "=" * 65)
    print("              FINAL TEST METRICS (original pKd scale)")
    print("=" * 65)
    for k, v in test_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    
    family_results, agg_metrics = per_family_metrics(test_preds, test_targets, test_clusters)
    print("\n--- Per-Family Aggregates ---")
    for k, v in agg_metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    print(f"\n--- Top-10 Families by Sample Count ---")
    sorted_fams = sorted(family_results.items(), key=lambda x: x[1]["n_samples"], reverse=True)[:10]
    print(f"  {'Family':>8s}  {'Samples':>8s}  {'Spearman':>10s}  {'RMSE':>8s}")
    for fam_id, fam_data in sorted_fams:
        print(f"  {fam_id:>8d}  {fam_data['n_samples']:>8d}  {fam_data['spearman']:>10.4f}  {fam_data['rmse']:>8.4f}")

    # --- Calibration plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(test_targets, test_preds, alpha=0.2, s=8, color='steelblue')
    lims = [min(test_targets.min(), test_preds.min()) - 0.5, max(test_targets.max(), test_preds.max()) + 0.5]
    axes[0].plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y=x')
    m, b = np.polyfit(test_targets, test_preds, 1)
    x_line = np.linspace(lims[0], lims[1], 100)
    axes[0].plot(x_line, m * x_line + b, 'r-', linewidth=2, label=f'Fit: y={m:.2f}x+{b:.2f}')
    axes[0].set_xlabel("Actual pKd", fontsize=12)
    axes[0].set_ylabel("Predicted pKd", fontsize=12)
    axes[0].set_title(f"Predicted vs Actual (Pearson r={test_metrics['pearson']:.3f}, R²={test_metrics['r2']:.3f})")
    axes[0].legend()
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)

    residuals = test_preds - test_targets
    axes[1].scatter(test_targets, residuals, alpha=0.2, s=8, color='coral')
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[1].set_xlabel("Actual pKd", fontsize=12)
    axes[1].set_ylabel("Residual (Pred − Actual)", fontsize=12)
    axes[1].set_title(f"Residual Plot (MAE={test_metrics['mae']:.3f})")
    plt.tight_layout()
    plt.savefig("calibration_plots.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved calibration_plots.png")

    # --- Training history plot ---
    if LOG_FILE.exists():
        log_df = pd.read_csv(LOG_FILE)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(log_df["epoch"], log_df["train_loss"], 'b-', label="Train Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].legend()
        axes[0, 1].plot(log_df["epoch"], log_df["rmse"], 'r-', label="Val RMSE")
        axes[0, 1].set_title("Validation RMSE")
        axes[0, 1].legend()
        axes[1, 0].plot(log_df["epoch"], log_df["spearman"], 'm-', label="Val Spearman ρ")
        axes[1, 0].plot(log_df["epoch"], log_df["pearson"], 'g-', label="Val Pearson r")
        axes[1, 0].set_title("Correlation Metrics")
        axes[1, 0].legend()
        axes[1, 1].plot(log_df["epoch"], log_df["ci"], 'c-', label="Val CI")
        axes[1, 1].set_title("Concordance Index")
        axes[1, 1].legend()
        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved training_history.png")

if __name__ == "__main__":
    main()
