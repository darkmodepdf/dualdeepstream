import torch
import pandas as pd
import numpy as np
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

    test_df = pd.read_csv(DATA_DIR / "test_split.csv")
    ab_tokenizer = AutoTokenizer.from_pretrained("alchemab/antiberta2-cssp")
    ag_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    test_dataset = AbAgAffinityDataset(test_df, ab_tokenizer, ag_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoderModel().to(device)
    
    load_latest_checkpoint(model, None, None, CKPT_DIR)

    test_metrics, test_preds, test_targets, test_clusters = evaluate(model, test_loader, device)

    print("\n" + "=" * 65)
    print("              FINAL TEST METRICS")
    print("=" * 65)
    for k, v in test_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    
    family_results, agg_metrics = per_family_metrics(test_preds, test_targets, test_clusters)
    print("\n--- Per-Family Aggregates ---")
    for k, v in agg_metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    # Calibration plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(test_targets, test_preds, alpha=0.2, s=8, color='steelblue')
    lims = [min(test_targets.min(), test_preds.min()) - 0.5, max(test_targets.max(), test_preds.max()) + 0.5]
    axes[0].plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y=x')
    axes[0].set_title(f"Predicted vs Actual (Pearson r={test_metrics['pearson']:.3f}, R²={test_metrics['r2']:.3f})")
    axes[0].legend()

    residuals = test_preds - test_targets
    axes[1].scatter(test_targets, residuals, alpha=0.2, s=8, color='coral')
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[1].set_title(f"Residual Plot (MAE={test_metrics['mae']:.3f})")
    plt.tight_layout()
    plt.savefig("calibration_plots.png", dpi=200, bbox_inches='tight')
    print("✓ Saved calibration_plots.png")

    # Training history plot
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
        print("✓ Saved training_history.png")

if __name__ == "__main__":
    main()
