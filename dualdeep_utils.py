"""
dualdeep_utils.py — Shared utilities: evaluation, checkpointing.
"""

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lifelines.utils import concordance_index


def evaluate(model, loader, device, scaler=None):
    """
    Evaluate model on a DataLoader.

    Args:
        model: DualEncoderModel
        loader: DataLoader
        device: torch device
        scaler: sklearn StandardScaler for pKd. If provided, predictions and
                targets are inverse-transformed to original pKd scale before
                computing metrics. If None, metrics are on raw (scaled) values.
    Returns:
        metrics_dict, predictions, targets, cluster_ids
    """
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

    # Inverse-transform to original pKd scale for meaningful metrics
    if scaler is not None:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        targets = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

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


def save_checkpoint(model, optimizer, scaler, scheduler, epoch, metric, best_metric, ckpt_dir, is_best=False, max_best=2):
    if not is_best:
        return
        
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    
    best_path = ckpt_dir / f"ckpt_best_epoch{epoch:03d}_spearman{metric:.4f}.pt"
    torch.save(state, best_path)
    
    existing = sorted(ckpt_dir.glob("ckpt_best_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(existing) > max_best:
        existing[0].unlink()
        existing.pop(0)
    print(f"  💾 Saved BEST checkpoint: {best_path.name}")


def load_latest_checkpoint(model, optimizer, scaler, scheduler, ckpt_dir):
    ckpts = sorted(ckpt_dir.glob("ckpt_best_*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
        
    if not ckpts:
        print("  No checkpoint found — starting from scratch")
        return 0, -float("inf")
        
    ckpt_path = ckpts[-1]
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu().byte())
    if "cuda_rng_state" in ckpt:
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu().byte())
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
        
    epoch = ckpt["epoch"]
    best_metric = ckpt.get("best_metric", -float("inf"))
    print(f"  ✓ Resumed from {ckpt_path.name} (epoch {epoch})")
    return epoch, best_metric
