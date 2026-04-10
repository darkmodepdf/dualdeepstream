import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lifelines.utils import concordance_index

def evaluate(model, loader, device):
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

def save_checkpoint(model, optimizer, scaler, epoch, metric, ckpt_dir, max_ckpts=2):
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
    existing = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(existing) > max_ckpts:
        existing[0].unlink()
        existing.pop(0)

def load_latest_checkpoint(model, optimizer, scaler, ckpt_dir):
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        print("  No checkpoint found — starting from scratch")
        return 0, -float("inf")
    ckpt_path = ckpts[-1]
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    torch.set_rng_state(ckpt["torch_rng_state"].cpu().byte())
    torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu().byte())
    print(f"  ✓ Resumed from {ckpt_path.name} (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["best_metric"]
