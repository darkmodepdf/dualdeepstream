import torch
import pandas as pd
import numpy as np
import os, csv, joblib
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dualdeep_model import DualEncoderModel
from dualdeep_dataset import AbAgAffinityDataset
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

    return (np.concatenate(all_embs), np.concatenate(all_targets), np.concatenate(all_clusters))

def main():
    print("=== Step 4: Training ===")
    DATA_DIR = Path(".")
    CKPT_DIR = DATA_DIR / "checkpoints"
    CKPT_DIR.mkdir(exist_ok=True)
    LOG_FILE = DATA_DIR / "training_log.csv"

    # Load pKd scaler for inverse-transform during validation
    scaler_path = DATA_DIR / "pKd_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"{scaler_path} not found. Run step 3 first.")
    pKd_scaler = joblib.load(scaler_path)
    print(f"✓ Loaded pKd scaler (mean={pKd_scaler.mean_[0]:.4f}, std={pKd_scaler.scale_[0]:.4f})")

    print("Loading split datasets...")
    train_df = pd.read_csv(DATA_DIR / "train_split.csv")
    val_df = pd.read_csv(DATA_DIR / "val_split.csv")
    test_df = pd.read_csv(DATA_DIR / "test_split.csv")

    print("Loading tokenizers...")
    ab_tokenizer = AutoTokenizer.from_pretrained("alchemab/antiberta2-cssp")
    ag_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    
    # Dataset uses pKd_scaled (z-score normalized) as target
    train_dataset = AbAgAffinityDataset(train_df, ab_tokenizer, ag_tokenizer, target_col="pKd_scaled")
    val_dataset   = AbAgAffinityDataset(val_df, ab_tokenizer, ag_tokenizer, target_col="pKd_scaled")
    test_dataset  = AbAgAffinityDataset(test_df, ab_tokenizer, ag_tokenizer, target_col="pKd_scaled")

    train_cluster_freq = train_df["ag_cluster_40"].value_counts().to_dict()
    train_weights = train_df["ag_cluster_40"].map(lambda c: 1.0 / train_cluster_freq[c]).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(train_weights),
        num_samples=len(train_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoderModel().to(device)

    # Print parameter summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: total={total_params:,}, trainable={trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # --- NN Baseline ---
    from scipy.stats import spearmanr
    from sklearn.metrics.pairwise import cosine_similarity

    emb_file = DATA_DIR / "nn_baseline_embs.npz"
    if emb_file.exists():
        print(f"\n--- Loading cached NN Baseline Embeddings from {emb_file.name} ---")
        data = np.load(emb_file)
        train_embs = data["train_embs"]
        train_targets_np = data["train_targets"]
        test_embs = data["test_embs"]
        test_targets_np = data["test_targets"]
    else:
        print("\n--- Computing NN Baseline ---")
        train_embs, train_targets_np, _ = compute_embeddings(model, train_loader, device)
        test_embs, test_targets_np, _ = compute_embeddings(model, test_loader, device)
        # NN baseline uses raw (scaled) targets — inverse transform for comparison
        train_targets_raw = pKd_scaler.inverse_transform(train_targets_np.reshape(-1, 1)).flatten()
        test_targets_raw = pKd_scaler.inverse_transform(test_targets_np.reshape(-1, 1)).flatten()
        np.savez(emb_file,
                 train_embs=train_embs, train_targets=train_targets_raw,
                 test_embs=test_embs, test_targets=test_targets_raw)
    
    # NN search on original pKd scale
    if "train_targets_raw" not in dir():
        # Loaded from cache — targets are already raw
        train_targets_raw = train_targets_np
        test_targets_raw = test_targets_np

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

    # --- Training setup ---
    NUM_EPOCHS = 10
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    amp_scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    start_epoch, best_spearman = load_latest_checkpoint(model, optimizer, amp_scaler, scheduler, CKPT_DIR)

    print(f"\n{'='*65}")
    print(f"  Starting training: epochs {start_epoch+1}→{NUM_EPOCHS}")
    print(f"  Batch size: 32 (train), 64 (val)")
    print(f"  LR: {LR}, Weight decay: {WEIGHT_DECAY}, Grad clip: {MAX_GRAD_NORM}")
    print(f"  Target: pKd_scaled (z-score, mean={pKd_scaler.mean_[0]:.4f})")
    print(f"{'='*65}\n")

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

            # Gradient clipping for stability with large PLM embeddings
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

        # Validate — metrics computed on original pKd scale (inverse-transformed)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, scaler=pKd_scaler)
        scheduler.step(val_metrics["spearman"])

        # Log to CSV
        log_row = {"epoch": epoch + 1, "train_loss": avg_train_loss, **val_metrics}
        write_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in log_row.items()})

        print(f"  Epoch {epoch+1}: loss={avg_train_loss:.4f} | "
              f"val_RMSE={val_metrics['rmse']:.4f} | "
              f"val_Pearson={val_metrics['pearson']:.4f} | "
              f"val_Spearman={val_metrics['spearman']:.4f} | "
              f"val_CI={val_metrics['ci']:.4f}")

        is_best = val_metrics["spearman"] > best_spearman
        if is_best:
            best_spearman = val_metrics["spearman"]

        save_checkpoint(
            model, optimizer, amp_scaler, scheduler,
            epoch + 1, val_metrics["spearman"], best_spearman,
            CKPT_DIR, is_best=is_best
        )

    print(f"\n✓ Training complete. Best validation Spearman ρ: {best_spearman:.4f}")

if __name__ == "__main__":
    main()
