import torch
import pandas as pd
import numpy as np
import os, csv
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dualdeep_model import DualEncoderModel
from dualdeep_dataset import AbAgAffinityDataset
from dualdeep_utils import evaluate, save_checkpoint, load_latest_checkpoint

def compute_embeddings(model, loader, device):
    all_ab_embs, all_ag_embs, all_targets, all_clusters = [], [], [], []
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
            all_ab_embs.append(combined)
            all_targets.append(batch["target"].numpy())
            all_clusters.append(batch["cluster_id"].numpy())

    return (np.concatenate(all_ab_embs), np.concatenate(all_targets), np.concatenate(all_clusters))

def main():
    print("=== Step 4: Training ===")
    DATA_DIR = Path(".")
    CKPT_DIR = DATA_DIR / "checkpoints"
    CKPT_DIR.mkdir(exist_ok=True)
    LOG_FILE = DATA_DIR / "training_log.csv"

    print("Loading split datasets...")
    train_df = pd.read_csv(DATA_DIR / "train_split.csv")
    val_df = pd.read_csv(DATA_DIR / "val_split.csv")
    test_df = pd.read_csv(DATA_DIR / "test_split.csv")

    print("Loading tokenizers...")
    ab_tokenizer = AutoTokenizer.from_pretrained("alchemab/antiberta2-cssp")
    ag_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    
    train_dataset = AbAgAffinityDataset(train_df, ab_tokenizer, ag_tokenizer)
    val_dataset   = AbAgAffinityDataset(val_df, ab_tokenizer, ag_tokenizer)
    test_dataset  = AbAgAffinityDataset(test_df, ab_tokenizer, ag_tokenizer)

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
        print(f"Saving computed embeddings to {emb_file.name}...")
        np.savez(emb_file, train_embs=train_embs, train_targets=train_targets_np, test_embs=test_embs, test_targets=test_targets_np)
    
    print("Computing cosine similarities...")
    CHUNK = 1000
    nn_preds = np.zeros(len(test_embs))
    for i in tqdm(range(0, len(test_embs), CHUNK), desc="NN search", mininterval=2.0):
        chunk = test_embs[i:i+CHUNK]
        sims = cosine_similarity(chunk, train_embs)
        nn_idx = sims.argmax(axis=1)
        nn_preds[i:i+CHUNK] = train_targets_np[nn_idx]

    nn_rho, _ = spearmanr(test_targets_np, nn_preds)
    print(f"🔑 NN Baseline Spearman ρ: {nn_rho:.4f}")

    NUM_EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    start_epoch, best_spearman = load_latest_checkpoint(model, optimizer, scaler, CKPT_DIR)

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
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

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / n_batches
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["spearman"])

        log_row = {"epoch": epoch + 1, "train_loss": avg_train_loss, **val_metrics}
        write_header = not LOG_FILE.exists() or epoch == start_epoch
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in log_row.items()})

        print(f"  Epoch {epoch+1}: loss={avg_train_loss:.4f} | val_RMSE={val_metrics['rmse']:.4f} | val_Spearman={val_metrics['spearman']:.4f}")

        if val_metrics["spearman"] > best_spearman:
            best_spearman = val_metrics["spearman"]
            save_checkpoint(model, optimizer, scaler, epoch + 1, best_spearman, CKPT_DIR)

if __name__ == "__main__":
    main()
