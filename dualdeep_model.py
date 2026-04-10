"""
dualdeep_model.py — DuaDeep-SeqAffinity dual-stream model.

Architecture (per paper §3.1–3.4):
  Heavy chain  → AntiBERTa2-CSSP (frozen) → mean pool → (768,)  ─┐
                                                                    ├─ concat → (1536,)
  Light chain  → AntiBERTa2-CSSP (frozen) → mean pool → (768,)  ─┘
                          │
                ┌─────────┴──────────┐
          TransformerBranch      CNNBranch
            (1536→1536)         (1536→128)
                │                    │
            avg pool              adapt pool
            (1536,)              (128,)
                └──── concat ────────┘ = (1664,)  = ab_fused

  Antigen → ESM-2 150M (frozen) → (seq_len, 640)
                          │
                ┌─────────┴──────────┐
          TransformerBranch      CNNBranch
            (640→640)           (640→128)
                │                    │
            avg pool              adapt pool
            (640,)               (128,)
                └──── concat ────────┘ = (768,)   = ag_fused

  Concat(ab_fused, ag_fused) → MLP → scalar pKd
"""

import torch
import torch.nn as nn
from transformers import AutoModel, EsmModel


class TransformerBranch(nn.Module):
    """
    Learnable Transformer encoder branch (paper §3.2.1).
    Stacked Transformer encoder layers on top of frozen PLM embeddings.
    Outputs global average pool → fixed-size vector.
    """

    def __init__(self, d_model, nhead=8, num_layers=2, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, L, D) — PLM embeddings
            attention_mask: (B, L) — 1=real token, 0=pad
        Returns:
            (B, D) — global average pooled Transformer output
        """
        # TransformerEncoder expects src_key_padding_mask where True=pad
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        out = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, D)

        # Global average pooling over non-pad tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)
        return pooled  # (B, D)


class CNNBranch(nn.Module):
    """
    1D CNN branch for local motif detection (paper §3.2.2).
    Conv1d(D_E, 256, kernel=3) → ReLU → Conv1d(256, 128, kernel=5) → ReLU → AdaptiveAvgPool1d(1)
    Output: (B, 128)
    """

    def __init__(self, in_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, L, D) — PLM embeddings
            attention_mask: (B, L) — 1=real token, 0=pad
        Returns:
            (B, 128) — locally-aware feature vector
        """
        # Mask padding before conv
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        # Conv1d expects (B, C, L)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.relu(self.conv1(x))  # (B, 256, L)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))  # (B, 128, L)
        x = self.dropout(x)
        # x is (B, 128, L)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).float()  # (B, 1, L)
            # Mask out padding just in case conv fields bled into them
            x = x * mask
            # Average over valid tokens
            x = x.sum(dim=2) / mask.sum(dim=2).clamp(min=1e-9)  # (B, 128)
        else:
            x = x.mean(dim=2)
        return x


class DualStreamBlock(nn.Module):
    """
    Combined Transformer + CNN dual-stream (paper §3.2 + §3.3).
    Takes PLM embeddings, runs both branches in parallel, concatenates outputs.
    Output dim = d_model + 128 (Transformer pool + CNN pool).
    """

    def __init__(self, d_model, nhead=8, num_transformer_layers=2, dropout=0.1):
        super().__init__()
        self.transformer_branch = TransformerBranch(
            d_model=d_model, nhead=nhead,
            num_layers=num_transformer_layers, dropout=dropout,
        )
        self.cnn_branch = CNNBranch(in_channels=d_model, dropout=dropout)
        self.output_dim = d_model + 128  # Transformer + CNN

    def forward(self, x, attention_mask=None):
        t_out = self.transformer_branch(x, attention_mask)  # (B, d_model)
        c_out = self.cnn_branch(x, attention_mask)           # (B, 128)
        return torch.cat([t_out, c_out], dim=-1)             # (B, d_model + 128)


class DualEncoderModel(nn.Module):
    """
    Full DuaDeep-SeqAffinity model.

    Frozen PLM encoders → dual-stream (Transformer + CNN) → fusion → MLP → pKd.
    """

    def __init__(self, ab_model_name="alchemab/antiberta2-cssp",
                 ag_model_name="facebook/esm2_t30_150M_UR50D",
                 nhead=8, num_transformer_layers=2,
                 mlp_hidden=None, dropout=0.1):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [512, 256, 128]

        # --- Frozen antibody encoder (shared for H and L) ---
        self.ab_encoder = AutoModel.from_pretrained(ab_model_name)
        for p in self.ab_encoder.parameters():
            p.requires_grad = False
        self.ab_encoder.eval()
        self.ab_hidden = self.ab_encoder.config.hidden_size  # 768

        # --- Frozen antigen encoder ---
        self.ag_encoder = EsmModel.from_pretrained(ag_model_name)
        for p in self.ag_encoder.parameters():
            p.requires_grad = False
        self.ag_encoder.eval()
        self.ag_hidden = self.ag_encoder.config.hidden_size  # 640

        # --- Embedding Projection (Align latent spaces) ---
        self.shared_dim = 512
        self.ab_proj = nn.Sequential(
            nn.Linear(self.ab_hidden, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.Dropout(dropout)
        )
        self.ag_proj = nn.Sequential(
            nn.Linear(self.ag_hidden, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.Dropout(dropout)
        )

        # --- Dual-stream blocks ---
        # The paper specifies ONE dual stream for the Antibody and ONE for the Antigen.
        # We process Heavy and Light chains separately through the frozen PLM, project
        # them to the shared hidden dimension, and then concatenate them sequence-wise
        # before feeding into the single Antibody dual-stream block.
        self.ab_dualstream = DualStreamBlock(
            d_model=self.shared_dim, nhead=nhead,
            num_transformer_layers=num_transformer_layers, dropout=dropout,
        )
        self.ag_dualstream = DualStreamBlock(
            d_model=self.shared_dim, nhead=nhead,
            num_transformer_layers=num_transformer_layers, dropout=dropout,
        )

        # Fused dims:
        #   Total input to MLP: (512 + 128) * 2 = 1280
        fused_dim = self.ab_dualstream.output_dim + self.ag_dualstream.output_dim

        # --- MLP Head ---
        layers = []
        in_dim = fused_dim
        for h in mlp_hidden:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp_head = nn.Sequential(*layers)

    def mean_pool(self, last_hidden_state, attention_mask):
        """Mean pool over non-pad tokens (used for NN baseline embeddings)."""
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, heavy_input_ids, heavy_attention_mask,
                light_input_ids, light_attention_mask,
                ag_input_ids, ag_attention_mask):
        # Frozen encoders — no gradient
        with torch.no_grad():
            h_embs = self.ab_encoder(
                input_ids=heavy_input_ids,
                attention_mask=heavy_attention_mask
            ).last_hidden_state  # (B, L_h, 768)

            l_embs = self.ab_encoder(
                input_ids=light_input_ids,
                attention_mask=light_attention_mask
            ).last_hidden_state  # (B, L_l, 768)

            ag_embs = self.ag_encoder(
                input_ids=ag_input_ids,
                attention_mask=ag_attention_mask
            ).last_hidden_state  # (B, L_ag, 640)

        # Project to shared space
        h_embs_proj = self.ab_proj(h_embs)  # (B, L_h, self.shared_dim)
        l_embs_proj = self.ab_proj(l_embs)  # (B, L_l, self.shared_dim)
        ag_embs_proj = self.ag_proj(ag_embs) # (B, L_ag, self.shared_dim)

        # Unify Antibody Streams (Concat sequence length)
        ab_embs_proj = torch.cat([h_embs_proj, l_embs_proj], dim=1) # (B, L_h + L_l, self.shared_dim)
        ab_attention_mask = torch.cat([heavy_attention_mask, light_attention_mask], dim=1) # (B, L_h + L_l)

        # Dual-stream feature extraction (LEARNABLE)
        ab_fused = self.ab_dualstream(ab_embs_proj, ab_attention_mask)    
        ag_fused = self.ag_dualstream(ag_embs_proj, ag_attention_mask)

        # Inter-protein fusion
        fused = torch.cat([ab_fused, ag_fused], dim=-1)  # (B, 1280)

        # Regression
        return self.mlp_head(fused).squeeze(-1)  # (B,)
