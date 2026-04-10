import torch
import torch.nn as nn
from transformers import AutoModel, EsmModel

class DualEncoderModel(nn.Module):
    def __init__(self, ab_model_name="alchemab/antiberta2-cssp",
                 ag_model_name="facebook/esm2_t30_150M_UR50D",
                 proj_dim=256, mlp_hidden=None, dropout=0.1):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [512, 256, 128]

        # --- Frozen antibody encoder (shared for H and L) ---
        self.ab_encoder = AutoModel.from_pretrained(ab_model_name)
        for p in self.ab_encoder.parameters():
            p.requires_grad = False
        self.ab_encoder.eval()
        ab_hidden = self.ab_encoder.config.hidden_size  # 768

        # --- Frozen antigen encoder ---
        self.ag_encoder = EsmModel.from_pretrained(ag_model_name)
        for p in self.ag_encoder.parameters():
            p.requires_grad = False
        self.ag_encoder.eval()
        ag_hidden = self.ag_encoder.config.hidden_size  # 640

        # --- Learned projection layers ---
        self.ab_proj = nn.Sequential(
            nn.Linear(ab_hidden * 2, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )
        self.ag_proj = nn.Sequential(
            nn.Linear(ag_hidden, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )

        # --- MLP Head ---
        layers = []
        in_dim = proj_dim * 2
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
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, heavy_input_ids, heavy_attention_mask,
                light_input_ids, light_attention_mask,
                ag_input_ids, ag_attention_mask):
        with torch.no_grad():
            h_out = self.ab_encoder(input_ids=heavy_input_ids,
                                    attention_mask=heavy_attention_mask).last_hidden_state
            l_out = self.ab_encoder(input_ids=light_input_ids,
                                    attention_mask=light_attention_mask).last_hidden_state
            ag_out = self.ag_encoder(input_ids=ag_input_ids,
                                     attention_mask=ag_attention_mask).last_hidden_state

        h_pooled = self.mean_pool(h_out, heavy_attention_mask)
        l_pooled = self.mean_pool(l_out, light_attention_mask)
        ab_pooled = torch.cat([h_pooled, l_pooled], dim=-1)
        ag_pooled = self.mean_pool(ag_out, ag_attention_mask)

        ab_proj = self.ab_proj(ab_pooled)
        ag_proj = self.ag_proj(ag_pooled)

        fused = torch.cat([ab_proj, ag_proj], dim=-1)
        return self.mlp_head(fused).squeeze(-1)
