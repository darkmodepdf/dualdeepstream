import json
from transformers import AutoConfig

ab_config = AutoConfig.from_pretrained("alchemab/antiberta2-cssp")

with open("antiberta_config.json", "w") as f:
    json.dump({
        "max_position_embeddings": getattr(ab_config, "max_position_embeddings", None),
        "vocab_size": getattr(ab_config, "vocab_size", None)
    }, f)
