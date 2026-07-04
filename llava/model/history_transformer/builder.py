# llava/model/history_transformer/builder.py
import os
import torch
from transformers import PretrainedConfig
from llava.model.history_transformer.history_transformer import VisionTransformer

def build_history_transformer(
        config: PretrainedConfig,
        checkpoint_path: str = None,
) -> VisionTransformer:
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        in_chans=3,
        embed_dim=768,
        depth=4,
        nhead=8,
        mlp_ratio=4.0,
        dropout=0.0,
        dtype=eval(config.model_dtype) if hasattr(config, 'model_dtype') else torch.float32
    )
    if checkpoint_path is not None:
        loaded_state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(loaded_state_dict)
    return model