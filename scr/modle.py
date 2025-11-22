"""
model.py

Provides a small Transformer model (DistilBERT or TinyBERT) and allows
replacing attention Linear layers with LoRA modules (Partial or Full).
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

# Import your LoRA modules
from PartialLoRA import PartialLoRALinear
from FullLoRA import FullLoRALinear

# -----------------------------
# Helper function to replace Linear layers in attention
# -----------------------------
def replace_with_lora(model, lora_type="partial", rank=4, alpha=1.0, dropout=0.0):
    """
    Replace attention projection Linear layers with LoRA modules.

    Args:
        model: Hugging Face Transformer model
        lora_type: "partial" or "full"
        rank: LoRA rank
        alpha: scaling factor (only used in FullLoRA)
        dropout: dropout rate (only used in FullLoRA)
    """
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Target query, key, value projections
            if isinstance(child, nn.Linear) and any(k in full_name.lower() for k in ["q", "v"]):
                in_features = child.in_features
                out_features = child.out_features

                if lora_type == "partial":
                    lora_layer = PartialLoRALinear(in_features, out_features, rank=rank)
                elif lora_type == "full":
                    lora_layer = FullLoRALinear(in_features, out_features, rank=rank, alpha=alpha, dropout=dropout)
                else:
                    raise ValueError("lora_type must be 'partial' or 'full'")

                # Copy original weights
                lora_layer.weight.data.copy_(child.weight.data)
                if hasattr(child, "bias") and child.bias is not None:
                    lora_layer.bias = nn.Parameter(child.bias.data.clone())
                setattr(module, child_name, lora_layer)

    return model

# -----------------------------
# Factory function to get a small model
# -----------------------------
def get_model(lora_type=None, rank=4, alpha=1.0, dropout=0.0):
    """
    Load a small transformer model and optionally add LoRA layers.

    Args:
        lora_type: "partial", "full", or None
    """
    config = DistilBertConfig()
    model = DistilBertModel(config)

    if lora_type:
        model = replace_with_lora(model, lora_type=lora_type, rank=rank, alpha=alpha, dropout=dropout)

    return model
