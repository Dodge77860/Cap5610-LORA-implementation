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
def replace_with_lora(model, lora_type="partial", *, rank=4, alpha=1.0, dropout=0.0):
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

            # Target query, value projections
            if isinstance(child, nn.Linear) and any(k in full_name.lower() for k in ["q", "v"]):
                if lora_type == "partial":
                    # PartialLoRALinear expects the original Linear layer
                    lora_layer = PartialLoRALinear(child, rank=rank)
                elif lora_type == "full":
                    # FullLoRALinear freezes original weight and bias inside constructor
                    lora_layer = FullLoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                else:
                    raise ValueError("lora_type must be 'partial' or 'full'")

                # **Do not manually copy weight or bias** here
                # The LoRA classes handle freezing and copying internally

                # Replace the original Linear layer with the LoRA layer
                setattr(module, child_name, lora_layer)

    return model


# -----------------------------
# Factory function to get a small model
# -----------------------------
def get_model(lora_type=None, *, rank=4, alpha=1.0, dropout=0.0):
    """
    Load a small transformer model and optionally add LoRA layers.

    Args:
        lora_type: "partial", "full", or None
    """
    config = DistilBertConfig()
    model = DistilBertModel(config)

    if lora_type:
        # Freeze all parameters of the base model before replacing with LoRA layers
        for param in model.parameters():
            param.requires_grad = False

        # DEBUG: Verify trainable parameters after freezing
        debug_trainable_params_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"DEBUG: Trainable parameters after initial freeze: {debug_trainable_params_after_freeze}")

        model = replace_with_lora(model, lora_type=lora_type, rank=rank, alpha=alpha, dropout=dropout)

    # Print parameter info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded with {lora_type or 'no'} LoRA.")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    return model
