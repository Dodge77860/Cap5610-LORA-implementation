import torch
import torch.nn as nn

class PartialLoRALinear(nn.Module):
    """
    A minimal LoRA replacement for nn.Linear:
    - Copies weights from an existing Linear layer (frozen)
    - Adds trainable low-rank matrices A and B
    - Forward: W x + (BA) x
    """

    def __init__(self, linear_layer: nn.Linear, *, rank=4):  # <- rank is now keyword-only
        super().__init__()
        self.rank = rank
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        # Copy the base weight (freeze it)
        self.weight = nn.Parameter(
            linear_layer.weight.data.clone(),
            requires_grad=False
        )

        # If layer has bias, keep it frozen too
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(
                linear_layer.bias.data.clone(),
                requires_grad=False
            )
        else:
            self.bias = None

        # LoRA matrices
        # A: down-projection (rank × in_features)
        # B: up-projection (out_features × rank)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x):
        # Base output
        base_out = x @ self.weight.t()

        # If LoRA rank is > 0, add update
        if self.rank > 0:
            # (batch, seq, in) -> update -> (batch, seq, out)
            lora_update = x @ self.A.t()
            lora_update = lora_update @ self.B.t()
            out = base_out + lora_update
        else:
            out = base_out

        # Add bias if exists
        if self.bias is not None:
            out += self.bias

        return out
