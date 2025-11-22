import torch
import torch.nn as nn

class PartialLoRALinear(nn.Module):
    """
    A minimal LoRA layer:
    - Takes an existing Linear layer
    - Freezes its weights
    - Adds trainable low-rank matrices A and B
    - Forward pass returns: W x + BA x
    """

    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.rank = rank

        # Freeze main weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # LoRA matrices (A: down, B: up)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x):
        # Low-rank update
        lora_update = self.B @ (self.A @ x.transpose(-1, -2))
        lora_update = lora_update.transpose(-1, -2)

        # Base + LoRA update
        return x @ self.weight.t() + lora_update
