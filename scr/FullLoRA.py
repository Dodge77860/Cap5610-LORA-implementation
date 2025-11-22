import torch
import torch.nn as nn
import math

class FullLoRALinear(nn.Module):
    """
    Complete LoRA layer implementation.
    Replaces a Linear layer with a LoRA-enhanced version.

    Features:
    - Low-rank matrices (A, B)
    - Scaling factor (alpha)
    - Optional dropout
    - Weight merge/unmerge for efficient inference
    """

    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False

        # Original dense weight (frozen)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # LoRA down and up matrices
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # Optional dropout before LoRA
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Normal dense output
        result = x @ self.weight.t()

        # LoRA update
        if not self.merged:
            update = self.dropout(x) @ self.A.t()
            update = update @ self.B.t()
            result = result + update * self.scaling

        return result

    def merge_weights(self):
        """For inference: merge low-rank update into the main weight."""
        if not self.merged:
            delta = (self.B @ self.A) * self.scaling
            self.weight.data += delta
            self.merged = True

    def unmerge_weights(self):
        """Reverse merge."""
        if self.merged:
            delta = (self.B @ self.A) * self.scaling
            self.weight.data -= delta
            self.merged = False
