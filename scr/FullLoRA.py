import torch
import torch.nn as nn
import math

class FullLoRALinear(nn.Module):
    """
    Full LoRA layer implementation:
    - Takes an existing Linear layer and copies its weights (frozen)
    - Adds trainable low-rank matrices A and B
    - Supports alpha scaling and dropout
    - Supports weight merging for inference
    """

    def __init__(self, linear_layer: nn.Linear, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        # ---- Base weight (copied + frozen) ----
        self.weight = nn.Parameter(linear_layer.weight.data.clone(), requires_grad=False)

        # ---- Base bias (copied + frozen) ----
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # ---- LoRA trainable parameters ----
        # A: down projection (rank x in_features)
        # B: up projection (out_features x rank)
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))

        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # He initialization
        nn.init.zeros_(self.B)  # Zero initialization is common in LoRA

        # Optional dropout before low-rank update
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Base output (frozen)
        result = x @ self.weight.t()
        if self.bias is not None:
            result = result + self.bias

        # LoRA update (only if unmerged)
        if not self.merged and self.rank > 0:
            update = self.dropout(x) @ self.A.t()
            update = update @ self.B.t()
            result = result + update * self.scaling

        return result

    def merge_weights(self):
        """
        Merge LoRA update into the frozen weight for inference.
        """
        if not self.merged:
            delta = (self.B @ self.A) * self.scaling
            self.weight.data += delta
            self.merged = True

    def unmerge_weights(self):
        """
        Reverse weight merging (restore base weight).
        """
        if self.merged:
            delta = (self.B @ self.A) * self.scaling
            self.weight.data -= delta
            self.merged = False
