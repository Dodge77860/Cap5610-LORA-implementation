# test_lora_vis.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from model import get_model  # Your get_model function with LoRA support

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 2
SEQ_LEN = 8
VOCAB_SIZE = 30522  # Standard DistilBERT vocab
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANKS = [1, 2, 4, 8]
LORA_TYPES = ["partial", "full"]

# -----------------------------
# Metrics storage
# -----------------------------
results = []

# -----------------------------
# Loop over LoRA configs
# -----------------------------
for lora_type in LORA_TYPES:
    for rank in RANKS:
        print(f"Testing {lora_type} LoRA with rank={rank}...")
        model = get_model(lora_type=lora_type, rank=rank).to(DEVICE)
        model.train()  # Ensure gradients are tracked

        # Generate dummy input (random token IDs)
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)

        # ---- LoRA matrix norms before forward/backward ----
        a_norm_before = sum(torch.norm(m.A).item() for m in model.modules() if hasattr(m, "A"))
        b_norm_before = sum(torch.norm(m.B).item() for m in model.modules() if hasattr(m, "B"))

        # Forward pass
        output = model(x)
        hidden_states = output.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)
        loss = hidden_states.sum()
        loss.backward()

        # ---- LoRA matrix norms after backward ----
        a_norm_after = sum(torch.norm(m.A).item() for m in model.modules() if hasattr(m, "A"))
        b_norm_after = sum(torch.norm(m.B).item() for m in model.modules() if hasattr(m, "B"))

        # ---- Metrics ----
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6).item()
        output_mean = hidden_states.mean().item()
        output_std = hidden_states.std().item()

        # ---- Merge / Unmerge Test ----
        # Forward again after merging LoRA weights
        for m in model.modules():
            if hasattr(m, "merge_weights"):
                m.merge_weights()
        output_merged = model(x)
        max_diff = (output_merged.last_hidden_state - hidden_states).abs().max().item()
        for m in model.modules():
            if hasattr(m, "unmerge_weights"):
                m.unmerge_weights()

        results.append({
            "LoRA Type": lora_type,
            "Rank": rank,
            "Total Params": total_params,
            "Trainable Params": trainable_params,
            "Gradient Norm": grad_norm,
            "Output Mean": output_mean,
            "Output Std": output_std,
            "A Norm Before": a_norm_before,
            "A Norm After": a_norm_after,
            "B Norm Before": b_norm_before,
            "B Norm After": b_norm_after,
            "Merge Max Diff": max_diff
        })

        # Zero gradients before next iteration
        model.zero_grad()
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

# -----------------------------
# Convert to DataFrame
# -----------------------------
df = pd.DataFrame(results)
print(df)

# -----------------------------
# Visualizations
# -----------------------------
sns.set(style="whitegrid")

# Trainable params vs rank
plt.figure(figsize=(8,5))
sns.barplot(x="Rank", y="Trainable Params", hue="LoRA Type", data=df)
plt.title("Trainable Parameters vs LoRA Rank")
plt.ylabel("Trainable Parameters")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()

# Gradient norm vs rank
plt.figure(figsize=(8,5))
sns.lineplot(x="Rank", y="Gradient Norm", hue="LoRA Type", marker="o", data=df)
plt.title("Gradient Norm vs LoRA Rank")
plt.ylabel("Total Gradient Norm")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()

# Output statistics
plt.figure(figsize=(8,5))
sns.lineplot(x="Rank", y="Output Mean", hue="LoRA Type", marker="o", data=df)
plt.title("Forward Output Mean vs LoRA Rank")
plt.ylabel("Output Mean")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.lineplot(x="Rank", y="Output Std", hue="LoRA Type", marker="o", data=df)
plt.title("Forward Output Std vs LoRA Rank")
plt.ylabel("Output Std")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()

# LoRA A/B norms
plt.figure(figsize=(8,5))
sns.barplot(x="Rank", y="A Norm After", hue="LoRA Type", data=df)
plt.title("LoRA A Matrix Norm After Backward")
plt.ylabel("Frobenius Norm")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x="Rank", y="B Norm After", hue="LoRA Type", data=df)
plt.title("LoRA B Matrix Norm After Backward")
plt.ylabel("Frobenius Norm")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()

# Merge / Unmerge difference
plt.figure(figsize=(6,5))
sns.scatterplot(x="Rank", y="Merge Max Diff", hue="LoRA Type", data=df, s=100)
plt.title("Max Difference After LoRA Merge/Unmerge")
plt.ylabel("Max |output_merged - output|")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()
