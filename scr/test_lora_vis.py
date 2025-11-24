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

        # Forward pass
        output = model(x)
        # Extract the last hidden states tensor
        hidden_states = output.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)
        loss = hidden_states.sum()
        loss.backward()

        # Metrics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6).item()
        output_mean = hidden_states.mean().item()
        output_std = hidden_states.std().item()

        results.append({
            "LoRA Type": lora_type,
            "Rank": rank,
            "Total Params": total_params,
            "Trainable Params": trainable_params,
            "Gradient Norm": grad_norm,
            "Output Mean": output_mean,
            "Output Std": output_std
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
