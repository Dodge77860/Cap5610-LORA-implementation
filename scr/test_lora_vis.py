# @title
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
LORA_TYPE = "full"  # Only full LoRA for clarity

# -----------------------------
# Metrics storage
# -----------------------------
results = []

# -----------------------------
# Loop over LoRA configs
# -----------------------------
for rank in RANKS:
    print(f"Testing {LORA_TYPE} LoRA with rank={rank}...")
    model = get_model(lora_type=LORA_TYPE, rank=rank).to(DEVICE)
    model.train()  # Ensure gradients are tracked

    # Generate dummy input (random token IDs)
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=DEVICE)

    # Forward pass
    output = model(x)
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
        "Rank": rank,
        "Total Params": total_params,
        "Trainable Params": trainable_params,
        "Gradient Norm": grad_norm,
        "Output Mean": output_mean,
        "Output Std": output_std
    })

    # Clear gradients
    model.zero_grad()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# -----------------------------
# Convert to DataFrame
# -----------------------------
df = pd.DataFrame(results)
print("Summary of results:")
print(df)

# -----------------------------
# Visualizations
# -----------------------------
sns.set(style="whitegrid")

# Log-scale bar plot: Total vs Trainable Parameters
plt.figure(figsize=(8,5))
df_melted = df.melt(id_vars="Rank", value_vars=["Total Params", "Trainable Params"],
                    var_name="Parameter Type", value_name="Count")
sns.barplot(x="Rank", y="Count", hue="Parameter Type", data=df_melted)
plt.yscale("log")  # Log scale to make trainable params visible
plt.title(f"{LORA_TYPE.capitalize()} LoRA: Total vs Trainable Parameters (Log Scale)")
plt.ylabel("Number of Parameters (log scale)")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()

# Gradient norm vs rank
plt.figure(figsize=(8,5))
sns.lineplot(x="Rank", y="Gradient Norm", marker="o", data=df)
plt.title(f"{LORA_TYPE.capitalize()} LoRA: Gradient Norm vs Rank")
plt.ylabel("Total Gradient Norm")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()

# Forward output statistics
plt.figure(figsize=(8,5))
sns.lineplot(x="Rank", y="Output Mean", marker="o", data=df)
plt.title(f"{LORA_TYPE.capitalize()} LoRA: Forward Output Mean vs Rank")
plt.ylabel("Output Mean")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.lineplot(x="Rank", y="Output Std", marker="o", data=df)
plt.title(f"{LORA_TYPE.capitalize()} LoRA: Forward Output Std vs Rank")
plt.ylabel("Output Std")
plt.xlabel("LoRA Rank")
plt.tight_layout()
plt.show()
