import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from model import get_model  # Your get_model function with LoRA support

# ---------------------------------------
# CONFIG
# ---------------------------------------
BATCH_SIZE = 2
SEQ_LEN = 8
VOCAB_SIZE = 30522
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANKS = [1, 2, 4, 8]
LORA_TYPE = "full"

results = []
output_samples = {}   # Store some forward passes to compare later

# ---------------------------------------
# RUN EXPERIMENT
# ---------------------------------------
for rank in RANKS:
    print(f"\nTesting {LORA_TYPE} LoRA with rank={rank}...\n")

    model = get_model(lora_type=LORA_TYPE, rank=rank).to(DEVICE)
    model.train()

    # Random toy input
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    out = model(x)
    H = out.last_hidden_state
    loss = H.sum()
    loss.backward()

    # Store first example output for comparison
    output_samples[rank] = H[0, 0, :20].detach().cpu().numpy()  # first 20 dims

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params

    # Memory estimate (float32 = 4 bytes)
    mem_estimate_mb = (trainable_params * 4) / 1e6

    # Gradients (flatten all)
    grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    grad_norm = grads.norm().item()

    results.append({
        "Rank": rank,
        "Total Params": total_params,
        "Trainable Params": trainable_params,
        "Param Ratio": ratio,
        "Grad Norm": grad_norm,
        "Memory_MB": mem_estimate_mb,
        "Output Mean": H.mean().item(),
        "Output Std": H.std().item()
    })

    model.zero_grad()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------
# CONVERT TO DATAFRAME
# ---------------------------------------
df = pd.DataFrame(results)
print("\nSummary Table:\n")
print(df)


# ---------------------------------------
# VISUAL 1 — Parameter ratio
# ---------------------------------------
print("\nParameter Ratios (Trainable / Total):\n")
print(df[["Rank", "Param Ratio"]])

plt.figure(figsize=(8,5))
sns.barplot(x="Rank", y="Param Ratio", data=df)
plt.title("Trainable / Total Parameter Ratio")
plt.xlabel("LoRA Rank")
plt.ylabel("Ratio")
plt.tight_layout()
plt.show()
print("\n\n")  # spacing



# ---------------------------------------
# VISUAL 2 — Trainable vs Total Params
# ---------------------------------------
print("\nTrainable vs Total Parameters:\n")
print(df[["Rank", "Total Params", "Trainable Params"]])

df_melt = df.melt(id_vars="Rank",
                  value_vars=["Total Params", "Trainable Params"],
                  var_name="Type", value_name="Count")

plt.figure(figsize=(8,5))
sns.barplot(x="Rank", y="Count", hue="Type", data=df_melt)
plt.yscale("log")
plt.title("Total vs Trainable Parameters (Log Scale)")
plt.tight_layout()
plt.show()
print("\n\n")   # spacing



# ---------------------------------------
# VISUAL 3 — Gradient Norm vs Rank
# ---------------------------------------
print("\nGradient Norms:\n")
print(df[["Rank", "Grad Norm"]])

plt.figure(figsize=(8,5))
sns.lineplot(x="Rank", y="Grad Norm", marker="o", data=df)
plt.title("Gradient Norm vs Rank")
plt.tight_layout()
plt.show()
print("\n\n")   # spacing



# ---------------------------------------
# VISUAL 4 — Gradient Distribution Histogram
# ---------------------------------------
for rank in RANKS:
    print(f"\nGradient Distribution for Rank={rank}: (Histogram Below)")

    model = get_model(lora_type=LORA_TYPE, rank=rank).to(DEVICE)
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    out = model(x)
    H = out.last_hidden_state
    H.sum().backward()

    grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    grads = grads.detach().cpu().numpy()

    plt.figure(figsize=(8,5))
    sns.histplot(grads, bins=50)
    plt.title(f"Gradient Distribution (Rank {rank})")
    plt.xlabel("Gradient Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    print("\n\n")   # spacing



# ---------------------------------------
# VISUAL 5 — Output Mean & Std
# ---------------------------------------
print("\nOutput Stats:\n")
print(df[["Rank", "Output Mean", "Output Std"]])

plt.figure(figsize=(8,5))
sns.lineplot(x="Rank", y="Output Mean", marker="o", data=df)
plt.title("Output Mean vs Rank")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.lineplot(x="Rank", y="Output Std", marker="o", data=df)
plt.title("Output Std vs Rank")
plt.tight_layout()
plt.show()
print("\n\n")   # spacing



# ---------------------------------------
# VISUAL 6 — Forward Pass Output Samples
# ---------------------------------------
print("\nSample Forward Output Vector (first 20 dims):\n")
for rank, vec in output_samples.items():
    print(f"Rank {rank}:", vec, "\n")
