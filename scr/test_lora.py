# test_lora.py
import torch
from model import get_model

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE = 2
SEQ_LEN = 8
VOCAB_SIZE = 30522  # DistilBERT default vocab size
RANK = 4
LORA_TYPE = "full"  # "partial" or "full"

# ----------------------------
# Create random input_ids
# ----------------------------
input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)

# ----------------------------
# Load model with LoRA
# ----------------------------
model = get_model(lora_type=LORA_TYPE, rank=RANK)
model.train()  # enable gradient computation

# ----------------------------
# Check parameters
# ----------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# ----------------------------
# Forward pass
# ----------------------------
output = model(input_ids=input_ids)
loss = output.last_hidden_state.mean()  # simple dummy loss
print("Forward pass successful. Output shape:", output.last_hidden_state.shape)

# ----------------------------
# Backward pass
# ----------------------------
loss.backward()
total_grad_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None))
print("Backward pass successful. Total gradient norm:", total_grad_norm.item())

# ----------------------------
# Test weight merge/unmerge (Full LoRA only)
# ----------------------------
for name, module in model.named_modules():
    if LORA_TYPE == "full" and hasattr(module, "merge_weights"):
        module.merge_weights()
        module.unmerge_weights()

print("LoRA merge/unmerge test completed successfully.")
