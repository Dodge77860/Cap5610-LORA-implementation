"""
train.py

Lightweight script to verify LoRA integration (Type A scope):
- Load small transformer (DistilBERT)
- Apply Partial or Full LoRA
- Run forward/backward pass on dummy data
- Print simple metrics (loss, trainable params, gradients)
"""

import torch
import torch.nn as nn
from model import get_model
from utils import count_trainable_params

# -----------------------------
# Configuration
# -----------------------------
LORA_TYPE = "partial"  # "partial" or "full"
RANK = 4
ALPHA = 16
DROPOUT = 0.0
SEQ_LEN = 10
BATCH_SIZE = 2
VOCAB_SIZE = 30522  # DistilBERT default

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dummy dataset (small, just for testing)
# -----------------------------
def generate_dummy_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(DEVICE)
    attention_mask = torch.ones(batch_size, seq_len).to(DEVICE)
    labels = torch.randint(0, 2, (batch_size,)).to(DEVICE)  # binary classification
    return input_ids, attention_mask, labels

# -----------------------------
# Main function
# -----------------------------
def main():
    # Load model with LoRA
    model = get_model(lora_type=LORA_TYPE, rank=RANK, alpha=ALPHA, dropout=DROPOUT)
    model.to(DEVICE)

    print(f"Model loaded with {LORA_TYPE} LoRA.")
    print(f"Trainable parameters: {count_trainable_params(model)}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Generate dummy input
    input_ids, attention_mask, labels = generate_dummy_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)
    # For DistilBERT, last_hidden_state shape: [batch, seq_len, hidden_dim]
    # Take first token (CLS) for simplicity
    cls_token = outputs.last_hidden_state[:, 0, :]
    
    # Simulate a classifier (linear layer) for testing
    classifier = nn.Linear(cls_token.size(-1), 2).to(DEVICE)
    logits = classifier(cls_token)

    # Compute loss
    loss = criterion(logits, labels)
    print(f"Forward pass loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()
    print("Backward pass successful. Gradients computed.")

    # Optional: check gradient norms
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm:.4f}")

if __name__ == "__main__":
    main()
