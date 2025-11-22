"""
utils.py

Utility functions for:
- Dataset loading (SST-2 or AG News)
- Computing metrics (accuracy)
- Plotting training curves
"""

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import os

# -----------------------------
# Dataset Loading
# -----------------------------
def load_dataset(dataset_name="sst2", tokenizer_name="distilbert-base-uncased", max_length=64, batch_size=8, subset_size=None):
    """
    Load a Hugging Face dataset and return DataLoaders for training and validation.

    Args:
        dataset_name: "sst2" or "ag_news"
        tokenizer_name: tokenizer to use
        max_length: max token length
        batch_size: batch size
        subset_size: limit number of examples for quick testing
    """
    from datasets import load_dataset

    if dataset_name.lower() == "sst2":
        dataset = load_dataset("glue", "sst2")
    elif dataset_name.lower() == "ag_news":
        dataset = load_dataset("ag_news")
    else:
        raise ValueError("dataset_name must be 'sst2' or 'ag_news'")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_fn(batch):
        return tokenizer(batch['sentence'] if "sentence" in batch else batch['text'],
                         padding='max_length',
                         truncation=True,
                         max_length=max_length)

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Optionally take subset for quick experiments
    if subset_size:
        dataset['train'] = dataset['train'].select(range(subset_size))
        dataset['validation'] = dataset['validation'].select(range(subset_size))

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset['validation'], batch_size=batch_size)

    return train_loader, val_loader

# -----------------------------
# Metrics
# -----------------------------
def accuracy(logits, labels):
    """
    Compute classification accuracy.
    """
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean().item()

# -----------------------------
# Plotting
# -----------------------------
def plot_training_curve(train_losses, val_losses=None, title="Training Loss"):
    """
    Plot training (and optional validation) loss curve.
    """
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------
# Utility for counting trainable parameters
# -----------------------------
def count_trainable_params(model):
    """
    Return the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
