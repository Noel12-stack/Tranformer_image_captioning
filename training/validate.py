"""
Validation Loop for Image Captioning.

Computes mean Cross-Entropy loss over the validation split with teacher forcing,
ignoring <PAD> tokens in target sequences.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluates model on validation dataloader.

    Args:
        model: Model under evaluation (ImageCaptioner or CNNLSTMBaseline).
        val_loader: DataLoader yielding (images, captions, image_names).
        criterion: CrossEntropyLoss with ignore_index=pad_idx.
        device: torch.device (cpu or cuda).

    Returns:
        float: Mean validation loss per token.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, captions, _ in val_loader:
            images = images.to(device)
            captions = captions.to(device)

            # Targets are shifted by 1 position to the right relative to decoder input
            targets = captions[:, 1:].contiguous()  # [B, T-1]

            # Forward pass produces logits for [B, T-1, vocab_size]
            logits = model(images, captions)
            if isinstance(logits, tuple):
                logits = logits[0]  # Transformer returns (logits, cross_attns)

            # Reshape for cross entropy loss
            # logits: [B * (T-1), vocab_size], targets: [B * (T-1)]
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)
