"""
Training Loop and Checkpoint Manager for Image Captioning.

Manages:
- Supervised teacher-forcing training loop
- Gradient clipping and Adam optimizer updates
- Learning rate scheduling
- Early stopping based on validation loss
- Persistent checkpointing and historical training loss logging to CSV
"""

import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any
from .validate import validate_epoch


class CaptionTrainer:
    """
    Orchestrates training, validation, early stopping, and checkpoint management.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epochs: int = 15,
        grad_clip: float = 5.0,
        scheduler: Any = None,
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "results",
        early_stopping_patience: int = 4,
        config: Dict = None,
        arch_type: str = "transformer",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.early_stopping_patience = early_stopping_patience
        self.config = config or {}
        self.arch_type = arch_type

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> float:
        """Runs a single epoch of training."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, captions, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)

            targets = captions[:, 1:].contiguous()  # [B, T-1]

            self.optimizer.zero_grad()

            logits = self.model(images, captions)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss.backward()

            # Clip gradients to prevent gradient explosion
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def fit(self) -> pd.DataFrame:
        """
        Executes full training across epochs with validation and early stopping.

        Returns:
            pd.DataFrame: History dataframe with epoch, train_loss, val_loss, lr, time.
        """
        print("\n" + "=" * 70)
        print(f"STARTING TRAINING: {self.arch_type.upper()} ({self.epochs} epochs)")
        print("=" * 70)

        start_training_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # 1. Train epoch
            train_loss = self.train_epoch(epoch)

            # 2. Validate epoch
            val_loss = validate_epoch(
                model=self.model,
                val_loader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
            )

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler if present
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "learning_rate": current_lr,
                "epoch_time_seconds": round(epoch_time, 2),
            }
            self.history.append(record)

            is_best = val_loss < self.best_val_loss
            star = " (* Best Checkpoint Saved *)" if is_best else ""
            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s{star}"
            )

            # Checkpoint management
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_path = os.path.join(self.checkpoint_dir, f"best_{self.arch_type}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "arch": self.arch_type,
                        "config": self.config,
                    },
                    best_path,
                )
                # Also save standard best_model.pt if transformer
                if self.arch_type == "transformer":
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                            "train_loss": train_loss,
                            "arch": self.arch_type,
                            "config": self.config,
                        },
                        os.path.join(self.checkpoint_dir, "best_model.pt"),
                    )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n[INFO] Early stopping triggered at epoch {epoch} (Patience: {self.early_stopping_patience})")
                    break

        total_time = time.time() - start_training_time
        print("=" * 70)
        print(f"Training completed in {total_time:.1f} seconds. Best Val Loss: {self.best_val_loss:.4f}")
        print("=" * 70 + "\n")

        # Save training history to CSV
        df_history = pd.DataFrame(self.history)
        history_csv_path = os.path.join(self.results_dir, f"training_history_{self.arch_type}.csv")
        df_history.to_csv(history_csv_path, index=False)
        if self.arch_type == "transformer":
            df_history.to_csv(os.path.join(self.results_dir, "training_history.csv"), index=False)

        return df_history
