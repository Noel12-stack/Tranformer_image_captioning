"""
Plotting Loss Curves and Learning Rate Schedule.

Reads historical training records and generates publication-quality figures:
- results/training_loss.png
- results/validation_loss.png
- results/combined_loss.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_curves(
    history_path: str = "results/training_history.csv",
    output_dir: str = "results",
) -> None:
    """
    Plots training and validation loss curves from saved history CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(history_path):
        print(f"[Warning] History file not found: {history_path}. Skipping loss plots.")
        return

    df = pd.read_csv(history_path)
    epochs = df["epoch"].tolist()
    train_loss = df["train_loss"].tolist()
    val_loss = df["val_loss"].tolist()

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Training Loss Figure
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(epochs, train_loss, marker="o", color="#1f77b4", linewidth=2.0, label="Training Loss (Cross-Entropy)")
    ax.set_title("Training Loss vs. Epoch", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss (Ignoring <PAD>)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    train_loss_path = os.path.join(output_dir, "training_loss.png")
    fig.savefig(train_loss_path, dpi=300)
    plt.close(fig)

    # 2. Validation Loss Figure
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(epochs, val_loss, marker="s", color="#d62728", linewidth=2.0, label="Validation Loss")
    ax.set_title("Validation Loss vs. Epoch", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    val_loss_path = os.path.join(output_dir, "validation_loss.png")
    fig.savefig(val_loss_path, dpi=300)
    plt.close(fig)

    # 3. Combined Training & Validation Loss Figure
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(epochs, train_loss, marker="o", color="#1f77b4", linewidth=2.0, label="Training Loss")
    ax.plot(epochs, val_loss, marker="s", color="#d62728", linewidth=2.0, label="Validation Loss")
    # Annotate minimum validation loss
    min_val_idx = val_loss.index(min(val_loss))
    best_epoch = epochs[min_val_idx]
    best_val = val_loss[min_val_idx]
    ax.annotate(
        f"Best Val: {best_val:.3f} (Epoch {best_epoch})",
        xy=(best_epoch, best_val),
        xytext=(best_epoch, best_val + 0.3),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontweight="bold",
        fontsize=9,
    )
    ax.set_title("Training and Validation Convergence", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Supervised Cross-Entropy Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    combined_path = os.path.join(output_dir, "combined_loss.png")
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)

    print(f"Loss curves saved to {output_dir}:")
    print(f"  - {train_loss_path}")
    print(f"  - {val_loss_path}")
    print(f"  - {combined_path}")


if __name__ == "__main__":
    plot_loss_curves()
