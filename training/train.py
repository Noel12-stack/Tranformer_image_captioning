"""
Main Training Entry Point for Image Captioning.

Supports:
- CNN-Transformer (primary model)
- CNN-LSTM (baseline model)
- Full Flickr8k dataset or synthetic dataset for CPU demonstration
- Reproducible random seed
"""

import os
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.vocabulary import Vocabulary
from dataset.preprocessing import get_image_transforms
from dataset.split_dataset import parse_captions_file, create_splits_by_image
from dataset.dataset import Flickr8kDataset, CaptionCollate, create_synthetic_dataset
from models.image_captioner import ImageCaptioner
from models.lstm_baseline import CNNLSTMBaseline
from training.trainer import CaptionTrainer


def set_seed(seed: int = 42):
    """Ensure exact reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    config_path: str = "configs/config.yaml",
    arch: str = "transformer",
    use_synthetic: bool = False,
    override_epochs: int = None,
    override_batch_size: int = None,
):
    # 1. Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device Selection] Using: {device}")

    # 2. Dataset Preparation
    data_cfg = config.get("dataset", {})
    images_dir = data_cfg.get("images_dir", "data/Flickr8k/Images")
    captions_file = data_cfg.get("captions_file", "data/Flickr8k/captions.txt")
    splits_dir = data_cfg.get("splits_dir", "data/Flickr8k/splits")

    if use_synthetic or not os.path.exists(captions_file):
        print("\n[INFO] Using synthetic dataset pipeline for fast testing & CPU validation...")
        images_dir, captions_file = create_synthetic_dataset(output_dir="data/synthetic")
        splits_dir = "data/synthetic/splits"

    # Parse and split images (Leakage Prevention: by Image ID)
    mapping = parse_captions_file(captions_file)
    train_images, val_images, test_images = create_splits_by_image(
        mapping,
        train_ratio=data_cfg.get("train_split", 0.80),
        val_ratio=data_cfg.get("val_split", 0.10),
        test_ratio=data_cfg.get("test_split", 0.10),
        seed=seed,
        output_dir=splits_dir,
    )

    # Check if precomputed features exist for high-speed CPU training
    features_cache_path = os.path.join(data_cfg.get("data_dir", "data/Flickr8k"), "features_cache.pt")
    features_cache = None
    if os.path.exists(features_cache_path) and not use_synthetic:
        print(f"\n[INFO] Loading precomputed CNN features from: {features_cache_path}...")
        features_cache = torch.load(features_cache_path, map_location="cpu")
        print(f"Loaded {len(features_cache)} cached image features.")
        # Restrict image lists to cached images
        train_images = [img for img in train_images if img in features_cache]
        val_images = [img for img in val_images if img in features_cache]
        test_images = [img for img in test_images if img in features_cache]
        print(f"Filtered to cached split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images")

    # 3. Build Vocabulary exclusively from training captions
    print("\nBuilding vocabulary from training captions...")
    train_captions = [cap for img in train_images for cap in mapping[img]]
    vocab = Vocabulary(min_freq=data_cfg.get("min_freq", 2))
    vocab.build_vocabulary(train_captions)

    os.makedirs("checkpoints", exist_ok=True)
    vocab_path = "checkpoints/vocab.json"
    vocab.save(vocab_path)
    print(f"Vocabulary successfully built: {len(vocab)} unique tokens (saved to {vocab_path})")

    # 4. Construct Datasets and DataLoaders
    batch_size = override_batch_size or config["training"].get("batch_size", 16)
    max_seq_len = data_cfg.get("max_seq_length", 30)

    train_dataset = Flickr8kDataset(
        images_dir=images_dir,
        image_names=train_images,
        image_to_captions=mapping,
        vocab=vocab,
        transform=get_image_transforms(split="train"),
        max_seq_length=max_seq_len,
        features_cache=features_cache,
    )
    val_dataset = Flickr8kDataset(
        images_dir=images_dir,
        image_names=val_images,
        image_to_captions=mapping,
        vocab=vocab,
        transform=get_image_transforms(split="val"),
        max_seq_length=max_seq_len,
        features_cache=features_cache,
    )

    collate_fn = CaptionCollate(pad_idx=vocab.PAD_IDX)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 5. Build Model
    epochs = override_epochs or config["training"].get("epochs", 15)
    lr = float(config["training"].get("learning_rate", 3e-4))
    weight_decay = float(config["training"].get("weight_decay", 1e-5))
    grad_clip = float(config["training"].get("grad_clip", 5.0))

    if arch == "lstm":
        base_cfg = config.get("baseline", {})
        model = CNNLSTMBaseline(
            vocab_size=len(vocab),
            embed_dim=base_cfg.get("embed_dim", 256),
            hidden_dim=base_cfg.get("hidden_dim", 256),
            num_layers=base_cfg.get("num_layers", 1),
            dropout=base_cfg.get("dropout", 0.1),
            pad_idx=vocab.PAD_IDX,
            sos_idx=vocab.SOS_IDX,
            eos_idx=vocab.EOS_IDX,
            pretrained_cnn=True,
            freeze_cnn=True,
        )
    else:
        m_cfg = config.get("model", {})
        dec_cfg = m_cfg.get("decoder", {})
        model = ImageCaptioner(
            vocab_size=len(vocab),
            d_model=dec_cfg.get("d_model", 256),
            nhead=dec_cfg.get("nhead", 4),
            num_layers=dec_cfg.get("num_layers", 2),
            dim_feedforward=dec_cfg.get("dim_feedforward", 512),
            dropout=dec_cfg.get("dropout", 0.1),
            pad_idx=vocab.PAD_IDX,
            sos_idx=vocab.SOS_IDX,
            eos_idx=vocab.EOS_IDX,
            pretrained_cnn=True,
            freeze_cnn=True,
        )

    model.to(device)

    # Parameter Inspection
    params_info = model.get_parameter_count()
    print("\n" + "-" * 50)
    print(f"Model Architecture: {arch.upper()}")
    print(f"Total Parameters     : {params_info['total_parameters']:,}")
    print(f"Trainable Parameters : {params_info['trainable_parameters']:,}")
    print(f"Frozen CNN Parameters: {params_info['frozen_parameters']:,}")
    print("-" * 50)

    # 6. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_IDX)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # 7. Execute Training
    trainer = CaptionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        grad_clip=grad_clip,
        scheduler=scheduler,
        checkpoint_dir="checkpoints",
        results_dir="results",
        early_stopping_patience=config["training"].get("early_stopping_patience", 4),
        config=config,
        arch_type=arch,
    )

    history_df = trainer.fit()
    return history_df, vocab, splits_dir, images_dir, captions_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--arch", type=str, default="transformer", choices=["transformer", "lstm"])
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        arch=args.arch,
        use_synthetic=args.synthetic,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
    )
