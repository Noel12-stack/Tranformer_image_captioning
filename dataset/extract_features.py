"""
Feature Precomputation Module for ResNet-50 CNN Encoder.

As specified in Project Objective 4.1:
'For efficiency, provide an option to precompute and save CNN image features.
 The precomputed features must be associated with the correct image IDs.
 Make sure the implementation does not accidentally calculate features using
 different preprocessing during training and inference.'

This script:
1. Loads the pretrained frozen ResNet-50 backbone.
2. Applies the standard deterministic validation/inference transform (Resize 224, CenterCrop 224, ImageNet Normalize).
3. Extracts raw [49, 2048] feature vectors from layer4.
4. Saves them in float16 to disk, indexed by image filename.
"""

import os
import argparse
from typing import List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from models.cnn_encoder import ResNet50Encoder
from dataset.preprocessing import get_image_transforms


class ImagePathDataset(Dataset):
    def __init__(self, image_names: List[str], images_dir: str, transform):
        self.image_names = image_names
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = os.path.join(self.images_dir, name)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        tensor = self.transform(img)
        return name, tensor


def precompute_features(
    image_names: List[str],
    images_dir: str = "data/Flickr8k/Images",
    output_cache_file: str = "data/Flickr8k/features_cache.pt",
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Extracts and caches [49, 2048] raw CNN feature vectors.
    """
    os.makedirs(os.path.dirname(output_cache_file), exist_ok=True)
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Precomputing features on: {device_obj} for {len(image_names)} images...")

    transform = get_image_transforms(split="val")
    encoder = ResNet50Encoder(pretrained=True, freeze_backbone=True)
    encoder.to(device_obj)
    encoder.eval()

    ds = ImagePathDataset(image_names, images_dir, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    cache: Dict[str, torch.Tensor] = {}
    total_batches = len(loader)

    with torch.no_grad():
        for i, (names, imgs) in enumerate(loader):
            imgs = imgs.to(device_obj)
            # Shape: [B, 49, 2048]
            raw_feats = encoder.extract_raw_features(imgs).cpu().to(torch.float16)
            for name, feat in zip(names, raw_feats):
                cache[name] = feat

            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                pct = (i + 1) / total_batches * 100
                print(f"  Processed [{i+1}/{total_batches}] batches ({pct:.1f}%)", flush=True)

    torch.save(cache, output_cache_file)
    print(f"Features saved successfully to: {output_cache_file} ({len(cache)} images)")
    return cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", type=str, default="data/Flickr8k/splits")
    parser.add_argument("--images_dir", type=str, default="data/Flickr8k/Images")
    parser.add_argument("--output_file", type=str, default="data/Flickr8k/features_cache.pt")
    parser.add_argument("--max_images", type=int, default=1000)
    args = parser.parse_args()

    # Read train, val, and test split images proportionally
    def read_split(filename):
        p = os.path.join(args.splits_dir, filename)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    train_list = read_split("train_images.txt")
    val_list = read_split("val_images.txt")
    test_list = read_split("test_images.txt")

    if args.max_images > 0:
        n_train = int(args.max_images * 0.80)
        n_val = int(args.max_images * 0.10)
        n_test = args.max_images - n_train - n_val
        all_imgs = train_list[:n_train] + val_list[:n_val] + test_list[:n_test]
    else:
        all_imgs = train_list + val_list + test_list

    precompute_features(all_imgs, args.images_dir, args.output_file)
