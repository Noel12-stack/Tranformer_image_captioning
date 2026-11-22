"""
Image Preprocessing and Caption Normalization Pipeline.

Ensures exact consistency between training and inference preprocessing:
1. Standard ImageNet mean and standard deviation normalization.
2. 224x224 RGB image resizing.
3. Lowercasing, alphanumeric cleaning, and punctuation stripping for captions.
"""

import re
import torchvision.transforms as transforms
from typing import List


# Standard ImageNet normalization parameters required by Torchvision pretrained ResNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_transforms(split: str = "train", image_size: int = 224) -> transforms.Compose:
    """
    Returns image transformation pipeline for train/val/test/inference.

    Args:
        split: 'train', 'val', 'test', or 'inference'
        image_size: target spatial resolution (224 for ResNet-50)

    Returns:
        torchvision.transforms.Compose
    """
    if split == "train":
        # Data augmentation during training
        return transforms.Compose([
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Deterministic preprocessing for validation, test, and live inference
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def clean_caption(caption: str) -> str:
    """
    Normalize caption text:
    - Lowercase all characters
    - Remove punctuation and special symbols
    - Collapse extra whitespace

    Args:
        caption: raw caption string

    Returns:
        cleaned caption string
    """
    caption = caption.strip().lower()
    # Replace punctuation with spaces while keeping alphanumeric characters
    caption = re.sub(r"[^a-zA-Z0-9\s]", " ", caption)
    # Collapse multiple whitespaces into a single space
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def tokenize_caption(caption: str) -> List[str]:
    """
    Tokenizes a normalized caption string into a list of word tokens.

    Args:
        caption: cleaned caption text

    Returns:
        List of string tokens
    """
    cleaned = clean_caption(caption)
    tokens = cleaned.split()
    return tokens
