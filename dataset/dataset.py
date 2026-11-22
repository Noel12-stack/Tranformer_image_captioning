"""
PyTorch Dataset and DataLoader Pipeline for Image Captioning.

Features:
1. Flickr8kDataset: Pairs images with each human-written caption.
2. CaptionCollate: Dynamically pads caption sequences to the max length in each batch.
3. Synthetic Dataset Generator: Generates a small synthetic dataset for testing,
   debugging, and running the entire pipeline when Flickr8k is not yet present.
"""

import os
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from .vocabulary import Vocabulary
from .preprocessing import get_image_transforms


class Flickr8kDataset(Dataset):
    """
    PyTorch Dataset for Image Captioning.

    Each item consists of:
    - Transformed RGB image tensor: [3, 224, 224]
    - Numericalized caption tensor: [<SOS>, w1, ..., wn, <EOS>]
    - Image filename string
    """

    def __init__(
        self,
        images_dir: str,
        image_names: List[str],
        image_to_captions: Dict[str, List[str]],
        vocab: Vocabulary,
        transform=None,
        max_seq_length: int = 30,
        features_cache: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = transform or get_image_transforms(split="val")
        self.max_seq_length = max_seq_length
        self.features_cache = features_cache

        # Flatten into (image_name, single_caption) pairs
        self.samples: List[Tuple[str, str]] = []
        for img_name in image_names:
            if img_name in image_to_captions:
                for caption in image_to_captions[img_name]:
                    self.samples.append((img_name, caption))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_name, caption = self.samples[idx]

        # Use precomputed raw features if available
        if self.features_cache is not None and img_name in self.features_cache:
            image_data = self.features_cache[img_name]
        else:
            img_path = os.path.join(self.images_dir, img_name)
            if not os.path.exists(img_path):
                # Create a placeholder image if image file is missing
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            else:
                image = Image.open(img_path).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)
            image_data = image

        # Numericalize: returns [<SOS>, token_ids..., <EOS>]
        tokens = self.vocab.numericalize(caption, max_len=self.max_seq_length)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        return image_data, tokens_tensor, img_name


class CaptionCollate:
    """
    Collate function to dynamically pad variable-length caption sequences in a batch.
    """

    def __init__(self, pad_idx: int = 0):
        self.pad_idx = pad_idx

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        # Separate images, tokens, and image names
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        img_names = [item[2] for item in batch]

        # Stack image tensors into [B, 3, 224, 224]
        images_tensor = torch.stack(images, dim=0)

        # Pad caption sequences to maximum length in this batch: [B, T_batch]
        captions_padded = torch.nn.utils.rnn.pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx,
        )

        return images_tensor, captions_padded, img_names


def create_synthetic_dataset(
    output_dir: str = "data/synthetic",
    num_images: int = 24,
    num_captions_per_image: int = 5,
) -> Tuple[str, str]:
    """
    Creates a small, fully functional synthetic dataset with distinct colored shapes
    and paired natural language captions.

    Enables testing the complete training, evaluation, and visualization pipeline.

    Args:
        output_dir: Directory where synthetic images and captions.txt will be saved.
        num_images: Number of synthetic images to create.
        num_captions_per_image: Number of captions per image.

    Returns:
        Tuple of (images_directory_path, captions_file_path)
    """
    images_dir = os.path.join(output_dir, "Images")
    os.makedirs(images_dir, exist_ok=True)
    captions_file = os.path.join(output_dir, "captions.txt")

    colors = [
        ("red", (230, 40, 40)),
        ("blue", (40, 100, 230)),
        ("green", (40, 180, 50)),
        ("yellow", (230, 210, 40)),
        ("purple", (160, 40, 200)),
        ("orange", (230, 120, 30)),
    ]
    shapes = ["circle", "square", "triangle", "star"]
    actions = ["sitting in grass", "resting on a white floor", "placed against a wall", "floating in space"]

    captions_lines = ["image,caption\n"]

    for i in range(num_images):
        img_filename = f"synthetic_{i:04d}.jpg"
        color_name, rgb = colors[i % len(colors)]
        shape_name = shapes[(i // len(colors)) % len(shapes)]
        action = actions[i % len(actions)]

        # Generate image
        img = Image.new("RGB", (224, 224), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)

        # Draw shape
        cx, cy, r = 112, 112, 50
        if shape_name == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb, outline=(0, 0, 0))
        elif shape_name == "square":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb, outline=(0, 0, 0))
        elif shape_name == "triangle":
            draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgb, outline=(0, 0, 0))
        else:
            # Diamond / star
            draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=rgb, outline=(0, 0, 0))

        img.save(os.path.join(images_dir, img_filename), quality=95)

        # 5 captions for this image
        caps = [
            f"a {color_name} {shape_name} is {action}",
            f"there is a bright {color_name} {shape_name} in the center",
            f"an image displaying a {color_name} {shape_name} clearly",
            f"a vivid {color_name} {shape_name} {action}",
            f"close up view of a {color_name} {shape_name}",
        ]

        for cap in caps[:num_captions_per_image]:
            captions_lines.append(f"{img_filename},{cap}\n")

    with open(captions_file, "w", encoding="utf-8") as f:
        f.writelines(captions_lines)

    return images_dir, captions_file
