"""
Dataset Splitting Module with Leakage Prevention.

CRITICAL METHODOLOGICAL REQUIREMENT:
Splits the dataset strictly by unique image ID, never by individual captions.
In Flickr8k, each image has 5 distinct human captions. If captions of the same image
appeared in both training and test sets, the model could memorize visual semantics,
invalidating generalization evaluation (data leakage).

Output:
- train_images.txt
- val_images.txt
- test_images.txt
- splits_info.json
"""

import os
import json
import random
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict


def parse_captions_file(captions_path: str) -> Dict[str, List[str]]:
    """
    Parses Flickr8k captions file supporting both formats:
    1. CSV format: 'image,caption' (with header 'image,caption')
    2. Tab / hash format: '1000268201_693b08cb0e.jpg#0\tA child in a pink dress...'

    Args:
        captions_path: Path to the captions text/csv file.

    Returns:
        Dict mapping image_filename -> list of 5 captions.
    """
    image_to_captions = defaultdict(list)

    if not os.path.exists(captions_path):
        raise FileNotFoundError(f"Captions file not found at: {captions_path}")

    with open(captions_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Determine format
    first_line = lines[0].strip()
    is_csv = "," in first_line and ("image" in first_line.lower() or ".jpg," in first_line)

    start_idx = 1 if ("image" in first_line.lower() and "caption" in first_line.lower()) else 0

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue

        if is_csv:
            parts = line.split(",", 1)
            if len(parts) == 2:
                img_name = parts[0].strip()
                caption = parts[1].strip()
                image_to_captions[img_name].append(caption)
        else:
            # Tab or space separated
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(" ", 1)

            if len(parts) == 2:
                img_id = parts[0].strip()
                # Remove suffix like #0, #1
                if "#" in img_id:
                    img_name = img_id.split("#")[0].strip()
                else:
                    img_name = img_id
                caption = parts[1].strip()
                image_to_captions[img_name].append(caption)

    return image_to_captions


def create_splits_by_image(
    image_to_captions: Dict[str, List[str]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    output_dir: str = "data/Flickr8k/splits",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split images randomly into train, validation, and test sets.

    Args:
        image_to_captions: Dictionary mapping image filenames to lists of captions.
        train_ratio: Proportion of unique images for training (default: 0.80).
        val_ratio: Proportion of unique images for validation (default: 0.10).
        test_ratio: Proportion of unique images for testing (default: 0.10).
        seed: Random seed for exact reproducibility.
        output_dir: Directory where split text files and summary are saved.

    Returns:
        Tuple of (train_images, val_images, test_images)
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    all_images = sorted(list(image_to_captions.keys()))
    total_images = len(all_images)
    total_captions = sum(len(caps) for caps in image_to_captions.values())

    # Reproducible shuffle
    rng = random.Random(seed)
    shuffled_images = all_images.copy()
    rng.shuffle(shuffled_images)

    n_train = int(total_images * train_ratio)
    n_val = int(total_images * val_ratio)

    train_images = sorted(shuffled_images[:n_train])
    val_images = sorted(shuffled_images[n_train : n_train + n_val])
    test_images = sorted(shuffled_images[n_train + n_val :])

    # Count captions
    n_train_caps = sum(len(image_to_captions[img]) for img in train_images)
    n_val_caps = sum(len(image_to_captions[img]) for img in val_images)
    n_test_caps = sum(len(image_to_captions[img]) for img in test_images)

    # Save split files
    os.makedirs(output_dir, exist_ok=True)

    def write_list(filename, lst):
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            for item in lst:
                f.write(f"{item}\n")

    write_list("train_images.txt", train_images)
    write_list("val_images.txt", val_images)
    write_list("test_images.txt", test_images)

    summary = {
        "random_seed": seed,
        "total_unique_images": total_images,
        "total_captions": total_captions,
        "train_images": len(train_images),
        "train_captions": n_train_caps,
        "val_images": len(val_images),
        "val_captions": n_val_caps,
        "test_images": len(test_images),
        "test_captions": n_test_caps,
    }

    with open(os.path.join(output_dir, "splits_info.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("DATASET SPLIT SUMMARY (Zero Data Leakage by Image ID):")
    print(f"Total Unique Images : {total_images}")
    print(f"Total Captions      : {total_captions}")
    print(f"Training Set        : {len(train_images)} images ({n_train_caps} captions)")
    print(f"Validation Set      : {len(val_images)} images ({n_val_caps} captions)")
    print(f"Test Set            : {len(test_images)} images ({n_test_caps} captions)")
    print(f"Splits saved to     : {output_dir}")
    print("=" * 60)

    return train_images, val_images, test_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Flickr8k dataset by image ID")
    parser.add_argument("--captions_file", type=str, default="data/Flickr8k/captions.txt")
    parser.add_argument("--output_dir", type=str, default="data/Flickr8k/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.path.exists(args.captions_file):
        mapping = parse_captions_file(args.captions_file)
        create_splits_by_image(mapping, seed=args.seed, output_dir=args.output_dir)
    else:
        print(f"Captions file not found at: {args.captions_file}")
        print("Please download Flickr8k or run with synthetic dataset.")
