"""
Caption Generation Module for Model Evaluation.

Generates captions for a set of image files using:
1. Greedy autoregressive decoding
2. Beam Search decoding (configurable beam width)
"""

import os
from typing import Dict, List, Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms
from dataset.preprocessing import get_image_transforms, clean_caption, tokenize_caption
from dataset.vocabulary import Vocabulary


def generate_predictions_for_split(
    model,
    vocab: Vocabulary,
    image_names: List[str],
    images_dir: str,
    device: torch.device,
    method: str = "greedy",
    beam_size: int = 3,
    max_len: int = 25,
) -> Dict[str, str]:
    """
    Generates captions for all images in a given list.

    Args:
        model: Trained ImageCaptioner or CNNLSTMBaseline model.
        vocab: Vocabulary object.
        image_names: List of image file names.
        images_dir: Directory where image files are stored.
        device: CPU or CUDA device.
        method: 'greedy' or 'beam'
        beam_size: Beam width when method == 'beam'.
        max_len: Maximum caption length.

    Returns:
        Dict mapping image_name -> generated caption string.
    """
    model.eval()
    model.to(device)
    transform = get_image_transforms(split="val")
    predictions: Dict[str, str] = {}

    features_cache_path = os.path.join(os.path.dirname(images_dir), "features_cache.pt")
    features_cache = None
    if os.path.exists(features_cache_path):
        features_cache = torch.load(features_cache_path, map_location="cpu")

    with torch.no_grad():
        for img_name in image_names:
            if features_cache is not None and img_name in features_cache:
                img_tensor = features_cache[img_name].unsqueeze(0).to(device)
            else:
                img_path = os.path.join(images_dir, img_name)
                if not os.path.exists(img_path):
                    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
                else:
                    img = Image.open(img_path).convert("RGB")

                img_tensor = transform(img).unsqueeze(0).to(device)

            if method == "beam" and hasattr(model, "generate_caption_beam_search"):
                token_ids = model.generate_caption_beam_search(
                    img_tensor, beam_size=beam_size, max_len=max_len
                )
            else:
                if hasattr(model, "generate_caption_greedy"):
                    res = model.generate_caption_greedy(img_tensor, max_len=max_len)
                    # If model returns (tokens, attn_maps) unpack tokens
                    token_ids = res[0] if isinstance(res, tuple) else res
                else:
                    raise AttributeError("Model lacks generate_caption_greedy method")

            caption = vocab.decode(token_ids, skip_special_tokens=True)
            predictions[img_name] = caption

    return predictions
