"""
Cross-Attention Heatmap Visualization Module.

Extracts real attention weights from the trained Transformer decoder layers
and overlays them onto the 7x7 spatial regions of the input image.

Methodological Notes:
- Each of the 49 tokens corresponds to a 32x32 receptive field in the 224x224 image.
- Attention weights are averaged across heads or inspected per head.
- Attention maps serve as diagnostic visualizations indicating feature relevance,
  not as definitive explanations of model reasoning.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset.preprocessing import get_image_transforms
from dataset.vocabulary import Vocabulary


def visualize_cross_attention(
    model,
    vocab: Vocabulary,
    image_path: str,
    output_path: str = None,
    device: torch.device = None,
    head_idx: int = None,  # None means average across all heads
    layer_idx: int = -1,   # Default to final decoder layer
    max_words: int = 25,
    token_ids: list = None,
    attention_maps: list = None,
) -> str:
    """
    Generates caption and plots cross-attention heatmaps for each predicted token.

    Args:
        model: Trained ImageCaptioner model.
        vocab: Vocabulary object.
        image_path: Path to input image file.
        output_path: Destination PNG file path.
        device: torch.device.
        head_idx: Attention head index (0..nhead-1), or None to average across all heads.
        layer_idx: Decoder layer index (-1 for final layer).
        max_words: Maximum number of word heatmaps to plot.
        token_ids: Optional pre-generated token IDs so caption matches top UI exactly.
        attention_maps: Optional pre-extracted attention maps corresponding to token_ids.

    Returns:
        output_path: Path where the visualization was saved.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    transform = get_image_transforms(split="val")

    # Load and preprocess image consistently with validation/test pipeline
    raw_img = Image.open(image_path).convert("RGB")
    raw_img_resized = raw_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 1. Use precomputed tokens/attention if provided, otherwise run greedy generation
    with torch.no_grad():
        if token_ids is None:
            token_ids, attention_maps = model.generate_caption_greedy(img_tensor, max_len=max_words)
        elif attention_maps is None:
            # Extract cross-attention weights for the provided token sequence (e.g. from Beam Search)
            from models.attention import generate_causal_mask
            clean_ids = [tid for tid in token_ids if tid != vocab.EOS_IDX]
            seq = [vocab.SOS_IDX] + clean_ids
            input_tokens = torch.tensor([seq], dtype=torch.long, device=device)
            visual_features = model.encoder(img_tensor)
            causal_mask = generate_causal_mask(input_tokens.size(1), device=device)
            _, cross_attns = model.decoder(
                tokens=input_tokens,
                memory=visual_features,
                self_attn_mask=causal_mask,
            )
            # cross_attns[layer_idx] has shape [1, nhead, seq_len, 49]
            # Position t in input_tokens predicts clean_ids[t]
            layer_attn = cross_attns[layer_idx][0].cpu()  # [nhead, seq_len, 49]
            attention_maps = [layer_attn[:, t, :] for t in range(len(clean_ids))]

    tokens = [vocab.i2w.get(tid, "<UNK>") for tid in token_ids if tid != vocab.EOS_IDX]
    num_tokens = min(len(tokens), len(attention_maps))

    if num_tokens == 0:
        print("[Warning] No tokens generated for attention visualization.")
        return ""

    # Subplot grid layout: 1 original image + num_tokens word maps
    total_plots = num_tokens + 1
    cols = min(4, total_plots)
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), dpi=250)
    axes = np.array(axes).flatten()

    # Subplot 0: Original image
    axes[0].imshow(raw_img_resized)
    axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    img_np = np.array(raw_img_resized) / 255.0

    # Subplots 1..N: Attention heatmaps per word
    for i in range(num_tokens):
        word = tokens[i]
        attn = attention_maps[i]  # [nhead, 49]

        if head_idx is not None:
            # Select specific head
            head_attn = attn[head_idx]  # [49]
            head_label = f"Head {head_idx}"
        else:
            # Average across heads
            head_attn = attn.mean(dim=0)  # [49]
            head_label = "Avg Heads"

        # Reshape to 7x7 spatial grid
        grid_7x7 = head_attn.view(7, 7).numpy()

        # Interpolate 7x7 grid to 224x224 resolution
        grid_tensor = torch.tensor(grid_7x7).unsqueeze(0).unsqueeze(0)  # [1, 1, 7, 7]
        upsampled = F.interpolate(grid_tensor, size=(224, 224), mode="bicubic", align_corners=False)
        heatmap = upsampled.squeeze().clamp(min=0.0).numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)

        ax = axes[i + 1]
        ax.imshow(img_np)
        im = ax.imshow(heatmap, cmap="jet", alpha=0.55)
        ax.set_title(f"Word {i+1}: '{word}'\n({head_label})", fontsize=10, fontweight="bold")
        ax.axis("off")

    # Hide unused axes
    for j in range(total_plots, len(axes)):
        axes[j].axis("off")

    layer_info = f"Decoder Layer {layer_idx if layer_idx >= 0 else model.decoder.num_layers + layer_idx + 1}"
    full_caption = " ".join(tokens)
    plt.suptitle(
        f"Visual Cross-Attention: \"{full_caption}\"\n({layer_info} | Diagnostic Receptive Field Focus)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    if output_path is None:
        os.makedirs("results/attention_visualizations", exist_ok=True)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"results/attention_visualizations/{img_name}_attention.png"

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Attention visualization saved: {output_path}")
    return output_path
