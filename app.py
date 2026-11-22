"""
Streamlit Web Demonstration Interface.

Attention-Based Image Caption Generation Using CNN and Transformer Architectures.
Provides interactive UI to:
1. Upload an arbitrary image (JPG, PNG).
2. Generate a natural language caption using our trained CNN-Transformer model.
3. Choose decoding algorithm (Greedy vs. Beam Search).
4. Visualize actual multi-head cross-attention spatial heatmaps across image regions.
"""

import os
import importlib
import torch
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

from dataset.vocabulary import Vocabulary
from dataset.preprocessing import get_image_transforms
from models.image_captioner import ImageCaptioner
import models.lstm_baseline as lstm_mod
importlib.reload(lstm_mod)
from models.lstm_baseline import CNNLSTMBaseline
import visualization.visualize_attention as vis_attn_mod
importlib.reload(vis_attn_mod)
from visualization.visualize_attention import visualize_cross_attention


# Page configuration
st.set_page_config(
    page_title="Image Captioning | CNN + Transformer",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Attention-Based Image Caption Generation")
st.markdown(
    "**College Seminar Project (CSE)** | Pretrained ResNet-50 CNN Encoder + "
    "Transformer Decoder with Multi-Head Cross-Attention (Compared against CNN-LSTM Baseline)"
)
st.write("---")

# Checkpoint paths
CHECKPOINT_PATH = "checkpoints/best_transformer.pt"
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = "checkpoints/best_model.pt"
LSTM_CHECKPOINT_PATH = "checkpoints/best_lstm.pt"
VOCAB_PATH = "checkpoints/vocab.json"


@st.cache_resource
def load_captioning_system(ckpt_mtime: float = 0.0):
    """Load Transformer and LSTM model checkpoints and vocabulary onto device."""
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(VOCAB_PATH):
        return None, None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(VOCAB_PATH)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint.get("config", {})
    m_cfg = config.get("model", {})
    dec_cfg = m_cfg.get("decoder", {})

    model = ImageCaptioner(
        vocab_size=len(vocab),
        d_model=dec_cfg.get("d_model", 256),
        nhead=dec_cfg.get("nhead", 4),
        num_layers=dec_cfg.get("num_layers", 2),
        dim_feedforward=dec_cfg.get("dim_feedforward", 1024),
        dropout=0.0,
        pad_idx=vocab.PAD_IDX,
        sos_idx=vocab.SOS_IDX,
        eos_idx=vocab.EOS_IDX,
        pretrained_cnn=True,
        freeze_cnn=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    lstm_model = None
    if os.path.exists(LSTM_CHECKPOINT_PATH):
        try:
            lstm_ckpt = torch.load(LSTM_CHECKPOINT_PATH, map_location=device)
            lstm_cfg = lstm_ckpt.get("config", {}).get("model", {}).get("decoder", {})
            lstm_model = CNNLSTMBaseline(
                vocab_size=len(vocab),
                embed_dim=lstm_cfg.get("d_model", 256),
                hidden_dim=lstm_cfg.get("d_model", 256),
                num_layers=1,
                dropout=0.0,
                pad_idx=vocab.PAD_IDX,
                sos_idx=vocab.SOS_IDX,
                eos_idx=vocab.EOS_IDX,
                pretrained_cnn=True,
                freeze_cnn=True,
            )
            lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
            lstm_model.to(device)
            lstm_model.eval()
        except Exception:
            lstm_model = None

    return model, lstm_model, vocab, device


ckpt_mtime = (
    (os.path.getmtime(CHECKPOINT_PATH) if os.path.exists(CHECKPOINT_PATH) else 0.0)
    + (os.path.getmtime(LSTM_CHECKPOINT_PATH) if os.path.exists(LSTM_CHECKPOINT_PATH) else 0.0)
)
model, lstm_model, vocab, device = load_captioning_system(ckpt_mtime)

# Sidebar options
st.sidebar.header("⚙️ Model Configuration")
decoding_mode = st.sidebar.radio(
    "Decoding Strategy:",
    options=["Greedy Search", "Beam Search"],
    help="Greedy selects highest probability token; Beam Search explores top-k sequences.",
)
beam_width = 3
if decoding_mode == "Beam Search":
    beam_width = st.sidebar.slider("Beam Size (k)", min_value=2, max_value=5, value=3)

show_attention = st.sidebar.checkbox(
    "Show Cross-Attention Heatmaps",
    value=True,
    help="Overlays the actual attention weights from the Transformer decoder across the 7x7 spatial grid.",
)

# Main UI layout
if model is None:
    st.warning(
        "⚠️ Trained model checkpoint not detected at `checkpoints/best_model.pt`.\n\n"
        "Please run model training first using:\n"
        "```bash\n"
        "python training/train.py --arch transformer\n"
        "```\n"
        "Or for a quick demonstration with synthetic data:\n"
        "```bash\n"
        "python training/train.py --arch transformer --synthetic --epochs 10\n"
        "```"
    )
else:
    param_info = model.get_parameter_count()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Encoder Backbone**: ResNet-50 (Frozen)")
    st.sidebar.markdown(f"**Decoder Layers**: {model.decoder.num_layers}")
    st.sidebar.markdown(f"**Attention Heads**: {model.decoder.layers[0].cross_attn.nhead}")
    st.sidebar.markdown(f"**Model Dimension ($d_{{model}}$)**: {model.d_model}")
    st.sidebar.markdown(f"**Trainable Parameters**: {param_info['trainable_parameters']:,}")
    st.sidebar.markdown(f"**Total Parameters**: {param_info['total_parameters']:,}")

    st.subheader("1. Select or Upload an Image")
    input_method = st.radio(
        "Image Source:",
        options=["Upload Custom Image (.jpg / .png)", "Pick from Held-Out Flickr8k Test Set"],
        horizontal=True,
    )

    image = None
    selected_test_img_name = None

    if input_method == "Upload Custom Image (.jpg / .png)":
        uploaded_file = st.file_uploader(
            "Choose an image for caption generation...",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        test_split_file = "data/Flickr8k/splits/test_images.txt"
        img_dir = "data/Flickr8k/Images"
        if os.path.exists(test_split_file) and os.path.exists(img_dir):
            with open(test_split_file, "r", encoding="utf-8") as f:
                test_imgs = [line.strip() for line in f if line.strip()]
            selected_test_img_name = st.selectbox(
                f"Select from {len(test_imgs)} unseen Flickr8k test images:",
                options=test_imgs[:100],
            )
            if selected_test_img_name:
                img_path = os.path.join(img_dir, selected_test_img_name)
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert("RGB")

    if image is not None:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(image, caption="Input Image", use_container_width=True)

        with col2:
            st.subheader("Generated Captions (Model Comparison)")
            with st.spinner("Extracting ResNet-50 features & decoding..."):
                transform = get_image_transforms(split="val")
                img_tensor = transform(image).unsqueeze(0).to(device)

                if decoding_mode == "Beam Search":
                    token_ids = model.generate_caption_beam_search(
                        img_tensor, beam_size=beam_width, max_len=25
                    )
                    attn_maps = None
                else:
                    token_ids, attn_maps = model.generate_caption_greedy(img_tensor, max_len=25)

                caption = vocab.decode(token_ids, skip_special_tokens=True)

                lstm_caption = None
                if lstm_model is not None:
                    lstm_ids = lstm_model.generate_caption_greedy(img_tensor, max_len=25)
                    lstm_caption = vocab.decode(lstm_ids, skip_special_tokens=True)

            st.markdown("#### 🌟 Proposed: CNN-Transformer (Multi-Head Cross-Attention)")
            st.success(f"**\"{caption.capitalize()}\"**")

            if lstm_caption is not None:
                st.markdown("#### 🔹 Baseline: Show, Attend and Tell (CNN + Bahdanau Attention + LSTM)")
                st.info(f"**\"{lstm_caption.capitalize()}\"**")

            st.markdown(
                f"- **Transformer Decoding**: {decoding_mode} ({len(caption.split())} words)\n"
                f"- **Vocabulary Size**: {len(vocab):,} words\n"
                f"- **Inference Device**: `{device}`"
            )

        # Attention visualization section
        if show_attention:
            st.write("---")
            st.subheader("🔍 Spatial Cross-Attention Analysis")
            st.markdown(
                "*Diagnostic visualization displaying how the Transformer decoder attends "
                "to different 7×7 spatial regions for each generated token.*"
            )
            with st.spinner("Generating cross-attention heatmaps..."):
                os.makedirs("results", exist_ok=True)
                temp_img_path = "results/temp_uploaded_img.png"
                image.save(temp_img_path)
                attn_fig_path = "results/temp_attn_vis.png"
                visualize_cross_attention(
                    model=model,
                    vocab=vocab,
                    image_path=temp_img_path,
                    output_path=attn_fig_path,
                    device=device,
                    token_ids=token_ids,
                    attention_maps=attn_maps,
                )
                if os.path.exists(attn_fig_path):
                    st.image(attn_fig_path, use_container_width=True)
                    # Clean up temp files
                    try:
                        os.remove(temp_img_path)
                        os.remove(attn_fig_path)
                    except Exception:
                        pass
