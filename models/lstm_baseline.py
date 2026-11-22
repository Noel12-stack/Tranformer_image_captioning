"""
CNN-LSTM ('Show, Attend and Tell') Baseline Model for Image Captioning.

Implements the Show, Attend and Tell architecture (Xu et al., 2015) to compare
recurrent single-head additive (Bahdanau) visual attention against the proposed
Transformer Decoder with Multi-Head Scaled Dot-Product Cross-Attention.

Architecture (Show, Attend and Tell):
    1. Pretrained ResNet-50 CNN encoder extracting L=49 spatial vectors [B, 49, 256].
    2. MLP initialization of LSTM initial states (h_0, c_0) from mean spatial features.
    3. Bahdanau (additive) soft spatial attention over the 49 regions at every step t:
       e_{t,i} = v_a^T tanh(W_a a_i + U_a h_{t-1})
       alpha_{t} = softmax(e_t)
    4. Doubly-stochastic gating scalar beta_t = sigmoid(W_beta h_{t-1}):
       z_hat_t = beta_t * sum_i(alpha_{t,i} * a_i)
    5. LSTMCell transition receiving concatenated [word_embedding(y_{t-1}), z_hat_t].
    6. Deep output linear projection to vocabulary logits.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .cnn_encoder import ResNet50Encoder


class BahdanauAttention(nn.Module):
    """
    Soft Spatial Additive Attention Network from 'Show, Attend and Tell' (Xu et al., 2015).
    """

    def __init__(self, visual_dim: int = 256, hidden_dim: int = 256, attn_dim: int = 256):
        super(BahdanauAttention, self).__init__()
        self.W_a = nn.Linear(visual_dim, attn_dim)   # Projects 49 visual features
        self.U_a = nn.Linear(hidden_dim, attn_dim)   # Projects decoder hidden state h_{t-1}
        self.v_a = nn.Linear(attn_dim, 1)            # Computes unnormalized scalar score e_{t,i}
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        visual_features: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_features: [B, 49, visual_dim]
            hidden_state: [B, hidden_dim] (previous LSTM hidden state h_{t-1})

        Returns:
            context_vector: [B, visual_dim] weighted sum of spatial features
            alpha: [B, 49] spatial attention weights summing to 1
        """
        # [B, 49, attn_dim]
        vis_proj = self.W_a(visual_features)
        # [B, 1, attn_dim]
        hid_proj = self.U_a(hidden_state).unsqueeze(1)

        # Additive attention energy: [B, 49]
        energy = self.v_a(self.tanh(vis_proj + hid_proj)).squeeze(2)

        # Normalized spatial weights: [B, 49]
        alpha = self.softmax(energy)

        # Weighted sum over the 49 regions: [B, visual_dim]
        context_vector = (visual_features * alpha.unsqueeze(2)).sum(dim=1)

        return context_vector, alpha


class AttentionLSTMDecoder(nn.Module):
    """
    Recurrent LSTM Decoder with Bahdanau Spatial Attention ('Show, Attend and Tell').
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super(AttentionLSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(
            visual_dim=embed_dim,
            hidden_dim=hidden_dim,
            attn_dim=hidden_dim,
        )

        # MLP initialization of h_0 and c_0 from mean of spatial visual features
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.init_c = nn.Linear(embed_dim, hidden_dim)

        # Doubly stochastic gating scalar beta_t = sigmoid(f_beta(h_{t-1}))
        self.f_beta = nn.Linear(hidden_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

        # Step-wise LSTMCell taking concatenated [word_embedding, attended_visual_context]
        self.lstm_cell = nn.LSTMCell(embed_dim + embed_dim, hidden_dim, bias=True)

        self.dropout = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def init_hidden_state(self, visual_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden and cell states (h_0, c_0) from mean across 49 spatial regions.

        Args:
            visual_features: [B, 49, embed_dim]

        Returns:
            h_0: [B, hidden_dim]
            c_0: [B, hidden_dim]
        """
        mean_features = visual_features.mean(dim=1)  # [B, embed_dim]
        h_0 = torch.tanh(self.init_h(mean_features))
        c_0 = torch.tanh(self.init_c(mean_features))
        return h_0, c_0

    def forward(
        self,
        tokens: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing using step-wise spatial attention.

        Args:
            tokens: Input token sequence [B, T-1].
            visual_features: Projected spatial visual features [B, 49, embed_dim].

        Returns:
            logits: Output logits [B, T-1, vocab_size].
        """
        batch_size, seq_len = tokens.size()
        device = tokens.device

        embeddings = self.dropout(self.embedding(tokens))  # [B, T-1, embed_dim]
        h, c = self.init_hidden_state(visual_features)     # [B, hidden_dim]

        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)

        for t in range(seq_len):
            # 1. Attend over the 49 spatial image regions using previous hidden state h
            context, _ = self.attention(visual_features, h)  # [B, embed_dim]
            gate = self.sigmoid(self.f_beta(h))              # [B, embed_dim]
            gated_context = gate * context                   # [B, embed_dim]

            # 2. Concatenate word embedding at step t with attended visual context
            lstm_input = torch.cat([embeddings[:, t, :], gated_context], dim=1)  # [B, 2 * embed_dim]

            # 3. Recurrent LSTMCell update
            h, c = self.lstm_cell(lstm_input, (h, c))

            # 4. Project hidden state to vocabulary logits
            logits[:, t, :] = self.fc_out(self.dropout(h))

        return logits


class CNNLSTMBaseline(nn.Module):
    """
    Complete 'Show, Attend and Tell' (CNN + Bahdanau Spatial Attention + LSTM) architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
        pretrained_cnn: bool = True,
        freeze_cnn: bool = True,
    ):
        super(CNNLSTMBaseline, self).__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.encoder = ResNet50Encoder(
            d_model=embed_dim,
            pretrained=pretrained_cnn,
            freeze_backbone=freeze_cnn,
        )

        self.decoder = AttentionLSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )

    def forward(
        self,
        images_or_features: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        Accepts images [B, 3, 224, 224] or precomputed features [B, 49, 2048].
        """
        if images_or_features.dim() == 4:
            visual_features = self.encoder(images_or_features)  # [B, 49, embed_dim]
        else:
            visual_features = self.encoder.project_raw_features(images_or_features.float())
        decoder_input = captions[:, :-1]  # [B, T-1]
        logits = self.decoder(decoder_input, visual_features)
        return logits

    @torch.no_grad()
    def generate_caption_greedy(
        self,
        image: torch.Tensor,
        max_len: int = 25,
    ) -> List[int]:
        """
        Greedy caption generation using Show, Attend and Tell step-wise spatial attention.
        """
        self.eval()
        device = next(self.parameters()).device

        if image.dim() == 4 or (image.dim() == 3 and image.size(0) == 3):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
            visual_features = self.encoder(image)
        else:
            if image.dim() == 2:
                image = image.unsqueeze(0)
            image = image.to(device).float()
            visual_features = self.encoder.project_raw_features(image)

        h, c = self.decoder.init_hidden_state(visual_features)
        current_token = torch.tensor([self.sos_idx], dtype=torch.long, device=device)
        generated_tokens = []

        for _ in range(max_len):
            embed = self.decoder.embedding(current_token)          # [1, embed_dim]
            context, _ = self.decoder.attention(visual_features, h)
            gate = self.decoder.sigmoid(self.decoder.f_beta(h))
            gated_context = gate * context                         # [1, embed_dim]

            lstm_input = torch.cat([embed, gated_context], dim=1)  # [1, 2 * embed_dim]
            h, c = self.decoder.lstm_cell(lstm_input, (h, c))
            logits = self.decoder.fc_out(h)                        # [1, vocab_size]

            # Suppress PAD and SOS
            logits[0, self.pad_idx] = float("-inf")
            logits[0, self.sos_idx] = float("-inf")

            next_token = torch.argmax(logits, dim=-1).item()
            if next_token == self.eos_idx:
                break

            generated_tokens.append(next_token)
            current_token = torch.tensor([next_token], dtype=torch.long, device=device)

        return generated_tokens

    def get_parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
        }
