"""
Transformer Decoder Module.

Implements the multi-layer Transformer decoder that consumes text token sequences
and visual features from the ResNet-50 CNN encoder to predict the next word.

Architecture per Decoder Layer:
    1. Masked Multi-Head Self-Attention (text attending to previous text)
    2. Add & LayerNorm
    3. Multi-Head Visual Cross-Attention (text queries attending to CNN spatial features)
    4. Add & LayerNorm
    5. Position-wise Feed-Forward Network (FFN)
    6. Add & LayerNorm

Tensor Flow:
    - Text token IDs: [B, T]
    - Word Embedding: [B, T, d_model]
    - Add Positional Encoding: [B, T, d_model]
    - CNN Visual Features: [B, 49, d_model]
    - Final Output Logits: [B, T, vocab_size]
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from .attention import MultiHeadAttention, PositionalEncoding


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of the Transformer Decoder.
    Contains masked self-attention, visual cross-attention, and a position-wise feed-forward network.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(TransformerDecoderLayer, self).__init__()

        # Sublayer 1: Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        # Sublayer 2: Visual Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

        # Sublayer 3: Position-wise Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward computation through one decoder layer.

        Args:
            x (torch.Tensor): Decoder input representations [B, T, d_model].
            memory (torch.Tensor): Visual features from CNN encoder [B, 49, d_model].
            self_attn_mask (torch.Tensor, optional): Causal mask [1, 1, T, T].
            cross_attn_mask (torch.Tensor, optional): Optional padding mask for memory.

        Returns:
            x (torch.Tensor): Updated decoder representation [B, T, d_model].
            cross_attn_weights (torch.Tensor): Cross-attention map [B, nhead, T, 49].
        """
        # --- 1. Masked Self-Attention (Query, Key, Value from decoder input x) ---
        residual = x
        self_out, _ = self.self_attn(query=x, key=x, value=x, mask=self_attn_mask)
        x = self.norm1(residual + self.dropout1(self_out))

        # --- 2. Visual Cross-Attention ---
        # Query from text tokens (x); Key and Value from CNN features (memory)
        residual = x
        cross_out, cross_attn_weights = self.cross_attn(
            query=x, key=memory, value=memory, mask=cross_attn_mask
        )
        x = self.norm2(residual + self.dropout2(cross_out))

        # --- 3. Position-wise Feed-Forward Network ---
        residual = x
        ffn_out = self.ffn(x)
        x = self.norm3(residual + self.dropout3(ffn_out))

        return x, cross_attn_weights


class TransformerDecoder(nn.Module):
    """
    Multi-layer Transformer Decoder for Image Caption Generation.

    Transforms input token sequences and visual features into next-token vocabulary distributions.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 100,
    ):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # 1. Token Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # 2. Sinusoidal Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # 3. Stack of Decoder Layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Final Layer Normalization
        self.final_norm = nn.LayerNorm(d_model)

        # 5. Output linear projection to vocabulary size
        # Shape: [B, T, d_model] -> [B, T, vocab_size]
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for training with teacher forcing or step-wise generation.

        Args:
            tokens (torch.Tensor): Input token IDs of shape [B, T].
            memory (torch.Tensor): Visual features of shape [B, 49, d_model].
            self_attn_mask (torch.Tensor, optional): Causal mask [1, 1, T, T].
            cross_attn_mask (torch.Tensor, optional): Optional memory mask.

        Returns:
            logits (torch.Tensor): Unnormalized vocabulary scores [B, T, vocab_size].
            all_cross_attn_weights (List[torch.Tensor]): List of attention weights from each layer.
        """
        # Step 1: Embed tokens and scale by sqrt(d_model) (standard Transformer convention)
        # [B, T] -> [B, T, d_model]
        x = self.embedding(tokens) * (self.d_model ** 0.5)

        # Step 2: Add positional encodings
        x = self.pos_encoding(x)

        # Step 3: Pass through stacked decoder layers
        all_cross_attn_weights = []
        for layer in self.layers:
            x, cross_weights = layer(
                x=x,
                memory=memory,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )
            all_cross_attn_weights.append(cross_weights)

        # Step 4: Final Layer Normalization
        x = self.final_norm(x)

        # Step 5: Linear projection to vocabulary size
        # [B, T, d_model] -> [B, T, vocab_size]
        logits = self.fc_out(x)

        return logits, all_cross_attn_weights
