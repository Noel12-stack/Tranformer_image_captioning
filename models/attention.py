"""
Attention Mechanisms and Positional Encodings.

This module implements from scratch:
1. Sinusoidal Positional Encoding (Vaswani et al., 2017).
2. Scaled Dot-Product Attention with masking and attention weight extraction.
3. Multi-Head Attention used for both:
   - Masked Multi-Head Self-Attention (text tokens attending to past text tokens)
   - Multi-Head Cross-Attention (text query tokens attending to spatial image regions)

Mathematical Formulations:
    - Sinusoidal Positional Encoding:
      PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    - Scaled Dot-Product Attention:
      Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M) * V

    - Multi-Head Attention:
      MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
      where head_i = Attention(Q * W_Q^i, K * W_K^i, V * W_V^i)
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    Injects positional order into token embeddings so the permutation-invariant
    Transformer decoder can distinguish sequence order.

    Shape:
        Input:  [B, T, d_model]
        Output: [B, T, d_model]
    """

    def __init__(self, d_model: int = 256, max_len: int = 500, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create persistent buffer for positional encodings
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer: persistent state not updated by optimizer
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to token embeddings.

        Args:
            x (torch.Tensor): [B, T, d_model]

        Returns:
            torch.Tensor: [B, T, d_model] with positional information added.
        """
        seq_len = x.size(1)
        # Broadcast pe across batch dimension
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention.

    Args:
        q: Query tensor of shape [B, nhead, T_q, d_k]
        k: Key tensor of shape [B, nhead, T_k, d_k]
        v: Value tensor of shape [B, nhead, T_k, d_v]
        mask: Optional mask tensor broadcastable to [B, nhead, T_q, T_k].
              Positions with -inf or large negative numbers will have zero attention.
        dropout: Optional dropout layer applied to attention weights.

    Returns:
        output: Context vector tensor of shape [B, nhead, T_q, d_v]
        attn_weights: Attention weights of shape [B, nhead, T_q, T_k]
    """
    d_k = q.size(-1)

    # 1. Compute dot products of queries and keys: [B, nhead, T_q, T_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. Apply additive attention mask if provided
    if mask is not None:
        scores = scores + mask

    # 3. Softmax over key sequence dimension (dim=-1) for numerical stability
    attn_weights = torch.softmax(scores, dim=-1)

    # Handle cases where all entries were masked (e.g., all padding)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # 4. Optional dropout on attention probabilities
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # 5. Multiply attention probabilities by value vectors: [B, nhead, T_q, d_v]
    output = torch.matmul(attn_weights, v)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for Transformer architectures.

    Can be used as:
    1. Self-Attention: q, k, v all derived from decoder token sequence.
    2. Cross-Attention: q derived from text tokens; k and v derived from CNN visual features.

    Tensor Dimensions:
        q_in: [B, T_q, d_model]
        k_in: [B, T_k, d_model]
        v_in: [B, T_v, d_model] (where T_v == T_k)
        Output: [B, T_q, d_model]
        attn_weights: [B, nhead, T_q, T_k]
    """

    def __init__(self, d_model: int = 256, nhead: int = 4, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # Trainable projection matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward computation of multi-head attention.

        Args:
            query: [B, T_q, d_model]
            key:   [B, T_k, d_model]
            value: [B, T_k, d_model]
            mask:  Optional mask tensor [B, 1, T_q, T_k] or [1, 1, T_q, T_k]

        Returns:
            output: [B, T_q, d_model]
            attn_weights: [B, nhead, T_q, T_k]
        """
        batch_size, t_q, _ = query.size()
        t_k = key.size(1)

        # Step 1: Linear projections and split into heads
        # Shape transformations:
        # [B, T_q, d_model] -> [B, T_q, nhead, d_k] -> [B, nhead, T_q, d_k]
        q = self.w_q(query).view(batch_size, t_q, self.nhead, self.d_k).transpose(1, 2)
        # [B, T_k, d_model] -> [B, T_k, nhead, d_k] -> [B, nhead, T_k, d_k]
        k = self.w_k(key).view(batch_size, t_k, self.nhead, self.d_k).transpose(1, 2)
        # [B, T_k, d_model] -> [B, T_k, nhead, d_k] -> [B, nhead, T_k, d_k]
        v = self.w_v(value).view(batch_size, t_k, self.nhead, self.d_k).transpose(1, 2)

        # Step 2: Compute scaled dot-product attention
        # output: [B, nhead, T_q, d_k]
        # attn_weights: [B, nhead, T_q, T_k]
        context, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )

        # Step 3: Concatenate heads and project output
        # [B, nhead, T_q, d_k] -> [B, T_q, nhead, d_k] -> [B, T_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, t_q, self.d_model)
        output = self.w_o(context)

        return output, attn_weights


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Generate an upper-triangular causal attention mask.

    Prevents position t from attending to future tokens (> t).

    Returns:
        torch.Tensor of shape [1, 1, seq_len, seq_len], with 0.0 at allowed positions
        and -1e9 at masked future positions.
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len, seq_len]


def generate_padding_mask(pad_tensor: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Generate key padding mask from sequence of token IDs.

    Args:
        pad_tensor: [B, T] with token indices.
        pad_idx: ID representing <PAD>.

    Returns:
        torch.Tensor of shape [B, 1, 1, T] with -1e9 for pad positions and 0.0 elsewhere.
    """
    # True where token is PAD
    is_pad = (pad_tensor == pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
    mask = torch.zeros_like(is_pad, dtype=torch.float)
    mask = mask.masked_fill(is_pad, float("-inf"))
    return mask
