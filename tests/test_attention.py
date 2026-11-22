"""Unit tests for Multi-Head Attention, Scaled Dot-Product, and Positional Encoding."""

import unittest
import torch
from models.attention import (
    PositionalEncoding,
    MultiHeadAttention,
    generate_causal_mask,
    generate_padding_mask,
)


class TestAttentionMechanisms(unittest.TestCase):

    def test_positional_encoding(self):
        d_model = 256
        seq_len = 15
        batch_size = 2
        pe = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.0)
        x = torch.zeros(batch_size, seq_len, d_model)
        out = pe(x)

        self.assertEqual(out.shape, (batch_size, seq_len, d_model))
        # Verify positional encoding values are non-zero and bounded between -1 and 1
        self.assertTrue(torch.all(out >= -1.0) and torch.all(out <= 1.0))
        # Ensure position 0 is different from position 1
        self.assertFalse(torch.allclose(out[0, 0, :], out[0, 1, :]))

    def test_multihead_self_attention_dimensions_and_sum(self):
        batch_size = 3
        seq_len = 8
        d_model = 256
        nhead = 4
        mha = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=0.0)

        x = torch.randn(batch_size, seq_len, d_model)
        out, attn_weights = mha(query=x, key=x, value=x)

        # Output dimensions: [B, T, d_model]
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))
        # Attention weights dimensions: [B, nhead, T, T]
        self.assertEqual(attn_weights.shape, (batch_size, nhead, seq_len, seq_len))

        # Check weights sum to 1.0 along the key dimension (dim=-1)
        sums = attn_weights.sum(dim=-1)
        expected = torch.ones(batch_size, nhead, seq_len)
        self.assertTrue(torch.allclose(sums, expected, atol=1e-5))

        # Check no NaN or inf values
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isnan(attn_weights).any())

    def test_causal_mask_prevents_future_attention(self):
        seq_len = 5
        device = torch.device("cpu")
        mask = generate_causal_mask(seq_len, device=device)

        # Check upper triangle (excluding diagonal) is negative infinity
        self.assertEqual(mask.shape, (1, 1, seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    self.assertEqual(mask[0, 0, i, j].item(), float("-inf"))
                else:
                    self.assertEqual(mask[0, 0, i, j].item(), 0.0)

        # Run attention with causal mask
        d_model = 64
        nhead = 2
        mha = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=0.0)
        x = torch.randn(1, seq_len, d_model)
        _, attn_weights = mha(query=x, key=x, value=x, mask=mask)

        # In attn_weights, positions with j > i must have exactly 0.0 attention
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    self.assertAlmostEqual(attn_weights[0, :, i, j].max().item(), 0.0, places=5)

    def test_cross_attention_different_sequence_lengths(self):
        batch_size = 2
        text_len = 10     # T_q
        image_regions = 49 # T_k, T_v
        d_model = 128
        nhead = 4

        cross_attn = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=0.0)
        text_queries = torch.randn(batch_size, text_len, d_model)
        visual_memory = torch.randn(batch_size, image_regions, d_model)

        out, attn_weights = cross_attn(query=text_queries, key=visual_memory, value=visual_memory)

        # Output shape matches query shape
        self.assertEqual(out.shape, (batch_size, text_len, d_model))
        # Attention weights: [B, nhead, text_len, image_regions]
        self.assertEqual(attn_weights.shape, (batch_size, nhead, text_len, image_regions))

        # Weights must sum to 1 over the 49 image regions
        sums = attn_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
