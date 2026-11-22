"""Unit tests for Transformer Decoder and Layers."""

import unittest
import torch
from models.transformer_decoder import TransformerDecoderLayer, TransformerDecoder
from models.attention import generate_causal_mask


class TestTransformerDecoder(unittest.TestCase):

    def test_decoder_layer_forward(self):
        batch_size = 2
        text_len = 7
        visual_regions = 49
        d_model = 128
        nhead = 4

        layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256)
        x = torch.randn(batch_size, text_len, d_model)
        memory = torch.randn(batch_size, visual_regions, d_model)
        mask = generate_causal_mask(text_len, device=x.device)

        out, cross_weights = layer(x=x, memory=memory, self_attn_mask=mask)

        # Output shape matches decoder input shape
        self.assertEqual(out.shape, (batch_size, text_len, d_model))
        # Cross-attention weights shape: [B, nhead, text_len, visual_regions]
        self.assertEqual(cross_weights.shape, (batch_size, nhead, text_len, visual_regions))

    def test_full_transformer_decoder_dimensions(self):
        vocab_size = 500
        d_model = 256
        nhead = 4
        num_layers = 2
        batch_size = 3
        seq_len = 12

        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=512,
        )

        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        memory = torch.randn(batch_size, 49, d_model)
        mask = generate_causal_mask(seq_len, device=tokens.device)

        logits, cross_attns = decoder(tokens=tokens, memory=memory, self_attn_mask=mask)

        # Verify output logits dimension: [B, seq_len, vocab_size]
        self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
        # Verify cross_attns has one entry per layer
        self.assertEqual(len(cross_attns), num_layers)
        self.assertEqual(cross_attns[-1].shape, (batch_size, nhead, seq_len, 49))


if __name__ == "__main__":
    unittest.main()
