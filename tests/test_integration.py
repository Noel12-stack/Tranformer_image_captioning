"""Integration and Overfitting Verification Tests."""

import unittest
import torch
import torch.nn as nn
from models.image_captioner import ImageCaptioner
from models.lstm_baseline import CNNLSTMBaseline


class TestIntegrationAndOverfitting(unittest.TestCase):

    def test_synthetic_end_to_end_pipeline(self):
        """
        Integration test verifying synthetic pipeline without requiring Flickr8k download:
        1. Forward pass through CNN encoder and Transformer decoder
        2. Causal masked self-attention
        3. Visual cross-attention
        4. Cross-entropy loss calculation ignoring PAD
        5. Backward pass and optimizer update
        """
        vocab_size = 50
        d_model = 64
        nhead = 2
        batch_size = 4
        seq_len = 8

        model = ImageCaptioner(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=1,
            dim_feedforward=128,
            pretrained_cnn=False,
            freeze_cnn=True,
        )

        synthetic_images = torch.randn(batch_size, 3, 224, 224)
        synthetic_captions = torch.randint(1, vocab_size, (batch_size, seq_len))
        # Insert padding tokens at the end
        synthetic_captions[:, -2:] = 0

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        model.train()
        optimizer.zero_grad()

        # Full forward pass
        logits, cross_attns = model(synthetic_images, synthetic_captions)
        targets = synthetic_captions[:, 1:].contiguous()

        self.assertEqual(logits.shape, (batch_size, seq_len - 1, vocab_size))

        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        optimizer.step()

        # Test greedy inference works on the updated model
        model.eval()
        single_img = torch.randn(1, 3, 224, 224)
        gen_tokens, attn_maps = model.generate_caption_greedy(single_img, max_len=6)
        self.assertIsInstance(gen_tokens, list)
        self.assertIsInstance(attn_maps, list)

    def test_overfitting_on_tiny_subset(self):
        """
        Overfitting test (Section 6.3):
        Verifies that on a tiny fixed batch, the model is mathematically capable of
        reducing training loss towards zero, proving gradient flow, target shifting,
        and attention alignments are sound.
        """
        torch.manual_seed(42)
        vocab_size = 30
        d_model = 64
        batch_size = 2
        seq_len = 5

        model = ImageCaptioner(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            pretrained_cnn=False,
            freeze_cnn=True,
        )

        # Tiny fixed synthetic dataset
        fixed_images = torch.randn(batch_size, 3, 224, 224)
        fixed_captions = torch.tensor(
            [
                [1, 5, 12, 18, 2],   # Image 1 caption
                [1, 8, 14, 22, 2],   # Image 2 caption
            ],
            dtype=torch.long,
        )

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Record initial loss
        model.train()
        logits, _ = model(fixed_images, fixed_captions)
        targets = fixed_captions[:, 1:].contiguous()
        initial_loss = criterion(logits.view(-1, vocab_size), targets.view(-1)).item()

        # Train for 25 small iterations
        for _ in range(25):
            optimizer.zero_grad()
            logits, _ = model(fixed_images, fixed_captions)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Final loss must be substantially lower than initial loss
        self.assertLess(
            final_loss,
            initial_loss * 0.5,
            f"Overfitting failed: initial loss {initial_loss:.4f}, final loss {final_loss:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
