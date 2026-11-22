"""Unit tests for training dynamics, backpropagation, and checkpointing."""

import os
import unittest
import torch
import torch.nn as nn
from models.image_captioner import ImageCaptioner


class TestTrainingAndGradients(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 100
        self.d_model = 64
        self.nhead = 2
        self.num_layers = 1
        self.model = ImageCaptioner(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=128,
            pretrained_cnn=False,
            freeze_cnn=True,
        )

    def test_gradient_flow_and_parameter_freezing(self):
        batch_size = 2
        seq_len = 6
        images = torch.randn(batch_size, 3, 224, 224)
        captions = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        logits, _ = self.model(images, captions)
        targets = captions[:, 1:].contiguous()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(logits.view(-1, self.vocab_size), targets.view(-1))

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        # Backpropagation
        loss.backward()

        # 1. Check frozen CNN parameters have NO gradients
        for name, param in self.model.encoder.backbone.named_parameters():
            self.assertIsNone(param.grad, f"Frozen parameter {name} received a gradient!")

        # 2. Check visual projection layer parameters DO have gradients
        self.assertIsNotNone(self.model.encoder.projection.weight.grad)
        self.assertGreater(self.model.encoder.projection.weight.grad.abs().sum().item(), 0.0)

        # 3. Check Transformer decoder parameters DO have gradients
        self.assertIsNotNone(self.model.decoder.fc_out.weight.grad)
        self.assertGreater(self.model.decoder.fc_out.weight.grad.abs().sum().item(), 0.0)

    def test_optimizer_updates_trainable_parameters(self):
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        captions = torch.randint(0, self.vocab_size, (batch_size, 5))

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=0.01)

        # Record parameter weights before step
        old_fc_weight = self.model.decoder.fc_out.weight.clone()

        self.model.train()
        optimizer.zero_grad()
        logits, _ = self.model(images, captions)
        targets = captions[:, 1:].contiguous()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(logits.view(-1, self.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        # Verify weights were updated
        new_fc_weight = self.model.decoder.fc_out.weight
        self.assertFalse(torch.allclose(old_fc_weight, new_fc_weight))

    def test_checkpoint_saving_and_loading(self):
        ckpt_path = "checkpoints/test_checkpoint.pt"
        os.makedirs("checkpoints", exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
            },
            ckpt_path,
        )
        self.assertTrue(os.path.exists(ckpt_path))

        # Re-instantiate a new model with same dimensions
        loaded_model = ImageCaptioner(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=128,
            pretrained_cnn=False,
            freeze_cnn=True,
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.eval()
        self.model.eval()

        # Run both on identical input in eval mode
        test_img = torch.randn(1, 3, 224, 224)
        test_cap = torch.randint(0, self.vocab_size, (1, 5))

        with torch.no_grad():
            out1, _ = self.model(test_img, test_cap)
            out2, _ = loaded_model(test_img, test_cap)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

        # Clean up test checkpoint
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


if __name__ == "__main__":
    unittest.main()
