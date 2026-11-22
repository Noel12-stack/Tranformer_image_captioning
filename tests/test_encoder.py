"""Unit tests for ResNet-50 CNN Encoder and Visual Projection."""

import unittest
import torch
from models.cnn_encoder import ResNet50Encoder


class TestCNNEncoder(unittest.TestCase):

    def setUp(self):
        # Instantiate without downloading pretrained weights for instant testing
        self.encoder = ResNet50Encoder(d_model=256, pretrained=False, freeze_backbone=True)
        self.encoder.eval()

    def test_encoder_output_dimensions(self):
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 224, 224)

        features = self.encoder(dummy_images)
        # Expected: [batch_size, 49 spatial regions, 256 d_model]
        self.assertEqual(features.shape, (batch_size, 49, 256))

    def test_backbone_parameter_freezing(self):
        # All backbone parameters must have requires_grad == False
        for name, param in self.encoder.backbone.named_parameters():
            self.assertFalse(param.requires_grad, f"Parameter {name} was not frozen!")

    def test_projection_parameter_trainability(self):
        # Projection layer and layer norm must be trainable
        self.assertTrue(self.encoder.projection.weight.requires_grad)
        self.assertTrue(self.encoder.projection.bias.requires_grad)
        self.assertTrue(self.encoder.layer_norm.weight.requires_grad)

    def test_raw_feature_extraction_and_projection(self):
        dummy_images = torch.randn(1, 3, 224, 224)
        raw = self.encoder.extract_raw_features(dummy_images)
        self.assertEqual(raw.shape, (1, 49, 2048))

        projected = self.encoder.project_raw_features(raw)
        self.assertEqual(projected.shape, (1, 49, 256))


if __name__ == "__main__":
    unittest.main()
