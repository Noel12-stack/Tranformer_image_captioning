"""
CNN Image Encoder Module.

This module implements the visual feature extraction backbone using a pretrained
ResNet-50 architecture. The classification head (fc) and adaptive average pooling
layers are removed, retaining the spatial convolutional feature map from layer 4.

Tensor Flow:
    1. Input image tensor:
       Shape: [B, 3, 224, 224]  (B = batch size, 3 channels, 224x224 pixels)
    2. ResNet-50 layer4 output:
       Shape: [B, 2048, 7, 7]   (2048 channels across a 7x7 spatial grid)
    3. Permutation and Flattening:
       Permute: [B, 7, 7, 2048]
       Flatten: [B, 49, 2048]   (49 spatial regions, each with a 2048-dim vector)
    4. Trainable Visual Projection:
       Linear(2048 -> d_model):
       Shape: [B, 49, d_model]  (e.g., [B, 49, 256])
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet50Encoder(nn.Module):
    """
    Pretrained ResNet-50 visual feature extractor with trainable linear projection.

    Attributes:
        freeze_backbone (bool): If True, freeze all ResNet-50 backbone parameters.
        d_model (int): Target dimension for visual feature vectors fed into Transformer.
        backbone (nn.Sequential): Truncated ResNet-50 (up to layer4).
        projection (nn.Linear): Trainable projection from 2048 -> d_model.
        layer_norm (nn.LayerNorm): Stabilizing layer norm after projection.
    """

    def __init__(self, d_model: int = 256, pretrained: bool = True, freeze_backbone: bool = True):
        super(ResNet50Encoder, self).__init__()
        self.d_model = d_model
        self.freeze_backbone = freeze_backbone

        # Load pretrained or uninitialized ResNet-50
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)

        # Retain layers up to layer4; remove avgpool and fc
        # Modules in sequential: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # Freeze CNN backbone parameters if requested
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Trainable visual projection layer: 2048 -> d_model
        # Mathematically: V_projected = V_CNN * W_v + b_v
        self.projection = nn.Linear(2048, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the visual encoder.

        Args:
            images (torch.Tensor): Batch of normalized images of shape [B, 3, 224, 224].

        Returns:
            torch.Tensor: Projected spatial features of shape [B, 49, d_model].
        """
        # Step 1: Feature map extraction from frozen CNN backbone
        # Shape: [B, 3, 224, 224] -> [B, 2048, 7, 7]
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)

        # Step 2: Reshape spatial grid: [B, 2048, 7, 7] -> [B, 49, 2048]
        # B = batch size, C = 2048, H = 7, W = 7
        batch_size, channels, height, width = features.size()
        # Permute to [B, H, W, C] then flatten to [B, H*W, C]
        features = features.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        # Verify: features shape is [B, 49, 2048]

        # Step 3: Trainable linear projection to Transformer embedding dimension
        # Shape: [B, 49, 2048] -> [B, 49, d_model]
        projected = self.projection(features)
        projected = self.layer_norm(projected)
        projected = self.dropout(projected)

        return projected

    def extract_raw_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract raw 2048-dimensional features without linear projection.
        Useful for precomputing and caching features on disk.

        Args:
            images (torch.Tensor): [B, 3, 224, 224]

        Returns:
            torch.Tensor: [B, 49, 2048]
        """
        with torch.no_grad():
            features = self.backbone(images)
            b, c, h, w = features.size()
            features = features.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        return features

    def project_raw_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        """
        Project cached [B, 49, 2048] features into [B, 49, d_model].
        """
        projected = self.projection(raw_features)
        projected = self.layer_norm(projected)
        projected = self.dropout(projected)
        return projected
