"""
Full CNN-Transformer Image Captioning Architecture.

Integrates:
1. ResNet-50 visual feature extractor with spatial projection (49 spatial regions).
2. Multi-layer Transformer decoder with masked self-attention and visual cross-attention.
3. Training forward pass with teacher forcing.
4. Inference methods:
   - Greedy decoding
   - Beam search decoding
   - Cross-attention map extraction for visualization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .cnn_encoder import ResNet50Encoder
from .transformer_decoder import TransformerDecoder
from .attention import generate_causal_mask


class ImageCaptioner(nn.Module):
    """
    End-to-end Image Captioning Model combining CNN encoder and Transformer decoder.

    Args:
        vocab_size (int): Total size of the vocabulary.
        d_model (int): Feature and embedding dimension (default: 256).
        nhead (int): Number of multi-head attention heads (default: 4).
        num_layers (int): Number of stacked decoder layers (default: 2).
        dim_feedforward (int): Hidden dimension of FFN (default: 512).
        dropout (float): Dropout probability (default: 0.1).
        pad_idx (int): Vocabulary index of <PAD> token.
        sos_idx (int): Vocabulary index of <SOS> token.
        eos_idx (int): Vocabulary index of <EOS> token.
        pretrained_cnn (bool): Whether to use pretrained weights for ResNet-50.
        freeze_cnn (bool): Whether to freeze ResNet-50 CNN parameters.
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
        sos_idx: int = 1,
        eos_idx: int = 2,
        pretrained_cnn: bool = True,
        freeze_cnn: bool = True,
    ):
        super(ImageCaptioner, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        # 1. ResNet-50 CNN Encoder + Visual Projection
        self.encoder = ResNet50Encoder(
            d_model=d_model,
            pretrained=pretrained_cnn,
            freeze_backbone=freeze_cnn,
        )

        # 2. Transformer Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_idx=pad_idx,
        )

    def forward(
        self,
        images_or_features: torch.Tensor,
        captions: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for supervised training using teacher forcing.

        Args:
            images_or_features: Batch of images [B, 3, 224, 224] OR precomputed
                                raw CNN features [B, 49, 2048].
            captions: Full tokenized captions of shape [B, T] (includes <SOS> and <EOS>).

        Returns:
            logits: Logits of shape [B, T-1, vocab_size].
            cross_attns: Cross-attention weights from each decoder layer.
        """
        device = images_or_features.device

        # Step 1: Extract or project visual features
        if images_or_features.dim() == 4:
            # images: [B, 3, 224, 224] -> visual_features: [B, 49, d_model]
            visual_features = self.encoder(images_or_features)
        else:
            # precomputed raw features: [B, 49, 2048] -> visual_features: [B, 49, d_model]
            visual_features = self.encoder.project_raw_features(images_or_features.float())

        # Step 2: Teacher forcing shift
        # Input to decoder excludes the final <EOS> / <PAD> token: shape [B, T-1]
        decoder_input = captions[:, :-1]
        seq_len = decoder_input.size(1)

        # Step 3: Generate causal attention mask to prevent looking ahead
        # Shape: [1, 1, seq_len, seq_len]
        causal_mask = generate_causal_mask(seq_len, device=device)

        # Step 4: Decoder forward pass
        # logits: [B, T-1, vocab_size]
        logits, cross_attns = self.decoder(
            tokens=decoder_input,
            memory=visual_features,
            self_attn_mask=causal_mask,
        )

        return logits, cross_attns

    def forward_with_features(
        self,
        visual_features: torch.Tensor,
        captions: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass when using pre-extracted CNN features.
        """
        device = visual_features.device
        decoder_input = captions[:, :-1]
        seq_len = decoder_input.size(1)
        causal_mask = generate_causal_mask(seq_len, device=device)

        logits, cross_attns = self.decoder(
            tokens=decoder_input,
            memory=visual_features,
            self_attn_mask=causal_mask,
        )
        return logits, cross_attns

    @torch.no_grad()
    def generate_caption_greedy(
        self,
        image: torch.Tensor,
        max_len: int = 25,
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Autoregressively generate caption using greedy decoding.

        Args:
            image: Single image tensor of shape [1, 3, 224, 224] or [3, 224, 224].
            max_len: Maximum number of tokens to generate.

        Returns:
            generated_tokens: List of token IDs (excluding <SOS>).
            attention_maps: List of attention matrices for each generated step.
                            Each element has shape [nhead, 49] from the final decoder layer.
        """
        self.eval()
        device = next(self.parameters()).device

        if image.dim() == 4 or (image.dim() == 3 and image.size(0) == 3):
            if image.dim() == 3:
                image = image.unsqueeze(0)  # [1, 3, 224, 224]
            image = image.to(device)
            visual_features = self.encoder(image)
        else:
            if image.dim() == 2:
                image = image.unsqueeze(0)  # [1, 49, 2048]
            image = image.to(device).float()
            visual_features = self.encoder.project_raw_features(image)

        # Step 2: Initialize sequence with <SOS>
        generated_tokens = [self.sos_idx]
        attention_maps = []

        for _ in range(max_len):
            input_tokens = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            seq_len = input_tokens.size(1)

            # Causal mask for current prefix
            causal_mask = generate_causal_mask(seq_len, device=device)

            # Decoder pass
            logits, cross_attns = self.decoder(
                tokens=input_tokens,
                memory=visual_features,
                self_attn_mask=causal_mask,
            )

            # Take prediction for the last position
            last_logits = logits[0, -1, :].clone()

            # Prevent <PAD> and <SOS> from being predicted
            last_logits[self.pad_idx] = float("-inf")
            last_logits[self.sos_idx] = float("-inf")

            next_token = torch.argmax(last_logits).item()
            generated_tokens.append(next_token)

            # Save cross-attention weights from the final decoder layer for this token step
            # cross_attns[-1]: [1, nhead, seq_len, 49] -> slice for the current step: [nhead, 49]
            step_attn = cross_attns[-1][0, :, -1, :].cpu()  # [nhead, 49]
            attention_maps.append(step_attn)

            if next_token == self.eos_idx:
                break

        # Remove the leading <SOS>
        output_tokens = generated_tokens[1:]
        return output_tokens, attention_maps

    @torch.no_grad()
    def generate_caption_beam_search(
        self,
        image: torch.Tensor,
        beam_size: int = 3,
        max_len: int = 25,
    ) -> List[int]:
        """
        Generate caption using Beam Search decoding.

        Args:
            image: Single image tensor [1, 3, 224, 224].
            beam_size: Number of hypotheses tracked simultaneously.
            max_len: Maximum token generation length.

        Returns:
            Best token sequence (excluding <SOS>).
        """
        self.eval()
        device = next(self.parameters()).device

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        visual_features = self.encoder(image)  # [1, 49, d_model]

        # Each hypothesis is a tuple: (score, tokens_list)
        hypotheses = [(0.0, [self.sos_idx])]
        completed = []

        for _ in range(max_len):
            all_candidates = []

            for score, tokens in hypotheses:
                if tokens[-1] == self.eos_idx:
                    completed.append((score, tokens))
                    continue

                input_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
                seq_len = input_tokens.size(1)
                causal_mask = generate_causal_mask(seq_len, device=device)

                logits, _ = self.decoder(
                    tokens=input_tokens,
                    memory=visual_features,
                    self_attn_mask=causal_mask,
                )

                last_logits = logits[0, -1, :].clone()
                last_logits[self.pad_idx] = float("-inf")
                last_logits[self.sos_idx] = float("-inf")

                log_probs = torch.log_softmax(last_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

                for k in range(beam_size):
                    cand_score = score + topk_log_probs[k].item()
                    cand_tokens = tokens + [topk_indices[k].item()]
                    all_candidates.append((cand_score, cand_tokens))

            if not all_candidates:
                break

            # Select top beam_size candidates
            ordered = sorted(all_candidates, key=lambda tup: tup[0] / len(tup[1]), reverse=True)
            hypotheses = ordered[:beam_size]

            # If all top hypotheses ended with EOS, terminate early
            if all(tokens[-1] == self.eos_idx for _, tokens in hypotheses):
                break

        # Collect any remaining hypotheses
        for score, tokens in hypotheses:
            if tokens[-1] == self.eos_idx:
                completed.append((score, tokens))
            else:
                completed.append((score, tokens + [self.eos_idx]))

        if not completed:
            return hypotheses[0][1][1:]

        # Select highest length-normalized score
        best_score, best_tokens = max(completed, key=lambda tup: tup[0] / len(tup[1]))
        return best_tokens[1:]

    def get_parameter_count(self) -> Dict[str, int]:
        """
        Count total, trainable, and frozen parameters.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
        }
