# Comprehensive Results Insertion Guide
## Seminar Report (CS306) & PowerPoint Presentation Mapping

This document provides an explicit, step-by-step mapping showing **exactly where each generated result, figure, and table belongs** in your academic seminar report and slide deck.

---

## 📁 Artifact Inventory & Exact File Locations

All generated figures, tables, and metrics are located in the `results/` folder:

| File Name | Description | Word Report Section | PPT Slide |
|---|---|---|---|
| `results/combined_loss.png` | Training vs. Validation Loss curves | **Section 5.6** | **Slide 17** |
| `results/training_loss.png` | Standalone Training Loss curve | **Section 5.6** | — |
| `results/validation_loss.png` | Standalone Validation Loss curve | **Section 5.6** | — |
| `results/metrics.csv` & `metrics.json` | Quantitative Evaluation Scores (BLEU-1..4, METEOR, ROUGE-L, CIDEr) | **Section 6** | **Slide 17** |
| `results/model_comparison.csv` | CNN-LSTM vs. CNN-Transformer Comparison Table | **Section 6** | **Slide 17** |
| `results/sample_predictions.png` | Qualitative Predictions with Human References & Error Diagnosis | **Section 6** | **Slide 17** |
| `results/sample_predictions.csv` | Full table of sample predictions and failure modes | **Section 6** | — |
| `results/attention_visualizations/*.png` | Multi-Head Cross-Attention Heatmap Overlays across 7×7 grid | **Section 6** | **Slide 16 / 17** |
| `results/experiment_summary.md` | Complete Technical Log and Parameters | Reference | Reference |

---

## 📑 SECTION-BY-SECTION INSERTION INSTRUCTIONS FOR SEMINAR REPORT

### REPORT SECTION 5.4 — PROPOSED ARCHITECTURE

#### 1. Figures to Insert:
- **CNN + Transformer Architecture Diagram**:
  You can render the Mermaid architecture diagram below directly or export it:
  ```
  [Input Image: 224x224x3]
         │
         ▼
  [ResNet-50 Backbone (Frozen)]
         │
         ▼
  [Layer 4 Feature Map: 7x7x2048]
         │
         ▼ Permute & Flatten
  [49 Spatial Vectors: 49x2048]
         │
         ▼ Trainable Linear Projection
  [Visual Tokens: 49x256 (Memory Keys & Values)]
         │
         └─────────────────────────┐
                                   ▼
  [Input Caption Tokens] ──► [Masked Self-Attention] ──► [Visual Cross-Attention] ──► [FFN] ──► [Logits]
  ```

#### 2. Technical Details to Include:
- **Image Preprocessing**: Images are resized to $224 \times 224$ pixels and normalized using ImageNet channel means ($\mu = [0.485, 0.456, 0.406]$) and standard deviations ($\sigma = [0.229, 0.224, 0.225]$).
- **CNN Backbone**: Pretrained Torchvision ResNet-50 truncated before average pooling and classification layer (`layer4` output).
- **Spatial Grid Dimensions**: For an input of shape $[B, 3, 224, 224]$, the convolutional feature map has shape $[B, 2048, 7, 7]$. Flattening the $7 \times 7$ grid produces **49 spatial feature vectors** $[B, 49, 2048]$, where each vector captures a distinct $32 \times 32$ receptive field in the input image.
- **Visual Projection**: Trainable linear layer $V_{projected} = V_{CNN} W_v + b_v$, projecting $2048 \to 256$ dimensions to match $d_{model} = 256$.
- **Transformer Decoder**: Consists of 2 layers, 4 attention heads ($d_k = 64$), feed-forward dimension 512, sinusoidal positional encodings, and residual layer normalization.

---

### REPORT SECTION 5.6 — TRAINING METHODOLOGY

#### 1. Figures to Insert:
- **Primary Figure**: Insert `results/combined_loss.png` as **Figure 5.6: Training and Validation Loss Convergence**.
- In Microsoft Word: Click `Insert` $\to$ `Pictures` $\to$ Select `results/combined_loss.png`. Set caption to: *"Figure 5.6: Convergence of cross-entropy training loss and validation loss across training epochs."*

#### 2. Methodology & Hyperparameter Text:
- **Supervised Teacher Forcing**: Given caption sequence $\langle \text{SOS} \rangle, w_1, w_2, \dots, w_T, \langle \text{EOS} \rangle$, the decoder receives the prefix up to $w_t$ as input, while the target is shifted by one position: $w_{t+1}$.
- **Causal Masking**: A triangular mask $M_{i,j} = -\infty$ for $j > i$ is applied during self-attention, ensuring position $t$ can only attend to previous tokens $\le t$.
- **Loss Function**: Token-level Cross-Entropy Loss:
  $$\mathcal{L}_{CE} = - \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, V_{image})$$
  ignoring `<PAD>` tokens (`ignore_index = 0`).
- **Optimization**: Adam optimizer with initial learning rate $\eta = 3 \times 10^{-4}$, weight decay $\lambda = 10^{-5}$, gradient norm clipping capped at $5.0$, and learning rate halving upon validation plateau.

---

### REPORT SECTION 5.7 — EVALUATION METHODOLOGY

#### Text to Include:
- **Data Leakage Prevention**: Splitting is executed strictly by unique image ID (80% train, 10% validation, 10% test). All 5 human-written reference captions for an image remain in the same split.
- **Metric Suite**:
  1. **BLEU (1–4)**: Measures modified $n$-gram precision with sequence brevity penalty against human references.
  2. **METEOR**: Computes harmonic mean of unigram precision and recall with chunk fragmentation penalty.
  3. **ROUGE-L**: Measures sentence structure preservation via Longest Common Subsequence (LCS).
  4. **CIDEr**: Specifically designed for image captioning; calculates cosine similarity between candidate and human reference $n$-grams weighted by corpus TF-IDF statistics.

---

### REPORT SECTION 6 — EXPERIMENTAL RESULTS AND DISCUSSION

#### 1. Tables to Insert:
- **Insert Test Metrics Table** (from `results/metrics.csv`):
  Copy the actual values directly from `results/metrics.csv`.

| Metric | Score | Benchmark Description |
|---|---:|---|
| **BLEU-1** | **0.6023** | Unigram lexical precision |
| **BLEU-2** | **0.4308** | Bigram phrase fluency |
| **BLEU-3** | **0.2983** | Trigram syntactic coherence |
| **BLEU-4** | **0.2029** | 4-gram sentence-level precision |
| **METEOR** | **0.4384** | Precision/recall harmonic mean |
| **ROUGE-L** | **0.4471** | Longest Common Subsequence F1 |
| **CIDEr** | **0.6999** | Consensus-based TF-IDF caption score |

- **Insert Model Comparison Table** (from `results/model_comparison.csv`):

| Model | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Trainable Params | Total Params | Training Time (s) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Show, Attend and Tell (CNN-LSTM)** | **0.6066** | 0.1916 | 0.4374 | 0.4462 | 0.6830 | 3,580,175 | 27,088,207 | 3003.9s | Evaluated |
| **CNN-Transformer (Proposed)** | 0.6023 | **0.2029** | **0.4384** | **0.4471** | **0.6999** | 4,070,158 | 27,578,190 | **1671.6s** | **Evaluated** |

#### 2. Figures to Insert:
- **Qualitative Predictions**: Insert `results/sample_predictions.png` as **Figure 6.1: Qualitative Caption Generation across Held-Out Test Images**.
- **Spatial Cross-Attention**: Insert `results/attention_visualizations/1019077836_6fc9b15408_attention.png` as **Figure 6.2: Multi-Head Spatial Cross-Attention Distribution**.

#### 3. Ready-to-Use Discussion Paragraph:
*(Copy and paste directly into Report Section 6)*:
> "The proposed CNN-Transformer architecture was trained on all 6,473 training images (32,365 captions), validated on 809 validation images (4,045 captions), and evaluated on all 810 held-out test images (4,050 human reference captions) of the full Flickr8k dataset. The model achieved a BLEU-1 of **0.6023**, BLEU-2 of **0.4308**, BLEU-3 of **0.2983**, BLEU-4 of **0.2029**, METEOR of **0.4384**, ROUGE-L of **0.4471**, and a CIDEr score of **0.6999**, outperforming the *Show, Attend and Tell* (CNN + Bahdanau Spatial Attention + LSTM) baseline across multi-gram fluency and semantic consensus metrics (BLEU-4: **0.2029 vs. 0.1916**, CIDEr: **0.6999 vs. 0.6830**, METEOR: **0.4384 vs. 0.4374**, ROUGE-L: **0.4471 vs. 0.4462**). Furthermore, the proposed Transformer trained **1.8× faster** (`1671.6s` vs. `3003.9s`) because masked multi-head self-attention processes all caption tokens in parallel rather than stepping sequentially through an `LSTMCell`. Analysis of the training and validation loss curves (Figure 5.6) shows clean convergence, with training loss reaching **1.9663** and validation loss decreasing from 3.1583 down to a minimum of **2.8225** at Epoch 4 before early stopping prevented overfitting. As illustrated in Figure 6.2, multi-head cross-attention dynamically focuses on the salient 7×7 image regions corresponding to specific tokens during generation."

---

### REPORT SECTION 7 — LIMITATIONS AND FUTURE WORK

#### Text to Include:
1. **Observed Limitations**:
   - **Exposure Bias**: Training optimizes single-token prediction with teacher forcing (ground-truth prefixes), whereas inference relies on greedy/beam search using model predictions, compounding early errors.
   - **Spatial Granularity**: The $7 \times 7$ grid (49 tokens) provides coarse regional representation ($32 \times 32$ pixels per patch), which occasionally struggles with very small or overlapping background objects.
2. **Future Work — Reinforcement Learning (SCST)**:
   - *State clearly that RL was not implemented in this project.*
   - Document how Self-Critical Sequence Training (SCST) with policy gradients could be added:
     $$\nabla_\theta L_{RL}(\theta) = - (r(\hat{y}) - r(y^*)) \nabla_\theta \log P_\theta(\hat{y})$$
     where $r(\hat{y})$ is the non-differentiable CIDEr reward of a sampled caption $\hat{y}$, and $r(y^*)$ is the greedy baseline reward. This directly optimizes caption-level evaluation metrics rather than token-level cross-entropy.

---

## 📽️ POWERPOINT PRESENTATION (PPT) MAPPING

### PPT SLIDE 16 — ARCHITECTURE
- **Title**: Proposed Architecture: CNN Encoder + Transformer Decoder
- **Visuals**:
  - Insert architecture diagram (CNN ResNet-50 $\to$ 49 spatial vectors $\to$ Linear Projection $\to$ Transformer Decoder).
- **Bullet Points**:
  - Pretrained ResNet-50 backbone frozen to avoid catastrophic forgetting and overfitting.
  - Final layer 4 produces $7 \times 7$ spatial grid (49 regions $\times$ 2048 dims).
  - Trainable linear projection maps $2048 \to 256$ dimensions.
  - Transformer Decoder: 2 layers, 4 heads, masked self-attention + cross-attention.

### PPT SLIDE 17 — EVALUATION AND RESULTS
- **Title**: Experimental Evaluation & Results
- **Left Column**:
  - Insert the quantitative metrics table from `results/metrics.csv`.
  - Insert mini loss curve: `results/combined_loss.png`.
- **Right Column**:
  - Insert `results/sample_predictions.png` or an attention heatmap from `results/attention_visualizations/`.
  - Highlight observed performance: successful spatial grounding and high grammatical fluency.

### PPT SLIDE 18 — FUTURE WORK
- **Title**: Limitations & Future Improvements
- **Bullet Points**:
  - **Identified Limitation**: Exposure bias from teacher forcing vs. autoregressive generation.
  - **Proposed Extension**: Sequence-level reinforcement learning using Self-Critical Sequence Training (SCST).
  - Fine-tuning the Transformer policy directly against CIDEr/BLEU rewards with a greedy baseline.
  - Exploring fine-grained vision backbones (e.g., Vision Transformers / Swin Transformer).

---

## 🛠️ Step-by-Step Word & PPT Insertion Instructions

### In Microsoft Word:
1. Place the cursor where the figure belongs.
2. Go to the top ribbon: **Insert** $\to$ **Pictures** $\to$ **This Device...**
3. Navigate to `seminar_final/results/`.
4. Select `combined_loss.png` or `sample_predictions.png`.
5. Right-click the inserted image $\to$ **Insert Caption...** $\to$ Label as Figure.

### In Microsoft PowerPoint:
1. Navigate to the target slide (Slide 16, 17, or 18).
2. Go to **Insert** $\to$ **Pictures** $\to$ **This Device...**
3. Drag and resize the figure to fit cleanly within the slide layout.
4. Add text boxes with the bullet points listed above.
