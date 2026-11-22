# Attention-Based Image Caption Generation Using CNN and Transformer Architectures

An academic Deep Learning project developed for a Computer Science Engineering (CSE) seminar. This project investigates how effectively Transformer-based attention combines visual spatial representations and linguistic context to autoregressively generate natural language image descriptions.

---

## 📌 Project Overview

Traditional image captioning models (such as Show and Tell) employ a CNN encoder combined with a recurrent LSTM decoder. While functional, recurrent architectures suffer from an information compression bottleneck: all historical textual and spatial context must be squeezed into a single fixed-length hidden state vector $h_t$.

In this project, we implement and evaluate a modern **CNN + Transformer** architecture:
1. **Pretrained ResNet-50 CNN Encoder**: Extracts a $7 \times 7$ feature grid (49 spatial regions) from the final convolutional layer (`layer4`). The CNN backbone parameters are frozen to prevent overfitting.
2. **Trainable Visual Projection Layer**: Projects the 2048-dimensional convolutional features into the Transformer model dimension ($d_{model} = 256$).
3. **Masked Multi-Head Self-Attention**: Employs a causal triangular mask to enable token positions to attend only to preceding words during teacher-forced training and inference.
4. **Multi-Head Visual Cross-Attention**: Serves as the cross-modal bridge where text queries ($Q$) attend directly across all 49 spatial visual keys ($K$) and values ($V$), allowing each generated word to retrieve features from distinct image regions.
5. **Fair Academic Baseline (CNN-LSTM)**: Implemented on identical data splits, preprocessing, and vocabulary to empirically evaluate the advantages of Transformer attention over recurrent decoding.

---

## 📂 Project Folder Structure

```
seminar_final/
│
├── configs/
│   └── config.yaml                     # Central hyperparameter configuration
│
├── models/
│   ├── __init__.py
│   ├── cnn_encoder.py                  # ResNet-50 backbone + 2048->256 projection
│   ├── attention.py                    # Sinusoidal PE, Scaled Dot-Product, Multi-Head Attention
│   ├── transformer_decoder.py          # Multi-layer Transformer Decoder
│   ├── image_captioner.py              # End-to-end model (greedy & beam decoding)
│   └── lstm_baseline.py                # CNN-LSTM baseline for empirical comparison
│
├── dataset/
│   ├── __init__.py
│   ├── preprocessing.py                # Image transforms (ImageNet norm) & caption cleaning
│   ├── vocabulary.py                   # Special tokens (<PAD>, <SOS>, <EOS>, <UNK>) & mappings
│   ├── split_dataset.py                # Zero data-leakage partition strictly by Image ID
│   └── dataset.py                      # PyTorch Dataset, CollateFn, and synthetic generator
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                      # Training loop, early stopping, gradient clipping
│   ├── validate.py                     # Cross-entropy validation loop ignoring <PAD>
│   └── train.py                        # CLI training entry point
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                      # BLEU-1..4, METEOR, ROUGE-L, CIDEr
│   ├── generate_captions.py            # Autoregressive generation across test splits
│   ├── evaluate.py                     # Quantitative evaluation pipeline
│   └── compare_models.py               # Model comparison matrix generator
│
├── visualization/
│   ├── __init__.py
│   ├── plot_losses.py                  # Training and validation loss curves
│   ├── visualize_attention.py          # Real 7x7 spatial cross-attention heatmap overlays
│   └── sample_predictions.py           # Qualitative table and error mode diagnosis
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py                 # Dataset loading, collate, and vocabulary tests
│   ├── test_encoder.py                 # ResNet-50 dimensions and parameter freezing tests
│   ├── test_attention.py               # Causal mask, cross-attention dimensions, unit tests
│   ├── test_decoder.py                 # Transformer decoder dimensions and masking
│   ├── test_training.py                # Gradients, parameter updates, checkpointing
│   └── test_integration.py             # End-to-end synthetic pipeline & overfitting test
│
├── results/
│   ├── metrics.json                    # Evaluated test metrics (JSON)
│   ├── metrics.csv                     # Evaluated test metrics (CSV)
│   ├── model_comparison.csv            # CNN-LSTM vs. CNN-Transformer comparison table
│   ├── training_history.csv            # Epoch-by-epoch loss convergence records
│   ├── predictions.csv                 # Detailed test predictions vs. human references
│   ├── sample_predictions.csv          # Qualitative sample table with failure classifications
│   ├── sample_predictions.png          # Visual report figure of sample predictions
│   ├── combined_loss.png               # Training vs. Validation loss curve
│   ├── training_loss.png               # Standalone training loss
│   ├── validation_loss.png             # Standalone validation loss
│   ├── attention_visualizations/       # Heatmap overlays per generated word
│   └── experiment_summary.md           # Auto-compiled technical seminar summary
│
├── checkpoints/
│   ├── best_transformer.pt            # Best checkpoint for CNN-Transformer
│   ├── best_lstm.pt                   # Best checkpoint for CNN-LSTM Baseline
│   └── vocab.json                     # Serialized training vocabulary
│
├── app.py                              # Streamlit interactive web demonstration app
├── run_experiments.py                  # Master one-click experiment execution script
├── generate_report_results.py          # Auto-generates seminar report results section
├── RESULTS_INSERTION_GUIDE.md          # Exact mapping to Word report sections & PPT slides
├── requirements.txt                    # Python dependencies
└── README.md                           # Comprehensive documentation
```

---

## 💻 Hardware & Dependencies

- **Python Version**: 3.9+ (Tested on Python 3.14)
- **PyTorch**: 2.0+ (Supports both CPU and CUDA GPU acceleration)
- **Torchvision**: Pretrained ResNet-50 weights
- **Dependencies**: `numpy`, `pandas`, `matplotlib`, `pillow`, `nltk`, `pyyaml`, `streamlit`, `tabulate`

### Installation

```bash
# 1. Clone repository or navigate to seminar folder
cd seminar_final

# 2. Install required packages
pip install -r requirements.txt
```

---

## 🗃️ Dataset: Flickr8k & Synthetic Benchmark

### Zero Data Leakage Guarantee
In image-captioning benchmarks like Flickr8k, each image is paired with 5 human reference captions. **Splitting must be performed strictly by unique image ID**, never by caption row. Splitting by individual captions causes identical image semantics to appear in both training and test sets, artificially inflating metric scores. Our `split_dataset.py` enforces strict partitioning by image ID (80% train, 10% validation, 10% test).

### Using the Full Flickr8k Dataset
1. Download Flickr8k images and captions:
   - Place images into: `data/Flickr8k/Images/`
   - Place captions into: `data/Flickr8k/captions.txt`
2. Run dataset splitting:
   ```bash
   python dataset/split_dataset.py --captions_file data/Flickr8k/captions.txt
   ```

### Synthetic Dataset Fallback
If the 1GB Flickr8k dataset is not yet downloaded, the pipeline automatically generates a functional synthetic dataset (`data/synthetic/`) with geometric shapes, distinct colors, and natural language descriptions. This guarantees **100% reproducibility and runnable verification on any laptop CPU**.

---

## 🧪 Testing and Verification Suite

Run all 19 unit, integration, and overfitting tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The test suite validates:
- **Dataset**: Image loading, caption cleaning, numericalization, dynamic batch padding, special token index integrity (`<PAD>=0`, `<SOS>=1`, `<EOS>=2`, `<UNK>=3`).
- **CNN Encoder**: Feature map dimensions $[B, 2048, 7, 7] \to [B, 49, 2048]$, linear projection $[B, 49, 256]$, and ResNet backbone parameter freezing.
- **Attention**: Causal masking ($M_{i,j} = -\infty$ for $j > i$), attention weight normalization ($\sum = 1.0$), and cross-attention sequence length flexibility.
- **Decoder & Gradients**: Non-zero gradient flow into trainable decoder weights, zero gradients to frozen CNN parameters, and optimizer parameter updates.
- **Overfitting Verification**: Confirms that on a tiny batch, the model reduces training loss by over 50%, proving gradient flow and target shifting correctness.

---

## 🚀 Execution Commands

### 1. Execute All Experiments (One-Click)
Runs training for both architectures, computes all 7 metrics, generates comparison tables, loss graphs, attention heatmaps, and report text:
```bash
python run_experiments.py
```

### 2. Train Models Individually
- **Train Primary CNN-Transformer**:
  ```bash
  python training/train.py --arch transformer --epochs 15 --batch_size 16
  ```
- **Train Baseline CNN-LSTM**:
  ```bash
  python training/train.py --arch lstm --epochs 15 --batch_size 16
  ```

### 3. Evaluate Checkpoint on Held-Out Test Set
```bash
python evaluation/evaluate.py --checkpoint checkpoints/best_transformer.pt --vocab checkpoints/vocab.json
```

### 4. Generate Publication-Quality Visualizations
- **Loss Curves**:
  ```bash
  python visualization/plot_losses.py
  ```
- **Qualitative Predictions & Error Analysis**:
  ```bash
  python visualization/sample_predictions.py
  ```

### 5. Launch Interactive Demonstration Web App
```bash
streamlit run app.py
```
*Opens interactive UI in your web browser. Upload an image, choose Greedy or Beam Search decoding, generate captions, and inspect spatial cross-attention heatmaps.*

### 6. Auto-Generate Report Summary & Markdown Text
```bash
python generate_report_results.py
```

---

## 📊 Measured Experimental Results

Results trained on the full Flickr8k training split (6,473 images / 32,365 captions), validated on 809 validation images (4,045 captions), and evaluated on all 810 held-out test images (4,050 captions):

### Architectural Comparison Matrix (`results/model_comparison.csv`)

| Model | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Trainable Params | Total Params | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Show, Attend and Tell (CNN-LSTM)** | **0.6066** | 0.1916 | 0.4374 | 0.4462 | 0.6830 | 3,580,175 | 27,088,207 | 3003.9s |
| **CNN-Transformer (Proposed)** | 0.6023 | **0.2029** | **0.4384** | **0.4471** | **0.6999** | 4,070,158 | 27,578,190 | **1671.6s** |

### Key Experimental Findings
1. **Multi-Head Transformer vs. Show, Attend and Tell**: Compared against the strong *Show, Attend and Tell* (Bahdanau Attention + LSTM) baseline, the proposed CNN-Transformer achieves higher sentence-level fluency (**BLEU-4: 0.2029 vs. 0.1916**, **CIDEr: 0.6999 vs. 0.6830**, **METEOR: 0.4384 vs. 0.4374**, **ROUGE-L: 0.4471 vs. 0.4462**) while training **1.8× faster** (`1671.6s` vs. `3003.9s`) due to parallel masked self-attention instead of sequential step-by-step `LSTMCell` recurrence.
2. **Minimum Training & Validation Loss**: Training loss converged down to **1.9663**, and validation loss decreased smoothly from **3.1583** (Epoch 1) down to **2.8225** (Epoch 4), with early stopping selecting the optimal checkpoint before overfitting.
3. **Full Vocabulary**: Vocabulary contains **3,673 unique tokens** (`min_freq=3`) built exclusively from the 32,365 training captions.

---

## 📐 Mathematical Formulation & Tensor Dimensions

For batch size $B$, image resolution $224 \times 224$, model dimension $d_{model} = 256$, number of heads $h = 4$, sequence length $T$, and vocabulary size $V$:

```
1. Input Image:                     [B, 3, 224, 224]
2. ResNet-50 layer4 output:         [B, 2048, 7, 7]
3. Permute & Flatten spatial grid:  [B, 49, 2048]
4. Linear Visual Projection:        [B, 49, 256]           (V_image = V_CNN * W_v + b_v)
5. Caption Token Embeddings:        [B, T-1, 256]          (Tokens * sqrt(d_model) + PE)
6. Masked Self-Attention Q, K, V:   [B, 4, T-1, 64]        (Derived from text tokens)
7. Visual Cross-Attention:
   - Queries (Q):                   [B, 4, T-1, 64]        (Derived from decoder text tokens)
   - Keys (K) & Values (V):         [B, 4, 49, 64]         (Derived from visual tokens)
   - Attention Weights:             [B, 4, T-1, 49]        (Softmax(Q K^T / sqrt(64)))
8. Output Logits:                   [B, T-1, V]            (Projected to vocabulary)
```

---

## 🔮 Future Work: Reinforcement Learning (SCST)

*Note: Reinforcement learning was deliberately not implemented in this project to keep the focus on supervised cross-modal attention.*

As documented in **Section 7 of the seminar report** and **Slide 18 of the presentation**:
- Supervised training minimizes token-level cross-entropy (maximum likelihood estimation). This suffers from **exposure bias** and optimizes individual word probabilities rather than sentence-level quality.
- A natural future extension is fine-tuning the trained Transformer policy using **Self-Critical Sequence Training (SCST)** (Rennie et al., 2017) with policy gradients:
  $$\nabla_\theta L_{RL}(\theta) = - \mathbb{E}_{\hat{y} \sim P_\theta} \left[ (r(\hat{y}) - r(y^*)) \nabla_\theta \log P_\theta(\hat{y}) \right]$$
  where $r(\hat{y})$ is the sequence-level CIDEr or BLEU reward of a sampled caption $\hat{y}$, and $r(y^*)$ is the baseline reward obtained via greedy decoding.
