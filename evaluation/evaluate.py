"""
Evaluation Script for Image Captioning Models.

Computes benchmark metrics on the unseen test set:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr

Outputs:
- results/metrics.json
- results/metrics.csv
- results/predictions.csv
"""

import os
import json
import argparse
import pandas as pd
import torch

from dataset.vocabulary import Vocabulary
from dataset.preprocessing import tokenize_caption, clean_caption
from dataset.split_dataset import parse_captions_file
from models.image_captioner import ImageCaptioner
from models.lstm_baseline import CNNLSTMBaseline
from evaluation.generate_captions import generate_predictions_for_split
from evaluation.metrics import evaluate_all_metrics, METRIC_DESCRIPTIONS


def run_evaluation(
    checkpoint_path: str,
    vocab_path: str,
    images_dir: str,
    captions_file: str,
    test_split_file: str,
    results_dir: str = "results",
    device: str = "cpu",
    decoding_method: str = "greedy",
    beam_size: int = 3,
) -> pd.DataFrame:
    """
    Executes full evaluation on test split and saves structured reports.
    """
    os.makedirs(results_dir, exist_ok=True)
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # 1. Load Vocabulary
    print(f"Loading vocabulary from: {vocab_path}")
    vocab = Vocabulary.load(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")

    # 2. Load Checkpoint and Reconstruct Model
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    arch_type = checkpoint.get("arch", "transformer")
    config = checkpoint.get("config", {})
    m_cfg = config.get("model", {})
    dec_cfg = m_cfg.get("decoder", {})

    if arch_type == "lstm":
        base_cfg = config.get("baseline", {})
        model = CNNLSTMBaseline(
            vocab_size=len(vocab),
            embed_dim=base_cfg.get("embed_dim", 256),
            hidden_dim=base_cfg.get("hidden_dim", 256),
            num_layers=base_cfg.get("num_layers", 1),
            dropout=base_cfg.get("dropout", 0.1),
            pad_idx=vocab.PAD_IDX,
            sos_idx=vocab.SOS_IDX,
            eos_idx=vocab.EOS_IDX,
            pretrained_cnn=False,
            freeze_cnn=True,
        )
    else:
        model = ImageCaptioner(
            vocab_size=len(vocab),
            d_model=dec_cfg.get("d_model", 256),
            nhead=dec_cfg.get("nhead", 4),
            num_layers=dec_cfg.get("num_layers", 2),
            dim_feedforward=dec_cfg.get("dim_feedforward", 512),
            dropout=dec_cfg.get("dropout", 0.1),
            pad_idx=vocab.PAD_IDX,
            sos_idx=vocab.SOS_IDX,
            eos_idx=vocab.EOS_IDX,
            pretrained_cnn=False,
            freeze_cnn=True,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()

    # 3. Load Test Images & Ground Truth Reference Captions
    if not os.path.exists(test_split_file):
        raise FileNotFoundError(f"Test split file missing: {test_split_file}")

    with open(test_split_file, "r", encoding="utf-8") as f:
        test_images = [line.strip() for line in f if line.strip()]

    all_captions = parse_captions_file(captions_file)

    # Filter to only images present in both
    test_images = [img for img in test_images if img in all_captions]
    print(f"Evaluating on {len(test_images)} test images with {decoding_method} decoding...")

    # 4. Generate Predictions
    preds = generate_predictions_for_split(
        model=model,
        vocab=vocab,
        image_names=test_images,
        images_dir=images_dir,
        device=device_obj,
        method=decoding_method,
        beam_size=beam_size,
    )

    # 5. Prepare Tokenized Data for Metrics
    references_tokenized = []
    hypotheses_tokenized = []
    rows = []

    for img in test_images:
        ref_caps = all_captions[img]
        gen_cap = preds[img]

        # Tokenize references and hypothesis consistently
        ref_tokens = [tokenize_caption(rc) for rc in ref_caps]
        hyp_tokens = tokenize_caption(gen_cap)

        references_tokenized.append(ref_tokens)
        hypotheses_tokenized.append(hyp_tokens)

        rows.append({
            "image": img,
            "generated_caption": gen_cap,
            "reference_captions": " | ".join(ref_caps),
        })

    # 6. Compute Evaluation Metrics
    metrics = evaluate_all_metrics(references_tokenized, hypotheses_tokenized)

    # 7. Save outputs
    # Save metrics JSON & CSV
    metrics_json_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    df_metrics = pd.DataFrame([metrics])
    metrics_csv_path = os.path.join(results_dir, "metrics.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)

    # Save detailed predictions CSV
    df_preds = pd.DataFrame(rows)
    preds_csv_path = os.path.join(results_dir, "predictions.csv")
    df_preds.to_csv(preds_csv_path, index=False)

    # Display console report
    print("\n" + "=" * 65)
    print("TEST SET EVALUATION RESULTS:")
    print("=" * 65)
    for k, v in metrics.items():
        desc = METRIC_DESCRIPTIONS.get(k, {}).get("full_name", "")
        print(f"  {k:<10}: {v:>8.4f}   ({desc})")
    print("=" * 65)
    print(f"Metrics saved to    : {metrics_csv_path}")
    print(f"Predictions saved to: {preds_csv_path}")
    print("=" * 65 + "\n")

    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Model on Test Split")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.json")
    parser.add_argument("--images_dir", type=str, default="data/Flickr8k/Images")
    parser.add_argument("--captions_file", type=str, default="data/Flickr8k/captions.txt")
    parser.add_argument("--test_split", type=str, default="data/Flickr8k/splits/test_images.txt")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=3)
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        images_dir=args.images_dir,
        captions_file=args.captions_file,
        test_split_file=args.test_split,
        results_dir=args.results_dir,
        decoding_method=args.decoding,
        beam_size=args.beam_size,
    )
