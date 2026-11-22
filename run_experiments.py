"""
End-to-End Experiment Execution Pipeline.

Executes:
1. Supervised training of primary CNN-Transformer architecture
2. Supervised training of CNN-LSTM Baseline architecture
3. Test set evaluation across all 7 metrics (BLEU-1..4, METEOR, ROUGE-L, CIDEr)
4. Architectural comparison matrix generation
5. Publication-quality loss convergence plots
6. Real cross-attention spatial heatmap visualizations
7. Sample prediction error analysis
8. Automated seminar report summary generation
"""

import os
import time
import torch
import pandas as pd

from training.train import run_training
from evaluation.evaluate import run_evaluation
from evaluation.compare_models import compare_architectures
from evaluation.generate_captions import generate_predictions_for_split
from evaluation.metrics import evaluate_all_metrics
from dataset.vocabulary import Vocabulary
from dataset.split_dataset import parse_captions_file
from dataset.preprocessing import tokenize_caption
from models.lstm_baseline import CNNLSTMBaseline
from visualization.plot_losses import plot_loss_curves
from visualization.visualize_attention import visualize_cross_attention
from visualization.sample_predictions import generate_sample_predictions_report
from generate_report_results import generate_experiment_summary


def run_all_experiments(synthetic: bool = False, epochs: int = 12, batch_size: int = 64):
    print("=" * 80)
    print("STARTING FULL EXPERIMENTAL RUN FOR SEMINAR REPORT")
    print(f"Dataset mode: {'Synthetic (Flickr8k Schema)' if synthetic else 'Full Flickr8k'}")
    print(f"Epochs per model: {epochs} | Batch size: {batch_size}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # -------------------------------------------------------------
    # Experiment 1: Train CNN-Transformer Model
    # -------------------------------------------------------------
    print("\n>>> EXPERIMENT 1: Training CNN-Transformer Model...")
    t0 = time.time()
    tf_history, vocab, splits_dir, images_dir, captions_file = run_training(
        config_path="configs/config.yaml",
        arch="transformer",
        use_synthetic=synthetic,
        override_epochs=epochs,
        override_batch_size=batch_size,
    )
    tf_time = time.time() - t0

    # -------------------------------------------------------------
    # Experiment 2: Train CNN-LSTM Baseline Model
    # -------------------------------------------------------------
    print("\n>>> EXPERIMENT 2: Training CNN-LSTM Baseline Model...")
    t0 = time.time()
    lstm_history, _, _, _, _ = run_training(
        config_path="configs/config.yaml",
        arch="lstm",
        use_synthetic=synthetic,
        override_epochs=epochs,
        override_batch_size=batch_size,
    )
    lstm_time = time.time() - t0

    # -------------------------------------------------------------
    # Evaluation: Evaluate Primary CNN-Transformer on Test Split
    # -------------------------------------------------------------
    print("\n>>> EVALUATING CNN-Transformer on Test Split...")
    test_split_file = os.path.join(splits_dir, "test_images.txt")
    tf_metrics_df = run_evaluation(
        checkpoint_path="checkpoints/best_transformer.pt",
        vocab_path="checkpoints/vocab.json",
        images_dir=images_dir,
        captions_file=captions_file,
        test_split_file=test_split_file,
        results_dir="results",
        device=str(device),
        decoding_method="greedy",
    )
    tf_metrics = tf_metrics_df.iloc[0].to_dict()

    # -------------------------------------------------------------
    # Evaluation: Evaluate CNN-LSTM Baseline on Same Test Split
    # -------------------------------------------------------------
    print("\n>>> EVALUATING CNN-LSTM Baseline on Test Split...")
    with open(test_split_file, "r", encoding="utf-8") as f:
        test_images = [line.strip() for line in f if line.strip()]

    all_captions = parse_captions_file(captions_file)
    test_images = [img for img in test_images if img in all_captions]

    lstm_ckpt = torch.load("checkpoints/best_lstm.pt", map_location=device)
    lstm_model = CNNLSTMBaseline(
        vocab_size=len(vocab),
        embed_dim=256,
        hidden_dim=256,
        num_layers=1,
        dropout=0.0,
        pad_idx=vocab.PAD_IDX,
        sos_idx=vocab.SOS_IDX,
        eos_idx=vocab.EOS_IDX,
        pretrained_cnn=False,
        freeze_cnn=True,
    )
    lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
    lstm_model.to(device)
    lstm_model.eval()

    lstm_preds = generate_predictions_for_split(
        model=lstm_model,
        vocab=vocab,
        image_names=test_images,
        images_dir=images_dir,
        device=device,
        method="greedy",
    )

    lstm_refs_tok = [[tokenize_caption(rc) for rc in all_captions[img]] for img in test_images]
    lstm_hyps_tok = [tokenize_caption(lstm_preds[img]) for img in test_images]
    lstm_metrics = evaluate_all_metrics(lstm_refs_tok, lstm_hyps_tok)

    # -------------------------------------------------------------
    # Model Comparison Table
    # -------------------------------------------------------------
    print("\n>>> GENERATING Architectural Comparison...")
    # Parameter counts
    tf_ckpt = torch.load("checkpoints/best_transformer.pt", map_location="cpu")
    tf_total_params = sum(v.numel() for v in tf_ckpt["model_state_dict"].values())
    lstm_total_params = sum(v.numel() for v in lstm_ckpt["model_state_dict"].values())

    model_records = [
        {
            "Model": "Show, Attend and Tell (CNN-LSTM)",
            "BLEU-1": lstm_metrics.get("BLEU-1"),
            "BLEU-4": lstm_metrics.get("BLEU-4"),
            "METEOR": lstm_metrics.get("METEOR"),
            "ROUGE-L": lstm_metrics.get("ROUGE-L"),
            "CIDEr": lstm_metrics.get("CIDEr"),
            "Trainable Params": f"{max(0, lstm_total_params - 23508032):,}",
            "Total Params": f"{lstm_total_params:,}",
            "Training Time (s)": round(lstm_time, 1),
            "Status": "Evaluated",
        },
        {
            "Model": "CNN-Transformer (Proposed)",
            "BLEU-1": tf_metrics.get("BLEU-1"),
            "BLEU-4": tf_metrics.get("BLEU-4"),
            "METEOR": tf_metrics.get("METEOR"),
            "ROUGE-L": tf_metrics.get("ROUGE-L"),
            "CIDEr": tf_metrics.get("CIDEr"),
            "Trainable Params": f"{max(0, tf_total_params - 23508032):,}",
            "Total Params": f"{tf_total_params:,}",
            "Training Time (s)": round(tf_time, 1),
            "Status": "Evaluated",
        },
    ]
    compare_architectures(model_records, output_path="results/model_comparison.csv")

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    print("\n>>> GENERATING Loss Curves...")
    plot_loss_curves(
        history_path="results/training_history.csv",
        output_dir="results",
    )

    print("\n>>> GENERATING Attention Visualizations...")
    from models.image_captioner import ImageCaptioner
    tf_eval_model = ImageCaptioner(
        vocab_size=len(vocab),
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.0,
        pad_idx=vocab.PAD_IDX,
        sos_idx=vocab.SOS_IDX,
        eos_idx=vocab.EOS_IDX,
        pretrained_cnn=False,
        freeze_cnn=True,
    )
    tf_eval_model.load_state_dict(tf_ckpt["model_state_dict"])
    tf_eval_model.to(device)
    tf_eval_model.eval()

    # Visualize first 3 test images
    for sample_img in test_images[:3]:
        img_p = os.path.join(images_dir, sample_img)
        visualize_cross_attention(
            model=tf_eval_model,
            vocab=vocab,
            image_path=img_p,
            device=device,
        )

    print("\n>>> GENERATING Sample Predictions Report...")
    generate_sample_predictions_report(
        predictions_csv="results/predictions.csv",
        images_dir=images_dir,
        output_dir="results",
        num_samples=10,
    )

    print("\n>>> COMPILING Final Technical Experiment Summary...")
    generate_experiment_summary(results_dir="results", checkpoints_dir="checkpoints")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS SUCCESSFULLY EXECUTED AND SAVED TO `results/`!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()
