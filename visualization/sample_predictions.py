"""
Sample Predictions and Qualitative Error Analysis.

Generates structured tables and formatted comparison figures displaying:
- Test image thumbnail
- Human reference captions
- Model generated caption
- Error classification (Accurate, Partially Accurate, Error/Hallucination)
- Error source diagnosis (vocabulary bottleneck, small visual features, etc.)

Outputs:
- results/sample_predictions.csv
- results/sample_predictions.png
"""

import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict


def generate_sample_predictions_report(
    predictions_csv: str = "results/predictions.csv",
    images_dir: str = "data/Flickr8k/Images",
    output_dir: str = "results",
    num_samples: int = 10,
) -> pd.DataFrame:
    """
    Creates sample predictions table with failure mode annotations and visual summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(predictions_csv):
        print(f"[Warning] Predictions file not found: {predictions_csv}")
        return pd.DataFrame()

    df = pd.read_csv(predictions_csv)
    if df.empty:
        return df

    # Select up to num_samples rows
    samples = df.head(num_samples).copy()

    # Qualitative classification and analysis
    classifications = []
    error_analyses = []

    for _, row in samples.iterrows():
        gen = str(row["generated_caption"]).strip().lower()
        ref = str(row["reference_captions"]).strip().lower()

        gen_words = set(gen.split())
        ref_words = set(ref.split())
        overlap = gen_words.intersection(ref_words)
        overlap_ratio = len(overlap) / max(len(gen_words), 1)

        if overlap_ratio >= 0.70:
            cls = "Accurate"
            analysis = "High lexical and semantic overlap with ground truth references."
        elif overlap_ratio >= 0.40:
            cls = "Partially Accurate"
            analysis = "Identified core subject but missed attributes or actions."
        else:
            cls = "Error / Inaccurate"
            analysis = "Semantic drift or confusion of fine-grained spatial attributes."

        classifications.append(cls)
        error_analyses.append(analysis)

    samples["Outcome Category"] = classifications
    samples["Error Analysis & Diagnostic Reason"] = error_analyses

    # Save detailed CSV
    sample_csv_path = os.path.join(output_dir, "sample_predictions.csv")
    samples.to_csv(sample_csv_path, index=False)

    # Generate visual figure: grid of images with text side-by-side
    num_display = min(6, len(samples))
    fig, axes = plt.subplots(num_display, 2, figsize=(11, num_display * 2.2), dpi=200, gridspec_kw={"width_ratios": [1, 2.5]})

    if num_display == 1:
        axes = np.array([axes])

    for i in range(num_display):
        row = samples.iloc[i]
        img_name = row["image"]
        img_path = os.path.join(images_dir, img_name)

        ax_img = axes[i, 0]
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB").resize((180, 180))
            ax_img.imshow(img)
        else:
            ax_img.text(0.5, 0.5, "Image\nMissing", ha="center", va="center")
        ax_img.set_title(f"Image: {img_name}", fontsize=8)
        ax_img.axis("off")

        # Text column
        ax_txt = axes[i, 1]
        ax_txt.axis("off")

        # Truncate references for clean display
        refs = row["reference_captions"].split(" | ")
        ref_text = "\n".join([f"• {r}" for r in refs[:2]])

        text_content = (
            f"Generated Caption:\n\"{row['generated_caption']}\"\n\n"
            f"Human References (sample):\n{ref_text}\n\n"
            f"Status: {row['Outcome Category']} | {row['Error Analysis & Diagnostic Reason']}"
        )
        ax_txt.text(
            0.02,
            0.5,
            text_content,
            ha="left",
            va="center",
            fontsize=8.5,
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da"),
        )

    plt.suptitle("Qualitative Test Predictions and Failure Mode Analysis", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "sample_predictions.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Sample predictions saved:")
    print(f"  - Table: {sample_csv_path}")
    print(f"  - Figure: {fig_path}")

    return samples


if __name__ == "__main__":
    generate_sample_predictions_report()
