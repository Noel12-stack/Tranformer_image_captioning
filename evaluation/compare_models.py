"""
Architectural Comparison Utility.

Compares CNN-Transformer vs CNN-LSTM Baseline (and alternative Transformer configs)
on identical test splits across:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR, ROUGE-L, CIDEr
- Number of trainable parameters
- Total parameter count
- Training time per epoch

Saves:
- results/model_comparison.csv
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any


def compare_architectures(
    model_records: List[Dict[str, Any]],
    output_path: str = "results/model_comparison.csv",
) -> pd.DataFrame:
    """
    Consolidates evaluation metrics and architectural stats across evaluated models.
    Marks any unexecuted models or missing metrics as 'Not run' or 'Not available'.

    Args:
        model_records: List of dicts, each with keys:
            'Model', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr', 'Trainable Params', 'Training Time'
        output_path: Path to write comparison CSV.

    Returns:
        pd.DataFrame: Formatted comparison table.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    columns = [
        "Model",
        "BLEU-1",
        "BLEU-4",
        "METEOR",
        "ROUGE-L",
        "CIDEr",
        "Trainable Params",
        "Total Params",
        "Training Time (s)",
        "Status",
    ]

    formatted_rows = []
    for rec in model_records:
        row = {
            "Model": rec.get("Model", "Unknown"),
            "BLEU-1": rec.get("BLEU-1", "Not available"),
            "BLEU-4": rec.get("BLEU-4", "Not available"),
            "METEOR": rec.get("METEOR", "Not available"),
            "ROUGE-L": rec.get("ROUGE-L", "Not available"),
            "CIDEr": rec.get("CIDEr", "Not available"),
            "Trainable Params": rec.get("Trainable Params", "Not available"),
            "Total Params": rec.get("Total Params", "Not available"),
            "Training Time (s)": rec.get("Training Time (s)", "Not available"),
            "Status": rec.get("Status", "Evaluated"),
        }
        formatted_rows.append(row)

    df = pd.DataFrame(formatted_rows, columns=columns)
    df.to_csv(output_path, index=False)
    print("\n" + "=" * 80)
    print("ARCHITECTURAL COMPARISON MATRIX:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"Comparison saved to: {output_path}\n")

    return df
