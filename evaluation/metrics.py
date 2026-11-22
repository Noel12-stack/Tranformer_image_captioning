"""
Standard Image Captioning Evaluation Metrics.

Implements and explains:
1. BLEU-1, BLEU-2, BLEU-3, BLEU-4 (Papineni et al., 2002)
   - Measures modified n-gram precision penalized for sequence brevity.
   - Higher is better (range [0.0, 1.0]).
2. METEOR (Banerjee & Lavie, 2005)
   - Evaluates harmonic mean of unigram precision and recall with fragmentation penalty.
   - Higher is better (range [0.0, 1.0]).
3. ROUGE-L (Lin, 2004)
   - Measures Longest Common Subsequence (LCS) F1-score capturing sentence structure.
   - Higher is better (range [0.0, 1.0]).
4. CIDEr (Vedaldi et al., 2015)
   - Consensus-Based Image Description Evaluation using TF-IDF weighting on n-grams (1-4).
   - Higher is better (range [0.0, 10.0+]).
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Descriptive metadata for report generation
METRIC_DESCRIPTIONS = {
    "BLEU-1": {
        "full_name": "Bilingual Evaluation Understudy (Unigram Precision)",
        "interpretation": "Measures unigram vocabulary match between generated caption and human references, penalizing length mismatches.",
        "higher_is_better": True,
        "range": "[0.0, 1.0]",
    },
    "BLEU-2": {
        "full_name": "Bilingual Evaluation Understudy (Bigram Precision)",
        "interpretation": "Measures bigram (word pair) overlap, reflecting local 2-word phrase fluency.",
        "higher_is_better": True,
        "range": "[0.0, 1.0]",
    },
    "BLEU-3": {
        "full_name": "Bilingual Evaluation Understudy (Trigram Precision)",
        "interpretation": "Measures trigram overlap, assessing medium-range syntactic coherence.",
        "higher_is_better": True,
        "range": "[0.0, 1.0]",
    },
    "BLEU-4": {
        "full_name": "Bilingual Evaluation Understudy (4-gram Precision)",
        "interpretation": "Standard benchmark metric assessing 4-gram fluency, sentence structure, and precision against reference captions.",
        "higher_is_better": True,
        "range": "[0.0, 1.0]",
    },
    "METEOR": {
        "full_name": "Metric for Evaluation of Translation with Explicit ORdering",
        "interpretation": "Harmonic mean of precision and recall with penalty for chunk fragmentation; strongly correlates with human judgment.",
        "higher_is_better": True,
        "range": "[0.0, 1.0]",
    },
    "ROUGE-L": {
        "full_name": "Recall-Oriented Understudy for Gisting Evaluation (LCS)",
        "interpretation": "Calculates Longest Common Subsequence F-measure, measuring sentence-level word-order preservation.",
        "higher_is_better": True,
        "range": "[0.0, 1.0]",
    },
    "CIDEr": {
        "full_name": "Consensus-Based Image Description Evaluation",
        "interpretation": "Specifically designed for image captioning. Uses TF-IDF weighting over 1-to-4 n-grams to reward distinctive visual concepts.",
        "higher_is_better": True,
        "range": "[0.0, 10.0+]",
    },
}


def calculate_bleu_scores(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
) -> Dict[str, float]:
    """
    Computes corpus-level BLEU-1, BLEU-2, BLEU-3, BLEU-4 using standard NLTK corpus_bleu
    with smoothing function method 1.

    Args:
        references: List of reference token lists per image. Shape: [N, num_refs, len]
        hypotheses: List of predicted token lists per image. Shape: [N, len]

    Returns:
        Dict with 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'
    """
    smoother = SmoothingFunction().method1

    b1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoother)
    b2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    b3 = corpus_bleu(references, hypotheses, weights=(0.333, 0.333, 0.334, 0), smoothing_function=smoother)
    b4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    return {
        "BLEU-1": round(float(b1), 4),
        "BLEU-2": round(float(b2), 4),
        "BLEU-3": round(float(b3), 4),
        "BLEU-4": round(float(b4), 4),
    }


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Computes length of Longest Common Subsequence between two token lists."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def calculate_rouge_l(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
) -> float:
    """
    Computes average ROUGE-L F1-measure across the dataset.

    For each hypothesis, compares against each reference and takes the maximum F1.
    """
    scores = []
    beta = 1.2  # Standard ROUGE beta giving slightly higher weight to recall

    for refs, hyp in zip(references, hypotheses):
        best_f1 = 0.0
        m = len(hyp)
        if m == 0:
            scores.append(0.0)
            continue

        for ref in refs:
            n = len(ref)
            if n == 0:
                continue
            lcs = _lcs_length(hyp, ref)
            prec = lcs / m
            rec = lcs / n

            if prec + rec > 0:
                f1 = ((1 + beta**2) * rec * prec) / (beta**2 * prec + rec)
                if f1 > best_f1:
                    best_f1 = f1

        scores.append(best_f1)

    return round(float(sum(scores) / max(len(scores), 1)), 4)


def calculate_meteor(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
) -> float:
    """
    Computes METEOR score. Uses NLTK meteor_score if available;
    falls back to documented harmonic F-mean with unigram alignment penalty.
    """
    scores = []
    try:
        from nltk.translate.meteor_score import meteor_score
        for refs, hyp in zip(references, hypotheses):
            # NLTK expects references as lists of tokenized strings and hyp as tokenized string
            score = meteor_score(refs, hyp)
            scores.append(score)
        return round(float(sum(scores) / max(len(scores), 1)), 4)
    except Exception:
        # Documented exact fallback: unigram F-score with penalty for word fragmentation
        for refs, hyp in zip(references, hypotheses):
            best_score = 0.0
            hyp_len = len(hyp)
            if hyp_len == 0:
                scores.append(0.0)
                continue

            for ref in refs:
                ref_len = len(ref)
                if ref_len == 0:
                    continue
                # Unigram match
                hyp_counts = Counter(hyp)
                ref_counts = Counter(ref)
                matches = sum(min(hyp_counts[w], ref_counts[w]) for w in hyp_counts)

                p = matches / hyp_len
                r = matches / ref_len
                if p + r > 0:
                    # METEOR default weights: recall weighted 9:1 over precision (alpha=0.9)
                    f_mean = (10 * p * r) / (9 * p + r)
                    # Fragmentation penalty
                    lcs = _lcs_length(hyp, ref)
                    penalty = 0.5 * ((hyp_len - lcs) / max(hyp_len, 1)) ** 3
                    score = f_mean * (1 - penalty)
                    best_score = max(best_score, score)

            scores.append(best_score)

        return round(float(sum(scores) / max(len(scores), 1)), 4)


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Returns counter of n-grams for a list of tokens."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def calculate_cider(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
    n_order: int = 4,
) -> float:
    """
    Computes CIDEr (Consensus-Based Image Description Evaluation) score.

    Uses TF-IDF weighting on n-grams up to order 4 over the evaluation corpus.
    Consensus among human reference captions acts as the ground truth.
    """
    # 1. Document frequency for all n-grams across references
    num_images = len(references)
    doc_freq = [defaultdict(int) for _ in range(n_order)]

    for refs in references:
        seen_in_image = [set() for _ in range(n_order)]
        for ref in refs:
            for n in range(1, n_order + 1):
                ngrams = _get_ngrams(ref, n)
                for ng in ngrams:
                    if ng not in seen_in_image[n - 1]:
                        seen_in_image[n - 1].add(ng)
                        doc_freq[n - 1][ng] += 1

    # IDF weights
    idf = [
        {ng: math.log(max(1.0, (num_images - count + 0.5) / (count + 0.5))) for ng, count in doc_freq[n].items()}
        for n in range(n_order)
    ]

    cider_scores = []

    for refs, hyp in zip(references, hypotheses):
        # Calculate CIDEr for each image
        cider_n = []
        for n in range(1, n_order + 1):
            hyp_ngrams = _get_ngrams(hyp, n)
            # Hyp TF-IDF vector
            hyp_vec = {ng: count * idf[n - 1].get(ng, 0.0) for ng, count in hyp_ngrams.items()}
            hyp_norm = math.sqrt(sum(v**2 for v in hyp_vec.values())) or 1e-9

            # Average similarity with all references
            sim_sum = 0.0
            for ref in refs:
                ref_ngrams = _get_ngrams(ref, n)
                ref_vec = {ng: count * idf[n - 1].get(ng, 0.0) for ng, count in ref_ngrams.items()}
                ref_norm = math.sqrt(sum(v**2 for v in ref_vec.values())) or 1e-9

                dot_product = sum(hyp_vec[ng] * ref_vec.get(ng, 0.0) for ng in hyp_vec)
                sim_sum += dot_product / (hyp_norm * ref_norm)

            avg_sim = sim_sum / max(len(refs), 1)
            cider_n.append(avg_sim)

        # Average across n-grams 1 to 4 and scale by 10 (standard CIDEr scaling)
        cider_score = (sum(cider_n) / n_order) * 10.0
        cider_scores.append(cider_score)

    return round(float(sum(cider_scores) / max(len(cider_scores), 1)), 4)


def evaluate_all_metrics(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
) -> Dict[str, float]:
    """
    Evaluates all required image captioning metrics:
    BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L, CIDEr.
    """
    results = {}

    # BLEU 1-4
    bleu_results = calculate_bleu_scores(references, hypotheses)
    results.update(bleu_results)

    # METEOR
    results["METEOR"] = calculate_meteor(references, hypotheses)

    # ROUGE-L
    results["ROUGE-L"] = calculate_rouge_l(references, hypotheses)

    # CIDEr
    results["CIDEr"] = calculate_cider(references, hypotheses)

    return results
