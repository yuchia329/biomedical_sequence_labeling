#!/usr/bin/env python3

import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute token-level precision, recall, f1, and accuracy (both per label and overall) for single-label tagging."
    )
    parser.add_argument("--pred", type=str, required=True,
                        help="Path to the predicted output (e.g. output.txt)")
    parser.add_argument("--gold", type=str, required=True,
                        help="Path to the gold label file (same format).")
    return parser.parse_args()

def load_conll_lines(file_path):
    """
    Loads a CoNLL-style file: each non-blank line has 'word\\tlabel'.
    Returns a list of lines (strings), skipping blanks.
    """
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # ignore blank lines
                lines.append(line)
    return lines

def compute_metrics(pred_lines, gold_lines):
    """
    Build a one-vs-rest confusion matrix for each label, then compute
    per-label, overall (micro-averaged), and overall (macro-averaged) precision, recall, f1, and accuracy.
    """
    if len(pred_lines) != len(gold_lines):
        raise ValueError(
            f"File length mismatch: pred={len(pred_lines)}, gold={len(gold_lines)}"
        )

    # 1. Gather all possible labels from predictions & gold
    all_labels = set()
    for p_line, g_line in zip(pred_lines, gold_lines):
        p_word, p_label = p_line.rsplit("\t", 1)
        g_word, g_label = g_line.rsplit("\t", 1)
        all_labels.add(p_label)
        all_labels.add(g_label)

    # 2. Initialize confusion matrix stats for each label
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    # 3. Fill confusion matrix counts
    for p_line, g_line in zip(pred_lines, gold_lines):
        p_word, p_label = p_line.rsplit("\t", 1)
        g_word, g_label = g_line.rsplit("\t", 1)

        for label in all_labels:
            if g_label == label and p_label == label:
                stats[label]["tp"] += 1
            elif g_label == label and p_label != label:
                stats[label]["fn"] += 1
            elif g_label != label and p_label == label:
                stats[label]["fp"] += 1
            else:
                stats[label]["tn"] += 1

    # 4. Compute per-label metrics
    per_label_results = {}
    for label in sorted(all_labels):
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]
        tn = stats[label]["tn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        per_label_results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

    # 5. Compute overall (micro-averaged) metrics by summing counts
    overall_tp = sum(stats[label]["tp"] for label in all_labels)
    overall_fp = sum(stats[label]["fp"] for label in all_labels)
    overall_fn = sum(stats[label]["fn"] for label in all_labels)
    overall_tn = sum(stats[label]["tn"] for label in all_labels)

    micro_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    micro_recall    = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    micro_f1        = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    micro_accuracy  = (overall_tp + overall_tn) / (overall_tp + overall_fp + overall_fn + overall_tn) if (overall_tp + overall_fp + overall_fn + overall_tn) > 0 else 0.0

    # 6. Compute overall (macro-averaged) metrics by averaging per-label metrics
    num_labels = len(per_label_results)
    macro_precision = sum(result["precision"] for result in per_label_results.values()) / num_labels if num_labels > 0 else 0.0
    macro_recall    = sum(result["recall"] for result in per_label_results.values()) / num_labels if num_labels > 0 else 0.0
    macro_f1        = sum(result["f1"] for result in per_label_results.values()) / num_labels if num_labels > 0 else 0.0
    macro_accuracy  = sum(result["accuracy"] for result in per_label_results.values()) / num_labels if num_labels > 0 else 0.0

    return per_label_results, micro_precision, micro_recall, micro_f1, micro_accuracy, macro_precision, macro_recall, macro_f1, macro_accuracy

def main():
    args = parse_args()
    pred_lines = load_conll_lines(args.pred)
    gold_lines = load_conll_lines(args.gold)

    (results, micro_p, micro_r, micro_f1, micro_acc,
     macro_p, macro_r, macro_f1, macro_acc) = compute_metrics(pred_lines, gold_lines)

    # Print per-label table
    print("== Per-Label Metrics ==")
    print("Label\t\tPrec\tRecall\tF1\t\tAccuracy")
    for label in sorted(results.keys()):
        prec = results[label]["precision"] * 100
        rec  = results[label]["recall"] * 100
        f1   = results[label]["f1"] * 100
        acc  = results[label]["accuracy"] * 100
        print(f"{label:12s}\t{prec:5.2f}\t{rec:6.2f}\t{f1:6.2f}\t{acc:6.2f}")

    # Print overall (micro-averaged) metrics
    print("\n== Overall (Micro-Averaged) Metrics ==")
    print(f"Precision: {micro_p*100:.2f}%")
    print(f"Recall:    {micro_r*100:.2f}%")
    print(f"F1:        {micro_f1*100:.2f}%")
    print(f"Accuracy:  {micro_acc*100:.2f}%")

    # Print overall (macro-averaged) metrics
    print("\n== Overall (Macro-Averaged) Metrics ==")
    print(f"Precision: {macro_p*100:.2f}%")
    print(f"Recall:    {macro_r*100:.2f}%")
    print(f"F1:        {macro_f1*100:.2f}%")
    print(f"Accuracy:  {macro_acc*100:.2f}%")

if __name__ == "__main__":
    main()
