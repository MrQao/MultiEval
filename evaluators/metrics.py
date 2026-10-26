from collections import defaultdict
import json
from pathlib import Path

from .storage import load_latest_results

def confusion_matrix(results):
    tp = tn = fp = fn = 0
    invalid = 0

    for result in results:
        gt = result["ground_truth"]
        pred = result["prediction"]

        if pred not in ("yes", "no"):
            invalid += 1
            continue

        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "yes" and pred == "no":
            fn += 1

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "invalid": invalid
    }

def calculate_metrics(results):
    cm = confusion_matrix(results)

    tp = cm["TP"]
    tn = cm["TN"]
    fp = cm["FP"]
    fn = cm["FN"]
    invalid = cm["invalid"]

    total = tp + tn + fp + fn + invalid

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0

    return {
        "accuracy": accuracy,
        "total": total,
        "invalid": invalid,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }

def calculate_metrics_by_category(results):
    groups = defaultdict(list)

    for result in results:
        groups[result["category"]].append(result)

    return {
        category: calculate_metrics(group)
        for category, group in groups.items()
    }

def generate_metrics_report(
    output_dir,
    expected_samples,
    results_path=None,
    report_name="metrics.json"
):
    output_dir = Path(output_dir)

    if results_path is None:
        results_path = output_dir / "results.jsonl"

    results = load_latest_results(results_path)

    metrics = calculate_metrics(results)
    by_category = calculate_metrics_by_category(results)

    report = {
        "source_file": str(results_path),
        "expected_samples": expected_samples,
        "evaluated_samples": len(results),
        "coverage": (
            len(results) / expected_samples
            if expected_samples else 0
        ),
        "is_complete": len(results) == expected_samples,
        "overall": metrics,
        "by_category": by_category
    }

    report_path = output_dir / report_name

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report

