from pathlib import Path

from .storage import load_latest_results
from .metrics import calculate_metrics
from .robustness import compare_robustness

def summarize_blur_experiments(results_root, radii, seed=42):
    results_root = Path(results_root)
    summary = []

    baseline_path = (
        results_root
        / f"pope90_qwen3vl_blur0_seed{seed}"
        / "results_parser_v2.jsonl"
    )

    baseline = load_latest_results(baseline_path)

    for radius in radii:
        result_path = (
            results_root
            / f"pope90_qwen3vl_blur{radius}_seed{seed}"
            / "results_parser_v2.jsonl"
        )

        results = load_latest_results(result_path)

        metrics = calculate_metrics(results)
        comparison = compare_robustness(baseline, results)

        summary.append({
            "radius": radius,
            "accuracy": metrics["accuracy"],
            "invalid_rate": (
                metrics["invalid"] / metrics["total"]
                if metrics["total"] else 0
            ),
            "flip_rate": comparison["flip_rate"],
            "invalid_transition_rate": comparison["invalid_transition_rate"],
            "paired_samples": comparison["paired_samples"],
        })

    return summary