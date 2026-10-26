import json
from pathlib import Path

from .storage import load_latest_results
from .robustness import compare_robustness, get_changed_cases

def generate_robustness_report(baseline_path, perturbed_path, output_path):
    baseline = load_latest_results(baseline_path)
    perturbed = load_latest_results(perturbed_path)

    comparison = compare_robustness(baseline, perturbed)
    changed_cases = get_changed_cases(baseline, perturbed)

    report = {
        "baseline_path": str(baseline_path),
        "perturbed_path": str(perturbed_path),
        "comparison": comparison,
        "changed_cases": changed_cases,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report