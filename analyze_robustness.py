from pathlib import Path

from evaluators.storage import load_latest_results
from evaluators.metrics import calculate_metrics
from evaluators.robustness import compare_robustness
from collections import Counter

def main():
    radii = [0, 1, 3, 5]
    results_root = Path("results")

    experiments = {}

    for radius in radii:
        path = (
                results_root
                / f"pope90_v2_qwen3vl_blur{radius}_seed42"
                / "results_parser_v2.jsonl"
        )

        experiments[radius] = load_latest_results(path)

    # 1. 检查样本是否一致
    sample_sets = {
        radius: {str(r["sample_id"]) for r in results}
        for radius, results in experiments.items()
    }

    baseline_ids = sample_sets[0]

    for radius in radii:
        assert sample_sets[radius] == baseline_ids, (
            f"Blur-{radius} 的样本集合与 Baseline 不一致"
        )

    for radius in radii:
        print(
            f"Blur-{radius}:",
            len(sample_sets[radius]),
            "条样本"
        )

    print("四组共同样本数:", len(
        set.intersection(*sample_sets.values())
    ))

    print("样本一致性检查通过！")

    # 2. 计算各组指标
    base_map = {
        str(r["sample_id"]): r
        for r in experiments[0]
    }
    for radius in radii:
        results = experiments[radius]

        metrics = calculate_metrics(results)

        comparison = compare_robustness(
            experiments[0],
            results
        )

        results = experiments[radius]

        transitions = Counter()

        for r in results:
            sample_id = str(r["sample_id"])

            before = base_map[sample_id]["prediction"]
            after = r["prediction"]

            transitions[(before, after)] += 1

        print(f"\nBlur Radius = {radius}")
        print("Accuracy:", metrics["accuracy"])
        print("FPR:", metrics["fpr"])
        print("Recall:", metrics["recall"])
        print("Invalid:", metrics["invalid"])
        print("Flip Rate:", comparison["flip_rate"])

        print(f"\nBlur Radius = {radius}")
        print("预测转移:", dict(transitions))

    #计算反转前后是否变truth了
    baseline_map = {
        str(r["sample_id"]): r
        for r in experiments[0]
    }

    blur5 = experiments[5]
    flip_cases = []
    ground_truth_counts = Counter()

    for r in blur5:
        original = baseline_map[str(r["sample_id"])]

        if (
                original["prediction"] == "yes"
                and r["prediction"] == "no"
        ):
            ground_truth_counts[r["ground_truth"].lower()] += 1

            flip_cases.append({
                "sample_id": r["sample_id"],
                "ground_truth": r["ground_truth"],
                "before_answer": original["raw_answer"],
                "after_answer": r["raw_answer"]
            })

    print("Yes → No 的标准答案分布:", ground_truth_counts)

    for case in flip_cases:
        print(case["sample_id"], case["ground_truth"])



if __name__ == "__main__":
    main()