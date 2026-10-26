
from functools import partial
from pathlib import Path

from collections import Counter
from torch.utils.data import Subset

from eval_datasets.pope import POPEDataset
from eval_datasets.perturbed import PerturbedDataset
from perturbations.blur import gaussian_blur

from models.qwen_vl import QwenVL
from evaluators.evaluator import Evaluator
from evaluators.experiment import (
    create_experiment_dir,
    save_experiment_config,
)
from evaluators.metrics import generate_metrics_report
from evaluators.reparse import reparse_results
from evaluators.robustness_report import generate_robustness_report
from evaluators.storage import load_completed_ids

from pathlib import Path

from evaluators.robustness_summary import summarize_blur_experiments
from visualizations.robustness_plot import plot_robustness


def main():
    seed = 42
    n_per_label = 5
    sample_count = 10
    radii = [0, 1, 3, 5]

    model_id = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

    # 1. 所有实验共享相同样本
    dataset = POPEDataset()
    sampled = dataset.stratified_subset(
        n_per_label=n_per_label,
        seed=seed
    )
    subset = dataset.stratified_subset(
        n_per_label=15,
        seed=seed
    )

    samples = [subset[i] for i in range(len(subset))]

    print("抽样总数:", len(samples))
    print("唯一 ID 数:", len(set(str(s.sample_id) for s in samples)))

    print("分组分布:", Counter(
        (s.category, s.ground_truth.lower())
        for s in samples
    ))

    sample_ids = [
        str(subset[i].sample_id)
        for i in range(len(subset))
    ]

    evaluator = Evaluator()
    model = None
    result_paths = {}

    # 2. 分别运行不同模糊强度
    for radius in radii:
        print(f"\n========== Blur radius={radius} ==========")

        output_dir = create_experiment_dir(
            dataset_name="pope90_v2",
            model_name=f"qwen3vl_blur{radius}",
            seed=seed
        )

        output_path = output_dir / "results.jsonl"

        config = {
            "dataset": "POPE",
            "model": "mlx-community/Qwen3-VL-2B-Instruct-4bit",
            "seed": seed,
            "n_per_label": 15,
            "sample_count": len(subset),
            "sample_ids": [
                str(subset[i].sample_id)
                for i in range(len(subset))
            ],
            "perturbation": "gaussian_blur",
            "radius": radius,
            "parser_version": "v2",
            "prompt": "original POPE question"
        }

        # 兼容之前已经创建的实验配置。
        # 新实验会完整记录 sample_ids；
        # 已有实验使用原来的配置进行一致性校验。
        config_path = output_dir / "config.json"
        if config_path.exists():
            import json

            with open(config_path, encoding="utf-8") as f:
                old_config = json.load(f)

            save_experiment_config(output_dir, old_config)

            # 至少检查影响实验可比性的关键参数
            for key in ("seed", "n_per_label", "radius"):
                if old_config.get(key) != config[key]:
                    raise ValueError(
                        f"旧实验配置不一致: {key}"
                    )

            if (
                "sample_ids" in old_config
                and old_config["sample_ids"] != sample_ids
            ):
                raise ValueError("旧实验的样本 ID 不一致")
        else:
            save_experiment_config(output_dir, config)

        test_dataset = PerturbedDataset(
            subset,
            transform=partial(
                gaussian_blur,
                radius=radius
            )
        )

        # 3. 判断是否需要模型推理
        completed = load_completed_ids(output_path)
        remaining = set(sample_ids) - completed

        if remaining:
            if model is None:
                model = QwenVL()

            evaluator.evaluate(
                model,
                test_dataset,
                output_path=output_path,
                max_samples=len(subset)
            )
        else:
            print("已有全部成功记录，跳过模型推理")

        # 4. 统一 Parser，保留原始结果
        reparsed_path = (
            output_dir / "results_parser_v2.jsonl"
        )

        # 离线重新解析成本很低，覆盖旧的派生文件即可
        if reparsed_path.exists():
            reparsed_path.unlink()

        reparse_results(
            output_path,
            reparsed_path
        )

        # 5. 生成统一版本的 Metrics
        report = generate_metrics_report(
            output_dir=output_dir,
            expected_samples=len(subset),
            results_path=reparsed_path,
            report_name="metrics_parser_v2.json"
        )

        result_paths[radius] = reparsed_path

        print("Accuracy:", report["overall"]["accuracy"])
        print("Invalid:", report["overall"]["invalid"])
        print("Coverage:", report["coverage"])

    # 6. 所有实验与 baseline 对比
    print("\n========== Robustness Summary ==========")
    baseline_path = (
            Path("results")
            / f"pope90_qwen3vl_blur0_seed{seed}"
            / "results.jsonl"
    )

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"找不到 Baseline 结果: {baseline_path}"
        )

    for radius in radii:
        if radius == 0:
            continue

        comparison = generate_robustness_report(
            baseline_path=baseline_path,
            perturbed_path=result_paths[radius],
            output_path=Path("results")
            / f"robustness_blur0_vs_blur{radius}_parser_v2.json"
        )

        stats = comparison["comparison"]

        print(f"\nBlur radius={radius}")
        print("Flip Rate:", stats["flip_rate"])
        print(
            "Invalid Transition Rate:",
            stats["invalid_transition_rate"]
        )
        print("Flipped IDs:", stats["flipped_ids"])
        print(
            "Became Invalid IDs:",
            stats["became_invalid_ids"]
        )

        summary = summarize_blur_experiments(
            results_root="results",
            radii=[0, 1, 3, 5],
            seed=42
        )

        Path("results/figures").mkdir(parents=True, exist_ok=True)

        plot_robustness(
            summary,
            output_path="results/figures/blur_robustness.png"
        )

def check_repeated_ids():
    from collections import defaultdict
    from eval_datasets.pope import POPEDataset

    dataset = POPEDataset()

    groups = defaultdict(list)

    for i, row in enumerate(dataset.dataset):
        groups[str(row["question_id"])].append(i)

    for question_id, indices in groups.items():
        if len(indices) > 1:
            print("question_id:", question_id)
            print("数据集行号:", indices)
            print("---")

if __name__ == "__main__":
    check_repeated_ids()
    main()
