from functools import partial

from eval_datasets.pope import POPEDataset
from models.qwen_vl import QwenVL
from evaluators.evaluator import Evaluator
from evaluators.experiment import (
    create_experiment_dir,
    save_experiment_config
)
from evaluators.metrics import generate_metrics_report
from evaluators.storage import load_latest_results
from evaluators.parser import parse_yes_no
from evaluators.badcase import extract_bad_cases
from evaluators.report import export_badcase_html
from evaluators.error_analysis import save_error_annotations
from perturbations.blur import gaussian_blur
from evaluators.error_analysis import (
    load_error_annotations,
    count_error_labels
)
from eval_datasets.perturbed import PerturbedDataset

def main():
    seed = 42
    n_per_label = 5

    # 1. 准备数据
    dataset = POPEDataset()
    small_dataset = dataset.stratified_subset(
        n_per_label=n_per_label,
        seed=seed
    )

    # 2. 创建实验目录
    output_dir = create_experiment_dir(
        dataset_name="pope",
        model_name="qwen3vl_2b_v001",
        seed=seed
    )

    # 3. 保存实验配置
    config = {
        "dataset": "POPE",
        "model": "mlx-community/Qwen3-VL-2B-Instruct-4bit",
        "seed": seed,
        "n_per_label": n_per_label,
        "max_samples": len(small_dataset),
        "prompt": "original POPE question"
    }

    save_experiment_config(output_dir, config)

    # 4. 模型推理
    model = QwenVL()
    evaluator = Evaluator()

    evaluator.evaluate(
        model,
        small_dataset,
        output_path=output_dir / "results.jsonl",
        max_samples=len(small_dataset)
    )

    # 5. 生成指标报告
    report = generate_metrics_report(
        output_dir,
        expected_samples=len(small_dataset)
    )

    print("总体指标:", report["overall"])
    print("分组指标:", report["by_category"])
    print("覆盖率:", report["coverage"])

    results = load_latest_results(
        output_dir / "results.jsonl"
    )

    for r in results:
        if r["prediction"] is None:
            print("原始回答:", repr(r["raw_answer"]))
            print("标准答案:", r["ground_truth"])
            print("---")

def use_storage():
    results = load_latest_results(
        "./results/pope_qwen3vl_2b_v001_seed42/results.jsonl"
    )

    for r in results:
        r["prediction"] = parse_yes_no(r["raw_answer"])

    bad_cases = extract_bad_cases(results)

    for case_type, samples in bad_cases.items():
        print(case_type, len(samples))

    for r in bad_cases["false_negative"]:
        print("样本 ID:", r["sample_id"])
        print("类别:", r["category"])
        print("原始回答:", r["raw_answer"][:300])
        print("---")

    dataset = POPEDataset()

    bad_ids = {
        str(r["sample_id"])
        for r in bad_cases["false_negative"]
    }

    export_badcase_html(
        bad_cases=bad_cases,
        dataset=dataset,
        output_path="results/pope_badcases.html"
    )

    annotations = {
        "123": {
            "labels": ["answer_contradiction"],
            "note": "先否认勺子存在，随后又描述看到了部分勺子"
        },
        "357": {
            "labels": ["answer_contradiction"],
            "note": "我也觉得没有，答案是错的吧"
        },
        "457": {
            "labels": ["answer_contradiction"],
            "note": "滑雪板和双板定义不同，不算错其实"
        },
        "815": {
            "labels": ["answer_contradiction"],
            "note": "猫被无视了或是被看成狗了，不好说"
        },
        "904": {
            "labels": ["answer_contradiction"],
            "note": "色差太大或太纯，看成画了，开始莫名其妙写诗，没回答问题"
        },
        "1003": {
            "labels": ["answer_contradiction"],
            "note": "水台太边缘了，没识别到"
        },
        "1729": {
            "labels": ["answer_contradiction"],
            "note": "是有条塑料手拉袋，但是看着不像包之类的，太离谱了，答案有问题"
        },
        "1840": {
            "labels": ["answer_contradiction"],
            "note": "没回答问题，可能是被截断了，挺啰嗦的"
        },
        "2619": {
            "labels": ["answer_contradiction"],
            "note": "答案有问题，确实不是手提强，这是个手提包"
        },
        "2661": {
            "labels": ["answer_contradiction"],
            "note": "答案有问题，确实没人"
        },

    }

    save_error_annotations(
        annotations,
        "results/pope_error_annotations.json"
    )

    annotations = load_error_annotations(
        "results/pope_error_annotations.json"
    )

    print("错误原因统计:", count_error_labels(annotations))

    export_badcase_html(
        bad_cases=bad_cases,
        dataset=dataset,
        output_path="results/pope_badcases.html",
        annotations=annotations
    )

def blur_test():
    dataset = POPEDataset()
    sample = dataset[0]

    for radius in [0, 1, 3, 5]:
        blurred = gaussian_blur(sample.image, radius)
        blurred.save(f"blur_{radius}.jpg")

def dataset_wrapper():
    dataset = POPEDataset()

    blur_dataset = PerturbedDataset(
        dataset,
        transform=partial(gaussian_blur, radius=3)
    )

    original = dataset[0]
    blurred = blur_dataset[0]

    print(original.sample_id, blurred.sample_id)
    print(original.ground_truth, blurred.ground_truth)

    original.image.show()
    blurred.image.show()

if __name__ == "__main__":
    # main()
    # use_storage()
    # blur_test()
    dataset_wrapper()