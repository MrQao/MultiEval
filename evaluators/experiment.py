from pathlib import Path
import json

def create_experiment_dir(dataset_name, model_name, seed):
    experiment_id = f"{dataset_name}_{model_name}_seed{seed}"

    output_dir = Path("results") / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir

def save_experiment_config(output_dir, config):
    config_path = output_dir / "config.json"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            old_config = json.load(f)

        if old_config != config:
            raise ValueError(
                "实验配置不一致，请创建新的实验目录"
            )

        return

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)