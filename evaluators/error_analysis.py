import json
from pathlib import Path
from collections import Counter

def save_error_annotations(annotations, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            annotations,
            f,
            indent=2,
            ensure_ascii=False
        )

def load_error_annotations(path):
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def count_error_labels(annotations):
    counts = Counter()

    for annotation in annotations.values():
        counts.update(set(annotation.get("labels", [])))

    return dict(counts)