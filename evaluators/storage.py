import json
from dataclasses import asdict
from pathlib import Path

def save_result(result, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

def load_completed_ids(path):
    path = Path(path)

    if not path.exists():
        return set()

    completed_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record["status"] == "success":
                completed_ids.add(record["sample_id"])

    return completed_ids

def load_latest_results(path):
    latest = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            sample_id = record["sample_id"]

            if (
                sample_id not in latest
                or record["status"] == "success"
            ):
                latest[sample_id] = record

    return list(latest.values())