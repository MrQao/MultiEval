def compare_robustness(baseline, perturbed):
    base_map = {str(r["sample_id"]): r for r in baseline}
    pert_map = {str(r["sample_id"]): r for r in perturbed}

    shared_ids = base_map.keys() & pert_map.keys()

    valid_pairs = 0
    flipped = []
    became_invalid = []

    for sample_id in shared_ids:
        before = base_map[sample_id]["prediction"]
        after = pert_map[sample_id]["prediction"]

        if before in ("yes", "no") and after is None:
            became_invalid.append(sample_id)

        if before not in ("yes", "no"):
            continue
        if after not in ("yes", "no"):
            continue

        valid_pairs += 1

        if before != after:
            flipped.append(sample_id)

    return {
        "paired_samples": len(shared_ids),
        "valid_pairs": valid_pairs,
        "flip_rate": len(flipped) / valid_pairs if valid_pairs else None,
        "flipped_ids": flipped,
        "invalid_transition_rate": (
            len(became_invalid) / len(shared_ids)
            if shared_ids else None
        ),
        "became_invalid_ids": became_invalid,
    }

def get_changed_cases(baseline, perturbed):
    base_map = {
        str(r["sample_id"]): r
        for r in baseline
    }

    changed = []

    for after in perturbed:
        sample_id = str(after["sample_id"])

        if sample_id not in base_map:
            continue

        before = base_map[sample_id]

        if before["prediction"] != after["prediction"]:
            changed.append({
                "sample_id": sample_id,
                "ground_truth": before["ground_truth"],
                "before_prediction": before["prediction"],
                "after_prediction": after["prediction"],
                "before_answer": before["raw_answer"],
                "after_answer": after["raw_answer"],
            })

    return changed