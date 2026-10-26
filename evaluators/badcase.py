def extract_bad_cases(results):
    bad_cases = {
        "false_positive": [],
        "false_negative": [],
        "invalid": []
    }

    for r in results:
        gt = r["ground_truth"].lower()
        pred = r["prediction"]

        if pred not in ("yes", "no"):
            bad_cases["invalid"].append(r)

        elif gt == "no" and pred == "yes":
            bad_cases["false_positive"].append(r)

        elif gt == "yes" and pred == "no":
            bad_cases["false_negative"].append(r)

    return bad_cases