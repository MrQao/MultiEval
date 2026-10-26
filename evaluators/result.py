from dataclasses import dataclass


@dataclass
class EvalResult:
    sample_id: str
    model_name: str
    ground_truth: str
    raw_answer: str
    prediction: str | None
    inference_time: float
    status: str
    category: str | None = None