from abc import ABC, abstractmethod
from dataclasses import dataclass
from mlx_vlm import load


class BaseVLM(ABC):

    @abstractmethod
    def generate(self, image, question):
        pass

class TestVLM(BaseVLM):
    def generate(self, image, question):
        pass
    pass


@dataclass
class VLMResponse:
    answer: str
    model_name: str
    inference_time: float

model = TestVLM()
result = VLMResponse(
    answer="Yes",
    model_name="Qwen-VL",
    inference_time=0.5
)

def accuracy(ground_truth, predictions):
    correct = 0

    for gt, pred in zip(ground_truth, predictions):
        if gt == pred:
            correct += 1

    return correct / len(ground_truth)

def inference_with_retry(model, image, question, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = model.generate(image, question)
            return {
                "status": "success",
                "response": response,
                "attempts": attempt + 1
            }

        except TimeoutError as e:
            if attempt == max_retries - 1:
                return {
                    "status": "timeout",
                    "response": None,
                    "error": str(e),
                    "attempts": max_retries
                }


model_id = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

model, processor = load(model_id)

print("Qwen loaded successfully!")