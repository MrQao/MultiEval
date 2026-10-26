from .result import EvalResult
from .parser import parse_yes_no
from .storage import save_result, load_completed_ids

class Evaluator:

    def evaluate_one(self, model, sample):
        try:
            response = model.generate(
                sample.image,
                sample.question
            )

            return EvalResult(
                sample_id=sample.sample_id,
                model_name=response.model_name,
                ground_truth=sample.ground_truth,
                raw_answer=response.answer,
                prediction=parse_yes_no(response.answer),
                inference_time=response.inference_time,
                status="success",
                category=sample.category
            )

        except Exception as e:
            return EvalResult(
                sample_id=sample.sample_id,
                model_name=type(model).__name__,
                ground_truth=sample.ground_truth,
                raw_answer="",
                prediction=None,
                inference_time=0.0,
                status="error",
                category=sample.category
            )

    def evaluate(self, model, dataset, output_path, max_samples=20):
        completed_ids = load_completed_ids(output_path)
        attempt_count = 0
        success_count = 0

        for i in range(len(dataset)):
            if attempt_count >= max_samples:
                break

            sample = dataset[i]

            if sample.sample_id in completed_ids:
                continue

            result = self.evaluate_one(model, sample)
            save_result(result, output_path)

            attempt_count += 1

            if result.status == "success":
                success_count += 1
                completed_ids.add(sample.sample_id)

            print(
                f"[{attempt_count}/{max_samples}] "
                f"id={sample.sample_id}, "
                f"status={result.status}, "
                f"prediction={result.prediction}"
            )

        print(f"本次尝试: {attempt_count}")
        print(f"本次成功: {success_count}")