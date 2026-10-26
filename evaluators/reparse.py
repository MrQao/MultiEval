from dataclasses import replace
from pathlib import Path

from .parser import parse_yes_no
from .storage import load_latest_results, save_result
from .result import EvalResult

def reparse_results(input_path, output_path):
    results = load_latest_results(input_path)

    output_path = Path(output_path)

    if output_path.resolve() == Path(input_path).resolve():
        raise ValueError("不能覆盖原始结果文件")

    if output_path.exists():
        raise FileExistsError(
            f"目标文件已存在: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for record in results:
        result = EvalResult(**record)

        if result.status == "success":
            new_prediction = parse_yes_no(result.raw_answer)
        else:
            new_prediction = None

        new_result = replace(
            result,
            prediction=new_prediction
        )

        save_result(new_result, output_path)

    return output_path