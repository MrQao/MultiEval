from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

import time
from .base import BaseVLM, VLMResponse


class QwenVL(BaseVLM):
    def __init__(self):
        self.model_id = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

        self.model, self.processor = load(self.model_id)
        self.config = load_config(self.model_id)

    def generate(self, image, question):
        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            question,
            num_images=1
        )

        start_time = time.perf_counter()

        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            [image],
            max_tokens=256,
            verbose=False
        )

        inference_time = time.perf_counter() - start_time

        return VLMResponse(
            answer=result.text,
            model_name=self.model_id,
            inference_time=inference_time
        )