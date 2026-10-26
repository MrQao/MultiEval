from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_id = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

# 加载模型
model, processor = load(model_id)
config = load_config(model_id)

# 准备输入
image = ["./data/test.jpg"]
question = "Is there a cat in this image? Answer yes or no."

# 构建符合 Qwen 格式的 prompt
formatted_prompt = apply_chat_template(
    processor,
    config,
    question,
    num_images=len(image)
)

# 执行推理
response = generate(
    model,
    processor,
    formatted_prompt,
    image,
    max_tokens=64,
    verbose=False
)

print(response)