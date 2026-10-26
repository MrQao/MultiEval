from eval_datasets.pope import POPEDataset
from models.qwen_vl import QwenVL

dataset = POPEDataset()
model = QwenVL()

sample = dataset[0]

response = model.generate(
    sample.image,
    sample.question
)

print("问题:", sample.question)
print("标准答案:", sample.ground_truth)
print("模型回答:", response.answer)