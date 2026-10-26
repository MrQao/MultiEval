from datasets import load_dataset

pope = load_dataset(
    "oplex/POPE",
    split="test"
)

sample = pope[0]

print(sample.keys())
print(sample["question"])
print(sample["answer"])
print(type(sample["image"]))