from datasets import load_dataset
from torch.utils.data import Subset
from .base import BaseDataset, EvalSample
import random
from collections import defaultdict

class POPEDataset(BaseDataset):
    def __init__(self):
        self.dataset = load_dataset(
            "oplex/POPE",
            split="test"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> EvalSample:
        row = self.dataset[index]

        return EvalSample(
            sample_id=f"pope_{index}",
            image=row["image"],
            question=row["question"],
            ground_truth=row["answer"],
            category=row["category"]
        )

    def sample_by_category(self, n_per_category=10):
        groups = {}

        for i in range(len(self.dataset)):
            category = self.dataset[i]["category"]

            if category not in groups:
                groups[category] = []

            if len(groups[category]) < n_per_category:
                groups[category].append(i)

        return groups

    def stratified_subset(self, n_per_category=10):
        groups = self.sample_by_category(n_per_category)

        indices = [
            index
            for group in groups.values()
            for index in group
        ]

        return Subset(self, indices)

    def stratified_subset(self, n_per_label=5, seed=42):
        groups = defaultdict(list)

        for i in range(len(self.dataset)):
            row = self.dataset[i]

            key = (row["category"], row["answer"].lower())
            groups[key].append(i)

        rng = random.Random(seed)
        indices = []

        for key, group in groups.items():
            if len(group) < n_per_label:
                raise ValueError(f"{key} 样本数量不足")

            indices.extend(rng.sample(group, n_per_label))

        rng.shuffle(indices)

        return Subset(self, indices)