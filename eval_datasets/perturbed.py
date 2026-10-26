from dataclasses import replace

from .base import BaseDataset

class PerturbedDataset(BaseDataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        new_image = self.transform(sample.image)

        return replace(
            sample,
            image=new_image
        )