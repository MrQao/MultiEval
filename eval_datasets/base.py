from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from PIL import Image

@dataclass
class EvalSample:
    sample_id: str
    image: str | Image.Image
    question: str
    ground_truth: str
    options: Optional[list[str]] = None
    category: str | None = None

class BaseDataset(ABC):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index) -> EvalSample:
        pass