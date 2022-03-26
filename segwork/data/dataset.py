from typing import Any, Callable, Optional, Tuple
from torch import Tensor
from torchvision.datasets import VisionDataset

class WeightCalculator:
    pass

class SegmentationDataset(VisionDataset):
    """_summary_

    Args:
        VisionDataset (_type_): _description_
    """
    def __init__(
        self,
        root: str,
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        # split
    ):
        super().__init__(root, transforms, transform, target_transform)

    def __getitem__(self, index: int) -> Tuple[str, str, ]:
        pass

    def __len__(self):
        pass

    def compute_class_weights(self, calculator:WeightCalculator):
        pass


