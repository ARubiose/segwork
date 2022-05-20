""" Multiclass segmentation on image not implemented"""

from torch.utils.data import Dataset

class Visualizer:
    """Class to visualize images and segmentation masks from Dataset objects"""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def show(self, index, transform = None, target_transform = None):
        """Show images"""

    def draw_segmentation_mask(self):
        pass