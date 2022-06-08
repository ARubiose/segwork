""" Multiclass segmentation on image not implemented

Need to ignore_index attribute for clarity
"""

import abc
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np

from segwork.data.dataset import SegmentationDataset

class Visualizer(abc.ABC):
    """Class to visualize images and segmentation masks from Dataset objects"""

    def __init__(self, dataset: torch.utils.Dataset, colors):
        self.dataset = dataset

    def show(self, index, transform = None, target_transform = None):
        """Show images"""

    def draw_segmentation_mask(self):
        pass

    def visualize_dataset(self, index , dataset_dir, split='test',):

        print(f'Loading split {split} of dataset from {dataset_dir}...')
        #Only dataset so far
        dataset:Dataset
        # dataset = dataset(data_dir=dataset_dir, split=split)
        
        #Dataset information
        print(f'\nDataset information for {getattr(self.dataset)}')
        print(f'Number of classes: {dataset.n_classes_without_void}')
        print(f'Classes: {dataset.class_names_without_void}')
        # print(f'Colors: {dataset.class_colors_without_void}')

        visualizer = DatasetVisualizer(dataset)

        idx = index
        image = dataset[idx]['image']
        depth = dataset[idx]['depth']
        mask = dataset[idx]['label']

        visualizer.visualize_all(image, depth, mask)


class DatasetVisualizer():
    """
    Dataset visualizer
    

    """
    def __init__(self, dataset: torch.utils.Dataset, backend=None):
        self._dataset = dataset
        self._colors = dataset.class_colors
        self._classes = dataset.class_names
        self._get_color_map()
        self.idx = 0

    @property
    def dataset(self):
        return self._dataset

    def _get_color_map(self):
        self._map = self._colors.ListedColormap(self._colors / 255)
        self._bounds = np.arange(0, len(self._classes), 1, dtype='uint')
        self._norm = self._colors.Normalize(vmin=0, vmax=self._map.N-1)

    def get_mask(self, idx):
        return self.dataset[idx]['label']

    def get_image(self, idx):
        return self.dataset[idx]['image']

    def get_depth(self, idx):
        return self.dataset[idx]['depth']

    def print_dataset_information(self):
        print('\nDataset information')
        print(f'Dataset length ({self.dataset.split}): {len(self.dataset)}')
        print(f'Number of classes: {len(self._classes)}')
        print(f'Classes: {self._classes}')

    def visualize_all(self, idx=0):

        self.idx = idx - 1
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        fig.canvas.mpl_connect('button_press_event', lambda event : paint_images(event, self, axs))

        def paint_images(event, self, axs):

            fig.suptitle(f'Visualizing image {self.idx}', fontsize=16)
            self.idx += 1

            image = self.get_image(self.idx)
            depth = self.get_depth(self.idx)
            mask = self.get_mask(self.idx)

            # RGB Image
            axs[0].set_title(label='RGB Image', loc='left')
            axs[0].imshow(image)

            # Depth map
            axs[1].set_title(label='Depth Image', loc='left')
            axs[1].imshow(depth)

            # Mask
            axs[2].set_title(label='Mask', loc='left')
            mask_img = axs[2].imshow(mask, cmap=self._map, norm=self._norm)
            # cbar = fig.colorbar(mask_img, ax=axs[2], ticks=self._bounds, orientation="horizontal")
            # cbar.ax.set_xticklabels(self._classes, rotation=45)

            plt.show()
              

        paint_images(None, self, axs)

    def visualize_mask(self, mask, title=""):
        fig, axs = plt.subplots(2, 1, figsize=(15, 3))

        # Image show
        # (M, N) The values are mapped to colors using normalization and a colormap.
        mask_img = axs[0].imshow(mask, cmap=self._map, norm=self._norm)

        # Color bar with names
        cbar = fig.colorbar(mask_img, cax=axs[1], ticks=self._bounds, orientation="horizontal")
        cbar.ax.set_xticklabels(self._classes, rotation=45)

        plt.title(title)
        plt.show()