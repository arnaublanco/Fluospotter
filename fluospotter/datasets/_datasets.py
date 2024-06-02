"""Dataset class."""

import os
from ..io import load_files, validate_data_and_labels


class Dataset:
    """Class for datasets.

    Args:
        data_dir (str): Absolute path to the data directory.
        segmentation_dir (str, optional): Absolute path to the segmentation labels directory.
        spots_dir (str, optional): Absolute path to the spots labels directory.
    """

    def __init__(self, data_dir: str, segmentation_dir: str = None, spots_dir: str = None):
        self.data_dir = data_dir
        self.segmentation_dir = segmentation_dir
        self.spots_dir = spots_dir
        self.load_data()

    def load_data(self) -> None:
        """Load dataset paths into memory and validate their consistency."""
        # Load the data files
        data = load_files(self.data_dir)
        self.x_train, self.x_valid, self.x_test = data["train"], data["valid"], data["test"]

        # Load and validate segmentation labels if available
        if self.segmentation_dir is not None:
            labels_seg = load_files(self.segmentation_dir)
            self.y_seg_train, self.y_seg_valid, self.y_seg_test = labels_seg["train"], labels_seg["valid"], labels_seg[
                "test"]
            if not validate_data_and_labels(data, labels_seg):
                raise ValueError("Data files and segmentation annotations do not coincide.")

        # Load and validate spots labels if available
        if self.spots_dir is not None:
            labels_spots = load_files(self.spots_dir)
            self.y_spots_train, self.y_spots_valid, self.y_spots_test = labels_spots["train"], labels_spots["valid"], \
            labels_spots["test"]
            if not validate_data_and_labels(data, labels_spots):
                raise ValueError("Data files and spots annotations do not coincide.")

        # Prepare and normalize the data
        self.prepare_data()
        self.normalize_dataset()

    def prepare_data(self):
        """Empty method to prepare or convert data."""

    def normalize_dataset(self):
        """Empty method to normalise images in the dataset."""
