"""Dataset preparation functions."""

from typing import List, Tuple, Union, Dict, Any
import os
from tifffile import imread
import pandas as pd

# List of currently supported image file extensions.
EXTENSIONS = ("tif", "tiff")


def get_extension(filename):
    file_extension = os.path.splitext(filename)[1]
    return file_extension


def check_puncta_configuration_file(cfg: Dict) -> Dict:
    params = [
        "model_name", "pretrained", "loss",
        "alpha", "batch_size", "acc_grad", "n_samples", "neg_samples",
        "ovft_check", "patch_size", "optimizer", "lr", "n_epochs", "vl_interval",
        "cyclical_lr", "metric", "num_workers", "depth_last", "in_channels", "overlapping_puncta", "full_resolution"
    ]
    values = [
        "small_unet_3d", False, "cedice",
        1.0, 1, 4, 12, 1,
        4, "48/256/256", "adam", 3e-4, 20, 5,
        True, "DSC", 0, False, 1, False, False]

    for p in range(len(params)):
        if not (params[p] in cfg):
            cfg[params[p]] = values[p]
    return cfg


def check_seg_configuration_file(cfg: Dict) -> Dict:
    """Ensures the segmentation configuration dictionary contains all necessary parameters.

        Args:
            cfg: Configuration dictionary.

        Returns:
            Updated configuration dictionary with default values added for missing parameters.
    """
    params = [
        "n_classes", "model_name", "pretrained", "loss1", "loss2", "shape_priors",
        "alpha1", "alpha2", "batch_size", "acc_grad", "n_samples", "neg_samples",
        "ovft_check", "patch_size", "optimizer", "lr", "n_epochs", "vl_interval",
        "cyclical_lr", "metric", "num_workers", "depth_last", "in_channels", "refinement", "full_resolution", "instance_seg"
    ]
    values = [
        3, "dynunet", False, "ce", "dice", [],
        1.0, 1.0, 1, 4, 12, 1,
        4, "40/128/128", "adam", 3e-4, 20, 5,
        True, "DSC", 0, False, 1, False, False, False]

    for p in range(len(params)):
        if not (params[p] in cfg):
            cfg[params[p]] = values[p]
    return cfg


def save_metrics_csv(path: str, metrics: Dict):
    df = pd.DataFrame(metrics)
    dir_path = os.path.dirname(path)
    filename = os.path.basename(os.path.splitext(path)[0]) + ".csv"
    df.to_csv(os.path.join(dir_path, filename))


def load_metrics_csv(path: str) -> dict:
    df = pd.read_csv(path)
    metrics = df.to_dict(orient='list')
    return metrics


def load_files(fname: str, training: bool = False) -> Dict[str, List[str]]:
    """Imports data for custom training and inference.

    Args:
        fname: Path to data files.
        training: Only return testing images and labels if false.

    Returns:
        A dictionary with keys 'train', 'valid', 'test' containing lists of file paths.

    Raises:
        ValueError: If not all datasets are found when training.
    """
    expected = ["train", "valid", "test"]
    check_if_folders_exist = {folder: os.path.isdir(os.path.join(fname, folder)) for folder in expected}

    data = {}

    if training:
        if all(check_if_folders_exist.values()):
            # Training
            for folder in expected:
                data[folder] = load_folder(os.path.join(fname, folder))
        else:
            missing_folders = [folder for folder, exists in check_if_folders_exist.items() if not exists]
            raise ValueError(f"{expected} must be present when training. Missing folders {missing_folders}.")
    else:
        # Inference
        for folder in ["test"]:
            if check_if_folders_exist["test"]:
                data["test"] = load_folder(os.path.join(fname, "test"))
            else:
                raise ValueError("Test folder must be present for inference.")

    return data


def load_folder(fname: str) -> List[str]:
    """Loads all image files from a given folder and checks their consistency in size.

        Args:
            fname: Path to the folder containing image files.

        Returns:
            A list of file paths.

        Raises:
            ValueError: If the files do not coincide in shape or cannot be loaded.
    """
    files = [os.path.join(fname, file) for file in os.listdir(fname) if file.lower().endswith(EXTENSIONS)]
    size = None
    for f in files:
        try:
            file = imread(f)
            if size is None:
                size = file.shape
            elif size != file.shape:
                raise ValueError(f"Files do not coincide in shape. {f} has size {file.shape}, expected {size}")
        except Exception as e:
            raise ValueError(f"Could not load {os.path.join(fname, f)}.") from e
    return files


def validate_data_and_labels(data: Dict[str, List[str]], labels: Dict[str, List[str]], training: bool = True) -> bool:
    """Validates that data and label files match in their basename (excluding extensions).

        Args:
            data: Dictionary containing lists of data file paths.
            labels: Dictionary containing lists of label file paths.

        Returns:
            True if data and label files match, False otherwise.
    """
    if training:
        keys = ['train', 'valid', 'test']
    else:
        keys = ['test']
    for key in keys:
        if remove_extension(data[key]) != remove_extension(labels[key]):
            return False
    return True


def remove_extension(file_list):
    """Removes the file extension from a list of file paths.

        Args:
            file_list: List of file paths.

        Returns:
            A set of file basenames without extensions.
    """
    return {os.path.basename(os.path.splitext(file)[0]) for file in file_list}
