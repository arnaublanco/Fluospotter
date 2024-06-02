"""Dataset preparation functions."""

from typing import List, Tuple, Union, Dict
import os
from tifffile import imread

# List of currently supported image file extensions.
EXTENSIONS = ("tif", "tiff")


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
    files = [file for file in os.listdir(fname) if file.lower().endswith(EXTENSIONS)]
    size = None
    for f in files:
        try:
            file = imread(os.path.join(fname, f))
            if size is None:
                size = file.shape
            elif size != file.shape:
                raise ValueError(f"Files do not coincide in shape. {f} has size {file.shape}, expected {size}")
        except Exception as e:
            raise ValueError(f"Could not load {os.path.join(fname, f)}.") from e
    return files


def validate_data_and_labels(data: Dict[str, List[str]], labels: Dict[str, List[str]]) -> bool:
    for key in ['train', 'valid', 'test']:
        if remove_extension(data[key]) != remove_extension(labels[key]):
            return False
    return True


def remove_extension(file_list):
    return {os.path.splitext(file)[0] for file in file_list}


'''
def load_model(fname: Union[str, "os.PathLike[str]"]) -> tf.keras.models.Model:
    """Import a deepBlink model from file."""
    if not os.path.isfile(fname):
        raise ValueError(f"File must exist - '{fname}' does not.")
    if os.path.splitext(fname)[-1] != ".h5":
        raise ValueError(f"File must be of type h5 - '{fname}' does not.")

    try:
        model = tf.keras.models.load_model(
            fname,
            custom_objects={
                "combined_bce_rmse": combined_bce_rmse,
                "combined_dice_rmse": combined_dice_rmse,
                "combined_f1_rmse": combined_f1_rmse,
                "f1_score": f1_score,
                "leaky_relu": tf.nn.leaky_relu,
                "rmse": rmse,
            },
        )
        return model
    except ValueError as error:
        raise ImportError(f"Model '{fname}' could not be imported.") from error


def load_prediction(fname: Union[str, "os.PathLike[str]"]) -> pd.DataFrame:
    """Import a prediction file (output from deepBlink predict) as pandas dataframe."""
    if not os.path.isfile(fname):
        raise ValueError(f"File must exist - '{fname}' does not.")
    df = pd.read_csv(fname)
    if any([c in df.columns for c in ["x [µm]", "y [µm]"]]):
        raise ValueError(
            "Predictions must be in pixels, not microns. "
            "Please use 'pixel-size' 1 in predict."
        )
    if not all([c in df.columns for c in ["x [px]", "y [px]"]]):
        raise ValueError("Prediction file must contain columns 'x [px]' and 'y [px]'.")
    return df

'''
