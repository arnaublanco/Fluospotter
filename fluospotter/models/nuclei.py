"""segmentation class."""
import pdb

import numpy as np
import torch
from ..datasets import Dataset
from ..losses import combined_f1_rmse, f1_score, rmse
from ._models import Model
from ..networks.unet import CustomUNet
from ..training import train_model, evaluate
from ..io import check_configuration_file
from ..metrics import compute_segmentation_metrics
from ..data import get_loaders_test


class SegmentationModel(Model):
    """Class to segment nuclei; see base class."""

    def __init__(self, pretrained=None, model_name='small_unet_3d', in_channels=1, n_classes=2, patch_size=(64, 64, 16),
                 configuration={}, **kwargs):
        super().__init__(**kwargs)
        self.network = CustomUNet(model_name=model_name, pretrained=pretrained, in_c=in_channels, n_classes=n_classes,
                                  patch_size=patch_size).model
        self.model_name = model_name
        self.cfg = check_configuration_file(configuration)

    @property
    def metrics(self) -> list:
        """List of all metrics recorded during training."""
        return [
            f1_score,
            rmse,
            combined_f1_rmse,
        ]

    def train(
            self, dataset: Dataset, augment_val: bool = True, callbacks: list = None,
    ) -> None:
        if not dataset.training:
            raise ValueError('Dataset does not contain training data.')

        train_model(dataset=dataset, model=self, model_type="segmentation")

    def predict(self, dataset: Dataset) -> None:
        data_path, labels_path = dataset.segmentation_data_test()
        test_loaders = get_loaders_test(data_path=data_path, labels_path=labels_path,
                                        n_samples=int(self.cfg["n_samples"]), neg_samples=int(self.cfg["neg_samples"]),
                                        patch_size=tuple(map(int, self.cfg["patch_size"].split('/'))),
                                        num_workers=int(self.cfg["num_workers"]),
                                        depth_last=bool(self.cfg["depth_last"]), n_classes=int(self.cfg["n_classes"]))
        evaluate(self.network,test_loaders)

    def predict_on_image(self, image: np.ndarray) -> np.ndarray:
        """Predict on a single input image."""
        return self.network.predict(image[None, ..., None], batch_size=1).squeeze()

    def evaluate(self, dataset: Dataset, display: bool = True) -> None:
        test_loaders = get_loaders_test(data_path=dataset.data_dir, labels_path=dataset.segmentation_dir,
                                        n_samples=int(self.cfg["n_samples"]), neg_samples=int(self.cfg["neg_samples"]),
                                        patch_size=tuple(map(int, self.cfg["patch_size"].split('/'))),
                                        num_workers=int(self.cfg["num_workers"]),
                                        depth_last=bool(self.cfg["depth_last"]), n_classes=int(self.cfg["n_classes"]))
        predicted, actual = evaluate(self.network, test_loaders)
        compute_segmentation_metrics(predicted=predicted, actual=actual)
