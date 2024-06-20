"""puncta_detection class."""

import pdb

import numpy as np
import torch
from ..datasets import Dataset
from ._models import Model
from ..networks.unet import CustomUNet
from ..training import train_model
from ..inference import evaluate
from ..io import check_puncta_configuration_file, save_metrics_csv
from ..metrics import compute_segmentation_metrics
from ..data import get_loaders_test, display_segmentation_metrics


class SpotsModel(Model):
    """Class to predict spot localization; see base class."""

    def __init__(self, pretrained=None, model_name='small_unet_3d',
                 configuration={}, **kwargs):
        super().__init__(**kwargs)
        self.cfg = check_puncta_configuration_file(configuration)
        self.cfg["model_type"] = "puncta_detection"
        self.network = CustomUNet(model_name=model_name, pretrained=pretrained, in_c=int(self.cfg["in_channels"]), n_classes=2,
                                  patch_size=tuple(map(int, self.cfg["patch_size"].split('/')))).model
        self.model_name = model_name
        self.pretrained = pretrained

    def train(self, dataset: Dataset, **kwargs) -> None:
        if not dataset.training:
            raise ValueError('Dataset does not contain training data.')

        train_model(dataset=dataset, model=self)

    def predict(self, dataset: Dataset) -> None:
        data_path, labels_path = dataset.segmentation_data_test()
        test_loaders = get_loaders_test(data_path=data_path, labels_path=labels_path,
                                        n_samples=int(self.cfg["n_samples"]), neg_samples=int(self.cfg["neg_samples"]),
                                        patch_size=tuple(map(int, self.cfg["patch_size"].split('/'))),
                                        num_workers=int(self.cfg["num_workers"]),
                                        depth_last=bool(self.cfg["depth_last"]), n_classes=2)
        evaluate(self, test_loaders, compute_metrics=False)

    def evaluate(self, dataset: Dataset, display: bool = True) -> None:
        test_loaders = get_loaders_test(data_path=dataset.data_dir, labels_path=dataset.segmentation_dir,
                                        n_samples=int(self.cfg["n_samples"]), neg_samples=int(self.cfg["neg_samples"]),
                                        patch_size=tuple(map(int, self.cfg["patch_size"].split('/'))),
                                        num_workers=int(self.cfg["num_workers"]),
                                        depth_last=bool(self.cfg["depth_last"]), n_classes=2, im_size=tuple(map(int, self.cfg["im_size"].split('/'))), instance_seg=bool(self.cfg["instance_seg"]))
        metrics = evaluate(self, test_loaders, compute_metrics=True)
        save_metrics_csv(self.pretrained, metrics)