"""puncta_detection class."""

import pdb

from ..datasets import Dataset
from ._models import Model
from ..networks.unet import CustomUNet
from ..networks.resnet import CustomResNet
from ..training import train_model
from ..inference import evaluate
from ..io import check_puncta_configuration_file, save_metrics_csv
from ..data import get_loaders_test
import numpy as np


class SpotsModel(Model):
    """Class to predict spot localization; see base class."""

    def __init__(self, pretrained=None, overlapping_puncta=None, model_name='small_unet_3d',
                 configuration={}, **kwargs):
        super().__init__(**kwargs)
        self.cfg = check_puncta_configuration_file(configuration)
        self.cfg["model_type"] = "puncta_detection"
        self.network = CustomUNet(model_name=model_name, pretrained=pretrained, in_c=int(self.cfg["in_channels"]), n_classes=2,
                                  patch_size=tuple(map(int, self.cfg["patch_size"].split('/')))).model
        self.model_name = model_name
        if overlapping_puncta:
            self.refinement = CustomResNet(model_name="overlapping_puncta", pretrained=overlapping_puncta, in_c=int(self.cfg["in_channels"]), n_classes=2).model
        else:
            self.refinement = overlapping_puncta
        self.pretrained = pretrained

    def train(self, dataset: Dataset, **kwargs) -> None:
        if not dataset.training:
            raise ValueError('Dataset does not contain training data.')

        train_model(dataset=dataset, model=self)

    def predict(self, dataset: Dataset) -> None:
        test_loaders = get_loaders_test(data_path=dataset.data_dir, labels_path='',
                                        n_samples=int(self.cfg["n_samples"]), neg_samples=int(self.cfg["neg_samples"]),
                                        patch_size=tuple(map(int, self.cfg["patch_size"].split('/'))),
                                        num_workers=int(self.cfg["num_workers"]),
                                        depth_last=bool(self.cfg["depth_last"]), im_size=tuple(map(int, self.cfg["im_size"].split('/'))), n_classes=2)
        preds = evaluate(self, test_loaders, compute_metrics=False)
        return preds

    def predict_image(self, dataset: np.array) -> None:
        test_loaders = get_loaders_test(data_path=dataset, labels_path='',
                                        n_samples=int(self.cfg["n_samples"]),
                                        neg_samples=int(self.cfg["neg_samples"]),
                                        patch_size=tuple(map(int, self.cfg["patch_size"].split('/'))),
                                        num_workers=int(self.cfg["num_workers"]),
                                        depth_last=bool(self.cfg["depth_last"]), n_classes=2, im_size=tuple(map(int, self.cfg["im_size"].split('/'))), is_numpy=True)
        preds = evaluate(self, test_loaders, compute_metrics=False)
        return preds

    def predict_from_segmentation(self, volume: np.array, mask: np.array) -> None:

        test_loaders = get_loaders_test(data_path=dataset, labels_path='',
                                        n_samples=int(self.cfg["n_samples"]), neg_samples=int(self.cfg["neg_samples"]),
                                        patch_size=tuple(map(int, self.cfg["patch_size"].split('/'))),
                                        im_size=tuple(map(int, self.cfg["im_size"].split('/'))),
                                        num_workers=int(self.cfg["num_workers"]),
                                        instance_seg=bool(self.cfg["instance_seg"]),
                                        depth_last=bool(self.cfg["depth_last"]), n_classes=int(self.cfg["n_classes"]), is_numpy=True)
        predictions = evaluate(self, test_loaders, compute_metrics=False)
        return predictions

    def evaluate(self, dataset: Dataset, display: bool = True) -> None:
        test_loaders = get_loaders_test(data_path=dataset.data_dir, labels_path=dataset.spots_dir,
                                        n_samples=int(self.cfg["n_samples"]), neg_samples=int(self.cfg["neg_samples"]),
                                        patch_size=tuple(map(int, self.cfg["patch_size"].split('/'))),
                                        num_workers=int(self.cfg["num_workers"]),
                                        depth_last=bool(self.cfg["depth_last"]), n_classes=2, im_size=tuple(map(int, self.cfg["im_size"].split('/'))))
        metrics = evaluate(self, test_loaders, compute_metrics=True)
        save_metrics_csv(self.pretrained, metrics)