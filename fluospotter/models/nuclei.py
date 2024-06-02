"""segmentation class."""

import functools
import numpy as np
from ..losses import combined_f1_rmse, f1_score, rmse
from ._models import Model
from ..networks.unet import CustomUNet


class SegmentationModel(Model):
    """Class to segment nuclei; see base class."""

    def __init__(self, pretrained=None, model_name='small_unet_3d', in_channels=1, n_classes=2, patch_size=(64, 64, 16),
                 **kwargs):
        super().__init__(**kwargs)
        self.network = CustomUNet(model_name=model_name, pretrained=pretrained, in_c=in_channels, n_classes=n_classes,
                                  patch_size=patch_size).model

    @property
    def metrics(self) -> list:
        """List of all metrics recorded during training."""
        return [
            f1_score,
            rmse,
            combined_f1_rmse,
        ]

    def predict_on_image(self, image: np.ndarray) -> np.ndarray:
        """Predict on a single input image."""
        return self.network.predict(image[None, ..., None], batch_size=1).squeeze()
