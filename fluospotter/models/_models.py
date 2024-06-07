"""Model class, to be extended by specific types of models."""

from typing import Callable, Dict, List
import datetime

import numpy as np
import torch
from ..datasets import Dataset

DATESTRING = datetime.datetime.now().strftime("%Y%d%m_%H%M")

class Model:
    """Base class, to be subclassed by predictors for specific type of data.

    Args:
        dataset_args: Dataset arguments containing - version, cell_size, flip,
            illuminate, rotate, gaussian_noise, and translate.
        dataset_cls: Specific dataset class.
        network_args: Network arguments containing - n_channels.
        network_fn: Network function returning a built model.
        loss_fn: Loss function.
        optimizer_fn: Optimizer function.
        train_args: Training arguments containing - batch_size, epochs, learning_rate.
        pre_model: Loaded, pre-trained model to bypass a new network creation.

    Kwargs:
        batch_format_fn: Formatting function added in the specific model, e.g. spots.
        batch_augment_fn: Same as batch_format_fn for augmentation.
    """
    def __init__(self, **kwargs):
        pass

        #self.name = f"{DATESTRING}_{self.__class__.__name__}_{dataset_cls.name}_{network_fn.__name__}"
        #self.augmentation_args = augmentation_args
        #self.batch_augment_fn = kwargs.get("batch_augment_fn", None)
        #self.batch_format_fn = kwargs.get("batch_format_fn", None)
        #self.dataset_args = dataset_args
        #self.loss_fn = loss_fn
        #self.optimizer_fn = optimizer_fn
        #self.train_args = train_args
        #self.has_pre_model = pre_model is not None

    @property
    def metrics(self) -> list:
        """Return metrics."""
        return ["accuracy"]

    def train(
        self, dataset: Dataset, augment_val: bool = True, callbacks: list = None,
    ) -> None:
        """Training loop."""
        if callbacks is None:
            callbacks = []

        #if not self.has_pre_model:
        #    self.network.compile(
        #        loss=self.loss_fn,
        #        optimizer=self.optimizer_fn(float(self.train_args["learning_rate"])),
        #        metrics=self.metrics,
        #    )

        #train_sequence = Dataset(
        #    dataset.x_train,
        #    dataset.y_train,
        #    self.train_args["batch_size"],
        #    format_fn=self.batch_format_fn,
        #    augment_fn=self.batch_augment_fn,
        #    overfit=self.train_args["overfit"],
        #)

        #valid_sequence = Dataset(
        #    dataset.x_valid,
        #    dataset.y_valid,
        #    self.train_args["batch_size"],
        #    format_fn=self.batch_format_fn,
        #    augment_fn=self.batch_augment_fn if augment_val else None,
        #)

        #self.network.fit(
        #    train_sequence,
        #    epochs=self.train_args["epochs"],
        #    callbacks=callbacks,
        #    validation_data=valid_sequence,
        #    shuffle=True,
        #)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """Evaluate on images / masks and return l2 norm and f1 score."""
        if x.ndim < 4:
            x = np.expand_dims(x, -1)

        preds = self.network.predict(x)
        preds = np.float32(preds)
        y_float32 = np.float32(y)

        rmse_ = rmse(y_float32, preds) * self.dataset_args["cell_size"]
        f1_score_ = f1_score(y_float32, preds)

        return [f1_score_.numpy(), rmse_.numpy()]
