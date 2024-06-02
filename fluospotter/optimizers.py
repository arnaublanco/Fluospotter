"""Optimizers are used to update weight parameters in a neural network.

This file contains functions to return standard or custom optimizers.
"""

from typing import Callable, Any
import torch.optim as optim


def adam(learning_rate: float) -> Callable[[Any], optim.Adam]:
    """PyTorch's Adam optimizer with a specified learning rate.

    Args:
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        Callable[[Any], optim.Adam]: A function that takes model parameters and returns an Adam optimizer.
    """
    return lambda params: optim.Adam(params, lr=learning_rate)


def sgd(learning_rate: float, momentum: float = 0.0) -> Callable[[Any], optim.SGD]:
    """PyTorch's SGD optimizer with a specified learning rate and momentum.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        momentum (float): The momentum factor.

    Returns:
        Callable[[Any], optim.SGD]: A function that takes model parameters and returns an SGD optimizer.
    """
    return lambda params: optim.SGD(params, lr=learning_rate, momentum=momentum)