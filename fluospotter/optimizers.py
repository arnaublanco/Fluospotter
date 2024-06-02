"""Optimizers are used to update weight parameters in a neural network.

This file contains functions to return standard or custom optimizers.
"""

from typing import Callable, Any, Dict
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def adam(learning_rate: float) -> Callable[[Any], optim.Adam]:
    """PyTorch's Adam optimizer with a specified learning rate.

    Args:
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        Callable[[Any], optim.Adam]: A function that takes model parameters and returns an Adam optimizer.
    """
    return lambda params: optim.Adam(params, lr=learning_rate)


def sgd(learning_rate: float, momentum: float = 0.0, weight_decay: float = 0.0) -> Callable[[Any], optim.SGD]:
    """PyTorch's SGD optimizer with a specified learning rate and momentum.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        momentum (float): The momentum factor.
        weight_decay (float): The weight decay factor.

    Returns:
        Callable[[Any], optim.SGD]: A function that takes model parameters and returns an SGD optimizer.
    """
    return lambda params: optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


def cosineAnnealingWarmRestarts(optimizer: optim.Optimizer, T_0: Any, eta_min: Any) -> lr_scheduler.CosineAnnealingWarmRestarts:
    """Cosine Annealing Warm Restarts scheduler.

    Args:
        optimizer (optim.Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        eta_min (float): Minimum learning rate. Default: 0.

    Returns:
        lr_scheduler.CosineAnnealingWarmRestarts: Cosine annealing scheduler with warm restarts.
    """
    return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)


def cosineAnnealingLR(optimizer: optim.Optimizer, T_max: Any, eta_min: Any) -> lr_scheduler.CosineAnnealingLR:
    """Cosine Annealing Learning Rate scheduler.

    Args:
        optimizer (optim.Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.

    Returns:
        lr_scheduler.CosineAnnealingLR: Cosine annealing learning rate scheduler.
    """
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


def get_optimizer(optimizer_name: str, optimizer_params: Dict[str, Any], params: Any) -> optim.Optimizer:
    """Get an optimizer by name.

    Args:
        optimizer_name (str): The name of the optimizer.
        optimizer_params (Dict[str, Any]): Parameters for the optimizer.
        params (Any): Model parameters to be optimized.

    Returns:
        optim.Optimizer: The optimizer instance.
    """
    if optimizer_name == 'adam':
        return adam(learning_rate=float(optimizer_params["learning_rate"]))(params)
    elif optimizer_name == 'sgd':
        return sgd(learning_rate=float(optimizer_params["learning_rate"]), momentum=float(optimizer_params.get("momentum", 0.0)), weight_decay=float(optimizer_params.get("weight_decay", 0.0)))(params)
    else:
        raise ValueError(f'Invalid optimizer name: {optimizer_name}')


def get_scheduler(scheduler: str, optimizer: optim.Optimizer, T: Any, eta_min: int = 0) -> lr_scheduler.LRScheduler:
    """Get a learning rate scheduler by name.

    Args:
        scheduler (str): The name of the scheduler.
        optimizer (optim.Optimizer): The optimizer instance.
        T (int): Parameter for the scheduler.
        eta_min (float): Minimum learning rate.

    Returns:
        lr_scheduler._LRScheduler: The learning rate scheduler instance.
    """
    if scheduler == 'cosineAnnealingWarmRestarts':
        return cosineAnnealingWarmRestarts(optimizer, T, eta_min)
    elif scheduler == 'cosineAnnealingLR':
        return cosineAnnealingLR(optimizer, T, eta_min)
    else:
        raise ValueError(f'Invalid scheduler name: {scheduler}')