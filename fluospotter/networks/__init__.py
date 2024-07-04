"""Networks folder.

Contains functions returning the base architectures of used models.
"""

from .unet import CustomUNet


__all__ = [
    "unet",
]