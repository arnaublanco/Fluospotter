"""Resnet architecture."""
import pdb

import torch
from monai.networks.nets import resnet18, resnet34, resnet50, resnet101, resnet152
from ..io import get_extension


class CustomResNet(nn.Module):
    def __init__(self, model_name, pretrained=None, in_c=1, n_classes=2, patch_size=(64, 64, 16)):
        self.model = self.get_model(model_name=model_name, in_c=in_c, n_classes=n_classes, pretrained=pretrained,
                                    patch_size=patch_size)

    @staticmethod
    def get_model(model_name, in_c=1, n_classes=2, pretrained=None):
        if get_extension(pretrained) == "pt":
            model = torch.load(pretrained)
        elif model_name == "overlapping_puncta":
            model_detection = resnet34(pretrained=True, spatial_dims=2, n_input_channels=in_c, num_classes=2)
            model_localization = resnet34(pretrained=True, spatial_dims=2, n_input_channels=in_c, num_classes=4)
            model = CombinedModel(model_detection, model_localization)
        elif model_name == 'resnet34':
            model = resnet34(pretrained=True, spatial_dims=2, n_input_channels=in_c, num_classes=n_classes)
            if pretrained is not None:
                try:
                    model.load_state_dict(torch.load(pretrained))
                except Exception:
                    raise ValueError(f'Could not load pretrained weights {pretrained} for resnet34.')
        elif model_name == 'resnet50':
            model = resnet50(pretrained=True, spatial_dims=2, n_input_channels=in_c, num_classes=n_classes)
            if pretrained is not None:
                try:
                    model.load_state_dict(torch.load(pretrained))
                except Exception:
                    raise ValueError(f'Could not load pretrained weights {pretrained} for resnet50.')
        else:
            raise ValueError(f'The given ResNet model \'{model_name}\' is not valid.')

        return model


class CombinedModel(torch.nn.Module):
    def __init__(self, counting_model, regression_model):
        super(CombinedModel, self).__init__()
        self.counting_model = counting_model
        self.regression_model = regression_model

    def forward(self, x):
        # Classification using ResNet
        counting_output = self.counting_model(x)

        # Localization using U-Net
        regression_output = self.regression_model(x)

        return counting_output, regression_output