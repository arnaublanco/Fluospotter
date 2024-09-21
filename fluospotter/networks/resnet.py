"""Resnet architecture."""
import pdb

import torch
from monai.networks.nets import resnet18, resnet50, resnet101, resnet152
from torchvision.models import resnet34, ResNet34_Weights
from ..io import get_extension


class CustomResNet(torch.nn.Module):
    def __init__(self, model_name, pretrained=None, in_c=1, n_classes=2):
        super(CustomResNet, self).__init__()
        self.model = self.get_model(model_name=model_name, in_c=in_c, n_classes=n_classes, pretrained=pretrained)

    @staticmethod
    def get_model(model_name, in_c=1, n_classes=2, pretrained=None):
        if model_name == "overlapping_puncta":
            if isinstance(pretrained, str):
                if get_extension(pretrained) == ".pt":
                    model = torch.load(pretrained)
            else:
                model_detection = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
                pretrained_weights = model_detection.state_dict()
                conv1_weight = pretrained_weights['conv1.weight']
                averaged_conv1_weight = conv1_weight.mean(dim=1, keepdim=True)
                model_detection.conv1.weight = torch.nn.Parameter(averaged_conv1_weight)
                num_ftrs = model_detection.fc.in_features
                model_detection.fc = torch.nn.Linear(num_ftrs, out_features=2)
                model_localization = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
                pretrained_weights = model_localization.state_dict()
                conv1_weight = pretrained_weights['conv1.weight']
                averaged_conv1_weight = conv1_weight.mean(dim=1, keepdim=True)
                model_localization.conv1.weight = torch.nn.Parameter(averaged_conv1_weight)
                num_ftrs = model_localization.fc.in_features
                model_localization.fc = torch.nn.Linear(num_ftrs, out_features=4)
                try:
                    model_detection.load_state_dict(torch.load(pretrained[0]))
                    model_localization.load_state_dict(torch.load(pretrained[1]))
                except Exception:
                    raise ValueError(f'Could not load pretrained weights {pretrained} for resnet34.')
                model = CombinedModel(model_detection, model_localization)
        elif model_name == 'resnet34':
            model = resnet34(pretrained=False, spatial_dims=2, n_input_channels=in_c, num_classes=n_classes)
            if pretrained is not None:
                try:
                    model.load_state_dict(torch.load(pretrained))
                except Exception:
                    raise ValueError(f'Could not load pretrained weights {pretrained} for resnet34.')
        elif model_name == 'resnet50':
            model = resnet50(pretrained=False, spatial_dims=2, n_input_channels=in_c, num_classes=n_classes)
            if pretrained is not None:
                try:
                    model.load_state_dict(torch.load(pretrained))
                except Exception:
                    raise ValueError(f'Could not load pretrained weights {pretrained} for resnet50.')
        else:
            raise ValueError(f'The given ResNet model \'{model_name}\' is not valid.')

        return model


class CombinedModel(torch.nn.Module):
    def __init__(self, counting_model, localization_model):
        super(CombinedModel, self).__init__()
        self.counting_model = counting_model
        self.localization_model = localization_model

    def forward(self, x):
        # Classification using ResNet
        counting_output = self.counting_model(x)

        # Localization using ResNet
        localization_output = self.localization_model(x)

        return counting_output, localization_output