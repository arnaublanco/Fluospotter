"""UNet architecture."""

import torch
from monai.networks.nets import UNet, DynUNet


class CustomUNet:
    def __init__(self, model_name, pretrained=None, in_c=1, n_classes=2, patch_size=None):
        self.model = self.get_model(model_name=model_name, in_c=in_c, n_classes=n_classes, pretrained=pretrained,
                                    patch_size=patch_size)

    @staticmethod
    def get_kernels_strides(patch_size=(64, 64, 16), spacings=(1.5625, 1.5625, 5.0)):
        input_size = patch_size
        strides, kernels = [], []
        while True:
            spacing_ratio = [sp / min(spacings) for sp in spacings]
            stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, patch_size)]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            for idx, (i, j) in enumerate(zip(patch_size, stride)):
                if i % j != 0:
                    raise ValueError(
                        f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                    )
            patch_size = [i / j for i, j in zip(patch_size, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)

        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        return kernels, strides

    @staticmethod
    def get_dynunet(patch_size=(64, 64, 16), spacings=(1.5625, 1.5625, 5.0), in_channels=1, n_classes=2,
                    deep_supr_num=0):
        kernels, strides = CustomUNet.get_kernels_strides(patch_size, spacings)
        deep_supervision = False if deep_supr_num == 0 else True

        net = DynUNet(spatial_dims=3, in_channels=in_channels, out_channels=n_classes, kernel_size=kernels,
                      strides=strides,
                      upsample_kernel_size=strides[1:], norm_name="instance", deep_supervision=deep_supervision,
                      deep_supr_num=deep_supr_num)

        return net

    @staticmethod
    def get_model(model_name, in_c=1, n_classes=2, pretrained=None, patch_size=(64, 64, 16)):
        if model_name == 'small_unet_3d':
            model = UNet(spatial_dims=3, in_channels=in_c, out_channels=n_classes, channels=(16, 32,), strides=(2,),
                         num_res_units=1)
            if pretrained is not None:
                try:
                    model.load_state_dict(pretrained)
                except Exception as e:
                    raise ValueError(f'Could not load pretrained weights {pretrained} for small_unet_3d.') from e
        elif model_name == 'dynunet':
            # if patch_size = (256,256,48) -> spacings (1., 1., 5.)
            # if patch_size = (48,256,256) -> spacings (5., 1., 1.)
            model = CustomUNet.get_dynunet(patch_size=patch_size, spacings=(5., 1., 1.), in_channels=in_c, n_classes=n_classes)
            if pretrained is not None:
                try:
                    model.load_state_dict(pretrained)
                except Exception as e:
                    raise ValueError(f'Could not load pretrained weights {pretrained} for dynunet.') from e
        else:
            raise ValueError(f'The given nuclei segmentation model \'{model_name}\' is not valid.')

        setattr(model, 'n_classes', n_classes)
        setattr(model, 'patch_size', patch_size)

        return model