import torch
from torchvision.models import (
    EfficientNet,
    EfficientNet_B3_Weights
)
from torchvision.models.efficientnet import _efficientnet_conf
from torchvision.models._utils import _ovewrite_named_param
from model.utils import EmbeddingWrapper

def EfficientNetB3(n_classes: int=1000, pretrained: bool=False, linear_probe: bool=False) -> EmbeddingWrapper[EfficientNet]:
    inverted_residual_setting, last_channel = _efficientnet_conf('efficientnet_b3', width_mult=1.2, depth_mult=1.4)

    model = EfficientNet(inverted_residual_setting, 0.3, last_channel=last_channel, num_classes=n_classes)
    if pretrained:
        weights = EfficientNet_B3_Weights.verify(EfficientNet_B3_Weights.IMAGENET1K_V1)
        pretrained_state_dict = weights.get_state_dict(progress=True)
        new_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('classifier.')}
        model.load_state_dict(new_state_dict, strict=False)

    return EmbeddingWrapper(model, n_classes, linear_probe)
