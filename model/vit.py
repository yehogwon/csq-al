import torch
import torchvision.models as models

from model.utils import EmbeddingWrapper

def _get_vit_model(model_name: str) -> models.VisionTransformer:
    if model_name == 'vits16':
        return torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    elif model_name == 'vits8':
        return torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    elif model_name == 'vitb16':
        return torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    elif model_name == 'vitb8':
        return torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    else:
        raise ValueError(f'Invalid model name: {model_name}')

def _wrap_vit_model(model: models.VisionTransformer, n_classes: int, linear_probe: bool=False) -> EmbeddingWrapper[models.VisionTransformer]:
    model.emb_dim = 384
    return EmbeddingWrapper(
        model, 
        n_classes, linear_probe
    )

def vits16(n_classes: int=10, pretrained: bool=False, linear_probe: bool=False) -> EmbeddingWrapper[models.VisionTransformer]:
    assert pretrained, 'Only pretrained models are supported'

    _model = _get_vit_model('vits16')
    return _wrap_vit_model(_model, n_classes, linear_probe)

def vits8(n_classes: int=10, pretrained: bool=False, linear_probe: bool=False) -> EmbeddingWrapper[models.VisionTransformer]:
    assert pretrained, 'Only pretrained models are supported'

    _model = _get_vit_model('vits8')
    return _wrap_vit_model(_model, n_classes, linear_probe)

def vitb16(n_classes: int=10, pretrained: bool=False, linear_probe: bool=False) -> EmbeddingWrapper[models.VisionTransformer]:
    assert pretrained, 'Only pretrained models are supported'

    _model = _get_vit_model('vitb16')
    return _wrap_vit_model(_model, n_classes, linear_probe)

def vitb8(n_classes: int=10, pretrained: bool=False, linear_probe: bool=False) -> EmbeddingWrapper[models.VisionTransformer]:
    assert pretrained, 'Only pretrained models are supported'

    _model = _get_vit_model('vitb8')
    return _wrap_vit_model(_model, n_classes, linear_probe)
