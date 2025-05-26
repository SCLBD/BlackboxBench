import torch
import torchvision.models as tvmodels

__all__ = ['resnext101']

def resnext101(pretrained=False, **kwargs):
    model = tvmodels.resnext101_32x8d()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/resnext101_32x8d-8ba56ff5.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model