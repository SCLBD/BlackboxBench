import torch
import torchvision.models as tvmodels

__all__ = ['mobilenet_v2', 'mobilenet_v3']

def mobilenet_v2(pretrained=False, **kwargs):
    model = tvmodels.mobilenet_v2()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/mobilenet_v2-b0353104.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model

def mobilenet_v3(pretrained=False, **kwargs):
    model = tvmodels.mobilenet_v3_small()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/mobilenet_v3_small-047dcff4.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model