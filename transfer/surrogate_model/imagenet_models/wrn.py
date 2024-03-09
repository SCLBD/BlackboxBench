import torch
import torchvision.models as tvmodels

__all__ = ['wrn101', 'wrn50']

def wrn101(pretrained=False, **kwargs):
    model = tvmodels.wide_resnet101_2()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/wide_resnet101_2-32ee1156.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model


def wrn50(pretrained=False, **kwargs):
    model = tvmodels.wide_resnet50_2()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/wide_resnet50_2-95faca4d.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model