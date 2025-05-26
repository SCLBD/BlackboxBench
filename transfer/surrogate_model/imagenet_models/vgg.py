import torch
import torchvision.models as tvmodels

__all__ = ['vgg11_bn', 'vgg19', 'vgg19_bn', 'vgg16_bn']


def vgg11_bn(pretrained=False, **kwargs):
    model = tvmodels.vgg11_bn()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/vgg11_bn-6002323d.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model


def vgg19(pretrained=False, **kwargs):
    model = tvmodels.vgg19()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/vgg19-dcbb9e9d.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    model = tvmodels.vgg19_bn()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/vgg19_bn-c79401a0.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model

def vgg16_bn(pretrained=False, **kwargs):
    model = tvmodels.vgg16_bn(pretrained=pretrained)
    return model
