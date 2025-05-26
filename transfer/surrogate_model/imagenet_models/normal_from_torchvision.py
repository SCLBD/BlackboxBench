import torch
import torchvision.models as tvmodels

__all__ = ['alexnet', 'squeezenet1_1', 'googlenet', 'shufflenet_v2_x1_0', 'efficientnet_b7', 'regnet_y_16gf']


def alexnet(pretrained=False, **kwargs):
    model = tvmodels.alexnet(pretrained=pretrained)
    return model

def squeezenet1_1(pretrained=False, **kwargs):
    model = tvmodels.squeezenet1_1(pretrained=pretrained)
    return model

def googlenet(pretrained=False, **kwargs):
    model = tvmodels.googlenet(pretrained=pretrained)
    return model

def shufflenet_v2_x1_0(pretrained=False, **kwargs):
    model = tvmodels.shufflenet_v2_x1_0(pretrained=pretrained)
    return model

def efficientnet_b7(pretrained=False, **kwargs):
    model = tvmodels.efficientnet_b7(pretrained=pretrained)
    return model

def regnet_y_16gf(pretrained=False, **kwargs):
    model = tvmodels.regnet_y_16gf(weights=tvmodels.RegNet_Y_16GF_Weights.IMAGENET1K_V2)
    return model
