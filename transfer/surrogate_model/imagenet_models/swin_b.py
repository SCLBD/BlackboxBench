import timm
import torch

__all__ = ['swin_b', 'swin_t', 'swin_l', 'swin_s']

def swin_s(pretrained=False, **kwargs):
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained)
    return model

def swin_b(pretrained=False, **kwargs):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
    return model

def swin_t(pretrained=False, **kwargs):
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
    return model

def swin_l(pretrained=False, **kwargs):
    model = timm.create_model('swin_large_patch4_window7_224', pretrained=pretrained)
    return model