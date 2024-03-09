import torch
import torchvision.models as tvmodels

__all__ = ['vit_b_16', 'vit_b_32', 'vit_l_16']

def vit_b_16(pretrained=False, **kwargs):
    model = tvmodels.vit_b_16()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/vit_b_16-c867db91.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model

def vit_b_32(pretrained=False, **kwargs):
    model = tvmodels.vit_b_32()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/vit_b_32-d86f8d99.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model

def vit_l_16(pretrained=False, **kwargs):
    model = tvmodels.vit_l_16()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/vit_l_16-852ce7e3.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model