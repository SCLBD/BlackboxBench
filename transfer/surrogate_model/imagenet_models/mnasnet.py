import torch
import torchvision.models as tvmodels

__all__ = ['mnasnet']

def mnasnet(pretrained=False, **kwargs):
    model = tvmodels.mnasnet1_0()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/mnasnet1.0_top1_73.512-f206786ef8.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model