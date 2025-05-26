import torch
import torchvision.models as tvmodels
import timm

__all__ = ['vit_b_16_google', 'vit_s_16', 'vit_b_16']

def vit_b_16(pretrained=False, **kwargs):
    model = tvmodels.vit_b_16()
    if pretrained:
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/vit_b_16-c867db91.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model
#
# def vit_b_32(pretrained=False, **kwargs):
#     model = tvmodels.vit_b_32()
#     if pretrained:
#         state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/vit_b_32-d86f8d99.pth', map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
# def vit_l_16(pretrained=False, **kwargs):
#     model = tvmodels.vit_l_16()
#     if pretrained:
#         state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/vit_l_16-852ce7e3.pth', map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model


def vit_b_16_google(pretrained=False, **kwargs):
    '''
    Different from the above ViT-B/16 from torchvision,this version is reimplemented by Google, based on
    https://github.com/lukemelas/PyTorch-Pretrained-ViT
    '''
    from .pytorch_pretrained_vit import ViT
    model = ViT('B_16_imagenet1k', pretrained=pretrained, image_size=224, num_classes=1000)
    return model

def vit_s_16(pretrained=False, **kwargs):
    model = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
    return model
