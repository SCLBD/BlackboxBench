import timm
import torch

__all__ = ['convnext_b', 'convnext_l', 'convnext_t']

# ('Liu2023Comprehensive_ConvNeXt-B', {
#             'model': lambda: normalize_model(
#                 timm.create_model('convnext_base', pretrained=False), mu, sigma),
#             'gdrive_id': '10-nSm-qUftvfKXHeOAakBQl8rxm-jCbk',
#             'preprocessing': 'BicubicRes256Crop224',
#         })


def convnext_b(pretrained=False, **kwargs):
    model = timm.create_model('convnext_base', pretrained=pretrained)
    return model


def convnext_l(pretrained=False, **kwargs):
    model = timm.create_model('convnext_large', pretrained=pretrained)
    return model


def convnext_t(pretrained=False, **kwargs):
    model = timm.create_model('convnext_tiny', pretrained=pretrained)
    return model