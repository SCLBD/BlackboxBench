import timm
import torch

__all__ = ['swin_b', 'swin_t', 'swin_l']

# ('Liu2023Comprehensive_Swin-B', {
#             'model': lambda: normalize_model(timm.create_model(
#                 'swin_base_patch4_window7_224', pretrained=False), mu, sigma),
#             'gdrive_id': '1-4mtxQCkThJUVdS3wvQ6NnmMZuySqR3c',
#             'preprocessing': 'BicubicRes256Crop224'
#         })

def swin_b(pretrained=False, **kwargs):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
    return model

def swin_t(pretrained=False, **kwargs):
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
    return model

def swin_l(pretrained=False, **kwargs):
    model = timm.create_model('swin_large_patch4_window7_224', pretrained=pretrained)
    return model