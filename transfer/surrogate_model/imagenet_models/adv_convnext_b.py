from robustbench.utils import load_model

__all__ = ['adv_convnext_b']

# ('Liu2023Comprehensive_ConvNeXt-B', {
#             'model': lambda: normalize_model(
#                 timm.create_model('convnext_base', pretrained=False), mu, sigma),
#             'gdrive_id': '10-nSm-qUftvfKXHeOAakBQl8rxm-jCbk',
#             'preprocessing': 'BicubicRes256Crop224',
#         })

def adv_convnext_b(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name='Liu2023Comprehensive_ConvNeXt-B', dataset='imagenet', threat_model='Linf').model