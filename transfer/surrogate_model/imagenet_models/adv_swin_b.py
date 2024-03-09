from robustbench.utils import load_model

__all__ = ['adv_swin_b']

# ('Liu2023Comprehensive_Swin-B', {
#             'model': lambda: normalize_model(timm.create_model(
#                 'swin_base_patch4_window7_224', pretrained=False), mu, sigma),
#             'gdrive_id': '1-4mtxQCkThJUVdS3wvQ6NnmMZuySqR3c',
#             'preprocessing': 'BicubicRes256Crop224'
#         })

def adv_swin_b(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name='Liu2023Comprehensive_Swin-B', dataset='imagenet', threat_model='Linf').model