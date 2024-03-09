from robustbench.utils import load_model

__all__ = ['adv_wrn50']

# ('Salman2020Do_50_2', {
#             'model': lambda: normalize_model(pt_models.wide_resnet50_2(), mu, sigma),
#             'gdrive_id': '1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB',
#             'preprocessing': 'Res256Crop224'
#         })

def adv_wrn50(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model='Linf').model
