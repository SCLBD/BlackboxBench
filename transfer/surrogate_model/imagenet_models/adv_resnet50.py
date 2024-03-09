from robustbench.utils import load_model

__all__ = ['adv_resnet50']

# ('Salman2020Do_R50', {
#     'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
#     'gdrive_id': '1TmT5oGa1UvVjM3d-XeSj_XmKqBNRUg8r',
#     'preprocessing': 'Res256Crop224'
# }),

def adv_resnet50(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name='Salman2020Do_R50', dataset='imagenet', threat_model='Linf').model