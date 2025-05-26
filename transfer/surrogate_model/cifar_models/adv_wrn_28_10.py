from robustbench.utils import load_model

__all__ = ['adv_wrn_28_10']

# ('Carmon2019Unlabeled', {
#             'model':
#             lambda: WideResNet(depth=28, widen_factor=10, sub_block1=True),
#             'gdrive_id':
#             '15tUx-gkZMYx7BfEOw1GY5OKC-jECIsPQ',
#         })

def adv_wrn_28_10(pretrained=True, **kwargs):
    assert pretrained
    return load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')

