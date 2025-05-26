import surrogate_model.imagenet_models.adv_resnet50_gelu.advresnet_gbn_gelu as advres
from surrogate_model.imagenet_models.adv_resnet50_gelu.EightBN import EightBN
import torch

__all__ = ['adv_resnet50_gelu']

def adv_resnet50_gelu(pretrained=False):
    model = advres.__dict__['resnet50'](norm_layer=EightBN)
    ckpt_dir = './surrogate_model/NIPS2017/pretrained/advres50_gelu.pth'
    if pretrained:
        ckpt = torch.load(ckpt_dir)
        model.load_state_dict(ckpt['model'])

    return model