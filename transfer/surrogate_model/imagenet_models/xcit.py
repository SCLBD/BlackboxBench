import timm
import torch

__all__ = ['adv_xcit_s', 'xcit_s']


def adv_xcit_s(pretrained=False, **kwargs):
    model = timm.create_model('xcit_small_12_p16_224', num_classes=1000, drop_rate=0.0, drop_connect_rate=None,
                              drop_path_rate=0.05, drop_block_rate=None, global_pool=None, bn_momentum=None,
                              bn_eps=None, scriptable=False,
                              checkpoint_path='./surrogate_model/NIPS2017/pretrained/xcit-s12-ImageNet-eps-4.pth.tar')
    return model

def xcit_s(pretrained=False, **kwargs):
    model = timm.create_model('xcit_small_24_p16_224', pretrained=pretrained)
    return model