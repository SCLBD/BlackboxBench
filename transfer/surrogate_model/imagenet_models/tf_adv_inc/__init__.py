from .tf_adv_inception_v3 import IncV3AdvKitModel
from .tf_ens3_adv_inc_v3 import IncV3Ens3AdvKitModel
from .tf_ens4_adv_inc_v3 import IncV3Ens4AdvKitModel
from .tf_ens_adv_inc_res_v2 import IncResV2EnsKitModel

"""
please download checkpoints from 
https://github.com/ylhz/tf_to_pytorch_model
"""

__all__ = ['tf2torch_adv_inception_v3', 'tf2torch_ens3_adv_inc_v3', 'tf2torch_ens4_adv_inc_v3', 'tf2torch_ens_adv_inc_res_v2']

def tf2torch_adv_inception_v3(pretrained, path='./surrogate_model/NIPS2017/pretrained/tf2torch_adv_inception_v3.npy'):
    assert pretrained
    model = IncV3AdvKitModel(path)
    return model


def tf2torch_ens3_adv_inc_v3(pretrained, path='./surrogate_model/NIPS2017/pretrained/tf2torch_ens3_adv_inc_v3.npy'):
    assert pretrained
    model = IncV3Ens3AdvKitModel(path)
    return model


def tf2torch_ens4_adv_inc_v3(pretrained, path='./surrogate_model/NIPS2017/pretrained/tf2torch_ens4_adv_inc_v3.npy'):
    assert pretrained
    model = IncV3Ens4AdvKitModel(path)
    return model


def tf2torch_ens_adv_inc_res_v2(pretrained, path='./surrogate_model/NIPS2017/pretrained/tf2torch_ens_adv_inc_res_v2.npy'):
    assert pretrained
    model = IncResV2EnsKitModel(path)
    return model