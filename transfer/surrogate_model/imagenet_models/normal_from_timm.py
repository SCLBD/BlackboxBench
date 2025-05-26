import timm

__all__ = ['pnasnet5_l', 'deit_s', 'poolformer_s', 'pvt_b', 'tnt_s', 'cait_s']

def pnasnet5_l(pretrained=False, **kwargs):
    model = timm.create_model('pnasnet5large', pretrained=pretrained)
    return model

def deit_s(pretrained=False, **kwargs):
    model = timm.create_model('deit_small_patch16_224', pretrained=pretrained)
    return model

def poolformer_s(pretrained=False, **kwargs):
    model = timm.create_model('poolformer_s12', pretrained=pretrained)
    return model

def pvt_b(pretrained=False, **kwargs):
    model = timm.create_model('pvt_v2_b2', pretrained=pretrained)
    return model

def tnt_s(pretrained=False, **kwargs):
    model = timm.create_model('tnt_s_patch16_224', pretrained=pretrained)
    return model

def cait_s(pretrained=False, **kwargs):
    model = timm.create_model('cait_s24_224', pretrained=pretrained)
    return model
