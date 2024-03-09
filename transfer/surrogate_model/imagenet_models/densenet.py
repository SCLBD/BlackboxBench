import torch
import torchvision.models as tvmodels

__all__ = ['densenet121']

def densenet121(pretrained=False, **kwargs):
    model = tvmodels.densenet121()
    if pretrained:
        import re
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load('./surrogate_model/NIPS2017/pretrained/densenet121-a639ec97.pth', map_location='cpu')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model