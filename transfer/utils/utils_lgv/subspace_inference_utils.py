import itertools
import torch
import os
import copy
from datetime import datetime
import math
import numpy as np
import tqdm
from collections import defaultdict
from time import gmtime, strftime
import sys

import torch.nn.functional as F



def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i:i + n].view(tensor.shape))
        i += n
    return outList


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)

        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()


def extract_parameters(model):
    params = []
    for module in model.modules():
        for name in list(module._parameters.keys()):
            if module._parameters[name] is None:
                continue
            param = module._parameters[name]
            params.append((module, name, param.size()))
            module._parameters.pop(name)
    return params


def set_weights_old(params, w, device):
    offset = 0
    for module, name, shape in params:
        size = np.prod(shape)
        value = w[offset:offset + size]
        setattr(module, name, value.view(shape).to(device))
        offset += size
