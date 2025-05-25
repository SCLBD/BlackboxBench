from utils.registry import Registry
import numpy as np
from torch import nn
import torch
from torchvision import models as tmodels
from surrogate_model.utils import guess_arch_from_path, list_path, guess_and_load_model, save_checkpoints
import itertools
from tqdm import tqdm
import torchvision
import os
import copy
from collections import OrderedDict
from utils.utils_bayesian_attack import finetune_imagenet, finetune_cifar10
from utils.utils_lgv.train_lgv import train_lgv_imagenet, train_lgv_cifar10
from torchvision.transforms import functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model2vector(model):
    """
    Transform a pytorch model into its weight Tensor
    :param model: pytorch model
    :return: tensor of size (n_weights,)
    """
    w = flatten([param.detach().cpu() for param in model.parameters()])
    return w


def vector2model(w, model_cfg, update_bn=True, train_loader=None, **kwargs):
    """
    Build a pytorch model from given weight vector
    :param w: tensor of size (1, n_weights)
    :param model_cfg: model class
    :param update_bn: Update or not the BN statistics
    :param train_loader: Data loader to update BN stats
    :param kwargs: args passed to model_cfg
    :return: pytorch model
    """
    if update_bn and not train_loader:
        raise ValueError('train_loader must be provided with update_bn')
    w = w.detach().clone()
    new_model = model_cfg(**kwargs).cuda()
    offset = 0
    for param in new_model.parameters():
        param.data.copy_(w[offset:offset + param.numel()].view(param.size()).to('cuda'))
        offset += param.numel()
    if update_bn:
        bn_update(train_loader, new_model, verbose=False, subset=0.1)
    new_model.eval()
    return new_model


def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)


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
    model = nn.DataParallel(model)
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
    model = model.module
    model.apply(lambda module: _set_momenta(module, momenta))


def build_model_refinement(model_refinement_pipeline):

    """
    Transform an input string into the model refinement function.

    The minilanguage is as follows:
        fn1|fn2(arg1, arg2, ...)|...
    which describes the successive refinement 'fn's to the model,
    each function can optionally have one or more args, which are either positional or key:value.

    The output refinement function expects a pipeline of model refinement functions.

    :param model_refinement_pipeline: A string describing the model refinement pipeline.
    :return: model refinement function.
    :raises: ValueError: if model refinement name is unknown.
    """

    rf_pp = []
    for rf_name in model_refinement_pipeline.split('|'):
        try:
            rf_pp.append(Registry.lookup(f"model_refinement.{rf_name}")())
        except SyntaxError as err:
            raise ValueError(f"Syntax error on: {rf_name}") from err

    def _rf_fn(args, rfmodel_dir):
        for rf in rf_pp:
            rfmodel_dir = rf(args, rfmodel_dir)
        return rfmodel_dir

    return _rf_fn


@Registry.register("model_refinement.sample_from_isotropic")
def sample_from_isotropic(std, n_models, update_bn=True):
    """
    This function is the core of RD, modified based on the following source:
    link:
        https://github.com/Framartin/lgv-geometric-transferability
    citation:
        @inproceedings{gubri2022lgv,
          title={Lgv: Boosting adversarial example transferability from large geometric vicinity},
          author={Gubri, Martin and Cordy, Maxime and Papadakis, Mike and Traon, Yves Le and Sen, Koushik},
          booktitle={European Conference on Computer Vision},
          pages={603--618},
          year={2022},
          organization={Springer}
        }
    Original code file license is at the end of the script 'utils/utils_lgv/train_lgv.py'.
    """
    def _sample_from_isotropic(args, rfmodel_dir):
        """
        Sample from a model posterior which is an isotropic Gaussian N(mean_model, std*I), where std is a positive
        constant, and mean_model is from rfmodel_list.
        :param args: arguments
        :param rfmodel_dir: path of to-be-refined model
        :return: path of refined model
        """
        paths = list_path(rfmodel_dir)
        rfmodel_list = [guess_and_load_model(path, norm_layer=False, parallel=False, require_grad=True,
                                             load_as_ghost=False) for path in paths]  # load to-be-refined model without add normalization layer
        assert len(rfmodel_list) == 1, "The current version only support refining one source model, multi-model refinement will be complemented in the future."

        from surrogate_model import imagenet_models, cifar_models
        # prepare dataloader for updating running statistic of models added random direction
        if 'NIPS2017' in rfmodel_dir[0]:
            traindir = os.path.join(args.full_imagenet_dir, 'train')
            if 'inception_v3' in rfmodel_dir[0]:
                transform_train = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(342),
                                    torchvision.transforms.CenterCrop(299),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            else:
                transform_train = torchvision.transforms.Compose([
                                    torchvision.transforms.RandomResizedCrop(224),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_dataset = torchvision.datasets.ImageFolder(traindir, transform_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=10, pin_memory=True)
            model_cfg = imagenet_models.__dict__[guess_arch_from_path(args.source_model_path[0])]
            model_kwg = {}
        elif 'CIFAR10' in rfmodel_dir[0]:
            cifar10_dir = 'data/dataset/CIFAR10'
            transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = torchvision.datasets.CIFAR10(root=cifar10_dir, train=True, download=False, transform=transform_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
            model_cfg = cifar_models.__dict__[guess_arch_from_path(args.source_model_path[0])]
            model_kwg = {'num_classes': 10}

        w = model2vector(rfmodel_list[0])
        all_rfmodels = []
        for i in tqdm(range(n_models), desc="Export refined models"):
            w_random = w.detach().clone() + torch.randn(w.shape) * std
            model_noisy = vector2model(w_random, model_cfg, update_bn=update_bn, train_loader=train_loader, **model_kwg)
            all_rfmodels.append(model_noisy)

        rfmodel_dir = save_checkpoints(args.rfmodel_dir, all_rfmodels, name='model_noisy', sample=True)
        return rfmodel_dir

    return _sample_from_isotropic


@Registry.register("model_refinement.stochastic_weight_collecting")
def stochastic_weight_collecting(collect,
                                 mini_batch=512, epochs=10, lr=0.05, wd=1e-4, momentum=0.9):
    """
    This function is the core of LGV, modified based on the following source:
    link:
        https://github.com/Framartin/lgv-geometric-transferability
    citation:
        @inproceedings{gubri2022lgv,
          title={Lgv: Boosting adversarial example transferability from large geometric vicinity},
          author={Gubri, Martin and Cordy, Maxime and Papadakis, Mike and Traon, Yves Le and Sen, Koushik},
          booktitle={European Conference on Computer Vision},
          pages={603--618},
          year={2022},
          organization={Springer}
        }
    Original code file license is at the end of the script 'utils/utils_lgv/train_lgv.py'.
    """
    def _stochastic_weight_collecting(args, rfmodel_dir):
        """
        Collects weights in a single run along the SGD trajectory (with a high constant learning rate).
        """
        assert len(rfmodel_dir) == 1, "The current version only support refining one source model, " \
                                       "multi-model refinement will be complemented in the future."
        arch = guess_arch_from_path(rfmodel_dir[0])
        dataset = 'NIPS2017' if 'NIPS2017' in args.source_model_path[0] else 'CIFAR10'
        if not collect:
            return [f'{dataset}/LGV/{arch}/iter']
        else:
            if 'NIPS2017' in args.source_model_path[0]:
                return train_lgv_imagenet(dir=args.rfmodel_dir, data_path=args.full_imagenet_dir, batch_size=mini_batch,
                                          pretrained_ckpt=rfmodel_dir[0], epochs=epochs, eval_freq=10, eval_start=False,
                                          swa=True, swa_start=0, swa_lr=lr, wd=wd, momentum=momentum, swa_freq=4, seed=1)

            elif 'CIFAR10' in args.source_model_path[0]:
                return train_lgv_cifar10(dir=args.rfmodel_dir, data_path='data/dataset/CIFAR10', batch_size=mini_batch,
                                          pretrained_ckpt=rfmodel_dir[0], epochs=epochs, eval_freq=10, eval_start=False,
                                          swa=True, swa_start=0, swa_lr=lr, wd=wd, momentum=momentum, swa_freq=4, seed=1)
    return _stochastic_weight_collecting


@Registry.register("model_refinement.stochastic_weight_averaging")
def stochastic_weight_averaging(collect,
                                mini_batch=512, epochs=10, lr=0.05, wd=1e-4, momentum=0.9, lam=1, gamma=0.1):
    """
    This function is the core of SWA, modified based on the following source:
    link:
        https://github.com/qizhangli/MoreBayesian-attack
    citation:
        @article{li2023making,
          title={Making Substitute Models More Bayesian Can Enhance Transferability of Adversarial Examples},
          author={Li, Qizhang and Guo, Yiwen and Zuo, Wangmeng and Chen, Hao},
          booktitle={ICLR},
          year={2023}
        }
    """
    def _stochastic_weight_averaging(args, rfmodel_dir):
        """
        Averaging weights collected in a single run along the SGD trajectory (with a high constant learning rate).
        """
        assert len(rfmodel_dir) == 1, "The current version only support refining one source model, " \
                                       "multi-model refinement will be complemented in the future."
        arch = guess_arch_from_path(rfmodel_dir[0])
        dataset = 'NIPS2017' if 'NIPS2017' in args.source_model_path[0] else 'CIFAR10'
        if not collect:
            swa_path = list_path([f'{dataset}/SWA/{arch}/{arch}_morebayesian_attack.pt'])
            swa_weight = guess_and_load_model(swa_path[0], norm_layer=False, parallel=False, require_grad=True,
                                              load_as_ghost=False, dict_name='mean_state_dict')
        else:
            if 'NIPS2017' in args.source_model_path[0]:
                swa_weight, _ = finetune_imagenet(pretrain_ckpt=rfmodel_dir[0], batch_size=mini_batch,
                                                             lr=lr, wd=wd, momentum=momentum, epochs=epochs,
                                                             lam=lam, gamma=gamma, swa_start=0, swa_n=300, seed=0,
                                                             data_path=args.full_imagenet_dir,
                                                             save_dir=f'./surrogate_model/NIPS2017/SWA/{arch}/recollect_iter')
            elif 'CIFAR10' in args.source_model_path[0]:
                swa_weight, _ = finetune_cifar10(pretrain_ckpt=rfmodel_dir[0], batch_size=mini_batch,
                                                            lr=lr, wd=wd, momentum=momentum, epochs=epochs,
                                                            lam=lam, gamma=gamma, swa_start=0, swa_n=300, seed=0,
                                                            data_path='data/dataset/CIFAR10',
                                                            save_dir=f'./surrogate_model/CIFAR10/SWA/{arch}/recollect_iter')

        rfmodel_dir = save_checkpoints(args.rfmodel_dir, [swa_weight], name='swa')
        return rfmodel_dir

    return _stochastic_weight_averaging


@Registry.register("model_refinement.sample_from_swag")
def sample_from_swag(collect, beta, scale, n_models,
                     mini_batch=512, epochs=10, lr=0.05, wd=1e-4, momentum=0.9, lam=1, gamma=0.1):
    """
    This function is the core of Bayesian Attack, modified based on the following source:
    link:
        https://github.com/qizhangli/MoreBayesian-attack
    citation:
        @article{li2023making,
          title={Making Substitute Models More Bayesian Can Enhance Transferability of Adversarial Examples},
          author={Li, Qizhang and Guo, Yiwen and Zuo, Wangmeng and Chen, Hao},
          booktitle={ICLR},
          year={2023}
        }
    """
    def model_sampler(mean_model, sqmean_model):
        sample = copy.deepcopy(mean_model)
        noise_dict = OrderedDict()
        for (name, param_mean), param_sqmean, param_cur in zip(mean_model.named_parameters(), sqmean_model.parameters(),
                                                               sample.parameters()):
            var = torch.clamp(param_sqmean.data - param_mean.data ** 2, 1e-30)
            var = var + beta
            noise_dict[name] = var.sqrt() * torch.randn_like(param_mean, requires_grad=False)
        for (name, param_cur), (_, noise) in zip(sample.named_parameters(), noise_dict.items()):
            param_cur.data.add_(noise, alpha=scale)
        return sample


    def _sample_from_swag(args, rfmodel_dir):
        """
        Sample from a model posterior which is a stochastic weight averaging Gaussian.
        """
        assert len(rfmodel_dir) == 1, "The current version only support refining one source model, " \
                                       "multi-model refinement will be complemented in the future."
        arch = guess_arch_from_path(rfmodel_dir[0])
        dataset = 'NIPS2017' if 'NIPS2017' in args.source_model_path[0] else 'CIFAR10'
        if not collect:
            collect_paths = list_path([f'{dataset}/Bayesian_Attack/{arch}/{arch}_morebayesian_attack.pt'])
            mean_model = guess_and_load_model(collect_paths[0], norm_layer=False, parallel=False, require_grad=True,
                                              load_as_ghost=False, dict_name='mean_state_dict')
            sqmean_model = guess_and_load_model(collect_paths[0], norm_layer=False, parallel=False, require_grad=True,
                                                load_as_ghost=False, dict_name='sqmean_state_dict')
        else:
            if 'NIPS2017' in args.source_model_path[0]:
                mean_model, sqmean_model = finetune_imagenet(pretrain_ckpt=rfmodel_dir[0], batch_size=mini_batch,
                                                             lr=lr, wd=wd, momentum=momentum, epochs=epochs,
                                                             lam=lam, gamma=gamma, swa_start=0, swa_n=300, seed=0,
                                                             data_path=args.full_imagenet_dir,
                                                             save_dir=f'./surrogate_model/NIPS2017/Bayesian_Attack/{arch}/recollect_iter')
            elif 'CIFAR10' in args.source_model_path[0]:
                mean_model, sqmean_model = finetune_cifar10(pretrain_ckpt=rfmodel_dir[0], batch_size=mini_batch,
                                                            lr=lr, wd=wd, momentum=momentum, epochs=epochs,
                                                            lam=lam, gamma=gamma, swa_start=0, swa_n=300, seed=0,
                                                            data_path='data/dataset/CIFAR10',
                                                            save_dir=f'./surrogate_model/CIFAR10/Bayesian_Attack/{arch}/recollect_iter')

        for i in range(n_models):
            rfmodel = [model_sampler(mean_model, sqmean_model)]
            rfmodel_dir = save_checkpoints(args.rfmodel_dir, rfmodel, name=f'swag_sample_{i}', sample=True, exist_ok=True)
            del rfmodel

        # all_rfmodels = [model_sampler(mean_model, sqmean_model) for _ in range(n_models)]
        # rfmodel_dir = save_checkpoints(args.rfmodel_dir, all_rfmodels, name='swag_sample', sample=True)
        return rfmodel_dir

    return _sample_from_swag
