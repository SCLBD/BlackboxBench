import os

import torch
from utils.registry import Registry, parse_name
from utils.helper import update_and_clip
import difflib
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import re
from torch.autograd import Variable as V
import copy
from surrogate_model.utils import build_model, guess_and_load_model, guess_arch_from_path, list_path
from input_transformation import build_input_transformation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_grad_calculator(grad_calculator_name):
    """
    Transform an input string into the gradient calculator.
    :param grad_calculator_name: A string describing the method of calculating gradient.
    :return: a gradient calculator.
    :raises: ValueError: if gradient calculator name is unknown.
    """
    if grad_calculator_name:
        try:
            grad_calculator = Registry.lookup(f"gradient_calculation.{grad_calculator_name}")()
        except SyntaxError as err:
            raise ValueError(f"Syntax error on: {grad_calculator}") from err

    return grad_calculator


def linbp_backw_resnet50(img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=1.0):
    """
    This is the backward function for LinBP, modified based on the following source. Build upon I-FGSM framework,
    the complete LinBP algorithm also emphasizes a novel forward function---LinBP(), defined in 'loss_function.py'.
    link:
        https://github.com/qizhangli/linbp-attack
    citation:
        @inproceedings{guo2020backpropagating,
            title={Backpropagating Linearly Improves Transferability of Adversarial Examples.},
            author={Guo, Yiwen and Li, Qizhang and Chen, Hao},
            booktitle={NeurIPS},
            year={2020}
        }
    """
    for i in range(-1, -len(conv_out_ls)-1, -1):
        if i == -1:
            grads = torch.autograd.grad(loss, conv_out_ls[i])
        else:
            grads = torch.autograd.grad((conv_out_ls[i+1][0], conv_input_ls[i+1][1]), conv_out_ls[i], grad_outputs=(grads[0], main_grad_norm))
        normal_grad_2 = torch.autograd.grad(conv_out_ls[i][1], relu_out_ls[i][1], grads[1]*ori_mask_ls[i][2], retain_graph=True)[0]
        normal_grad_1 = torch.autograd.grad(relu_out_ls[i][1], relu_out_ls[i][0], normal_grad_2 * ori_mask_ls[i][1], retain_graph=True)[0]
        normal_grad_0 = torch.autograd.grad(relu_out_ls[i][0], conv_input_ls[i][1], normal_grad_1 * ori_mask_ls[i][0], retain_graph=True)[0]
        del normal_grad_2, normal_grad_1
        main_grad = torch.autograd.grad(conv_out_ls[i][1], conv_input_ls[i][1], grads[1])[0]
        alpha = normal_grad_0.norm(p=2, dim=(1, 2, 3), keepdim=True) / main_grad.norm(p=2, dim=(1, 2, 3), keepdim=True)
        main_grad_norm = xp * alpha * main_grad
    input_grad = torch.autograd.grad((conv_out_ls[0][0], conv_input_ls[0][1]), img, grad_outputs=(grads[0], main_grad_norm))
    return input_grad[0].data


@Registry.register("gradient_calculation.general")
def general_grad():
    """
    This is a basic function for calculating gradient on adversarial examples:
    $\nabla_x \mathcal{L}\left(f^{\prime \prime}\left(\mathcal{T}\left(x_t^*\right) ; \theta^{\prime
    \prime}\right), y ; \mathbf{I}\right)$
    """
    def _general_grad(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                      input_trans_func, ensemble_models, loss_func):
        """
        :param args: arguments
        :param iter: the current step index
        :param adv_img: a batch of adversarial examples to be updated
        :param true_label: true labels of [adv_img]
        :param target_label: target labels of [adv_img]
        :param grad_accumulate: accumulated gradient, useful for momentum-based methods
        :param grad_last: gradient at the last step
        :param ensemble_models: a model list containing all ensembled surrogate models
        :param loss_func: loss function
        :param input_trans_func: input transformation function for pre-processing [adv_img] before feeding
                                 them into [loss_func]
        :return: gradient on adversarial examples [adv_img]
        """

        n_samples = parse_name(difflib.get_close_matches('admix(strength=, n_samples=)', args.input_transformation.split('|'), 1, cutoff=0.1)[0])[2]\
        ['n_samples'] if 'admix' in args.input_transformation else 1
        n_copies = parse_name(difflib.get_close_matches('SI(n_copies=)', args.input_transformation.split('|'), 1, cutoff=0.1)[0])[2]\
        ['n_copies'] if 'SI' in args.input_transformation else 1

        grad_ls = []
        for _ in range(n_samples):
            for n_copies_iter in range(n_copies):
                # apply image transformation
                trans_img = input_trans_func(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter)

                # calculate loss
                loss = loss_func(args, trans_img, true_label, target_label, ensemble_models)

                # get gradient via backpropagation：nonlinear-BP / linear-BP
                if args.backpropagation == 'nonlinear':
                    loss.backward()
                    gradient = adv_img.grad.clone()
                    adv_img.grad.zero_()
                elif args.backpropagation == 'linear':
                    if 'vgg19_bn' in args.source_model_path[0]:
                        loss[0].backward()
                        gradient = adv_img.grad.clone()
                        adv_img.grad.zero_()
                    elif 'resnet50' in args.source_model_path[0]:
                        gradient = linbp_backw_resnet50(trans_img, loss[0], *loss[1])
                else:
                    raise ValueError("Only support `liner` or `nonlinear` backpropagation mode.")

                grad_ls.append(gradient)

        multi_grad = torch.mean(torch.stack(grad_ls), dim=0)

        return multi_grad

    return _general_grad


@Registry.register("gradient_calculation.convolved_grad")
def convolved_grad(kerlen):
    """
    This function is the core of TI-FGSM, modified based on the following source:
    link:
        https://github.com/dongyp13/Translation-Invariant-Attacks
    citation:
        @inproceedings{dong2019evading,
          title={Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks},
          author={Dong, Yinpeng and Pang, Tianyu and Su, Hang and Zhu, Jun},
          booktitle={Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition},
          year={2019}
        }
    The original license:
        MIT License
        Copyright 2017 Yinpeng Dong

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
    """
    def get_kernel(kernel_size):
        # define gaussian kernel for TI
        def gkern(kernlen=15, nsig=3):
            x = np.linspace(-nsig, nsig, kernlen)
            kern1d = st.norm.pdf(x)
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
            return kernel

        kernel = gkern(kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

        return gaussian_kernel

    def _convolved_grad(args, gradient, grad_accumulate, grad_var_last):
        """
        The nature of TI method is convolving the gradient with a Gaussian kernel to calculate the gradient of
        translated images efficiently, so TI operation belongs to the pipeline of 'gradient_calculation'. However, when
        composing VT and TI, we'd like to apply TI after tuning gradient with variance which belongs to the category of
        'update_dir_calculation'. Thus, the input/output of '_convolved_grad' must align with the input/output of
        'update_dir_calculation' pipeline,
        """
        gaussian_kernel = get_kernel(kernel_size=kerlen)
        padding = int((kerlen - 1) / 2)  # same padding
        conv_grad = F.conv2d(gradient, gaussian_kernel, bias=None, stride=1, padding=(padding, padding), groups=3)
        return conv_grad, grad_accumulate
    return _convolved_grad


@Registry.register("get_variance")
def get_variance():
    """
    This function is for calculating the variance in VT, modified based on the following source:
    link:
        https://github.com/JHL-HUST/VT
    citation:
        @inproceedings{wang2021enhancing,
          title={Enhancing the transferability of adversarial attacks through variance tuning},
          author={Wang, Xiaosen and He, Kun},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          pages={1924--1933},
          year={2021}
        }
    """
    def _get_variance(args, adv_img, true_label, target_label,
                      grad_accumulate, grad_last, grad_cur,
                      input_trans_func, ensemble_models, loss_func):
        global_grad = 0
        adv_img.requires_grad_(True)
        assert args.n_var_sample is not None, "Please assign a value for argument 'n_var_sample' if calculation of " \
                                              "gradient variance is desired."
        for c in range(args.n_var_sample):
            neighbor_bound = 1.5 if args.norm_type == 'inf' else 0.015 if args.norm_type == '2' else None
            adv_img_noise = adv_img + adv_img.new(adv_img.size()).uniform_(-neighbor_bound * args.epsilon, neighbor_bound * args.epsilon)
            adv_img_noise.retain_grad()
            gradient = general_grad()(args, 0, 0, adv_img_noise, true_label, target_label, grad_accumulate, grad_last,
                                      input_trans_func, ensemble_models, loss_func)
            global_grad = global_grad + gradient
        variance = global_grad / args.n_var_sample - grad_cur
        return variance

    return _get_variance


@Registry.register("gradient_calculation.skip_gradient")
def skip_gradient(gamma):
    """
    Do not use multiple GPUs if you'd like to run SGM.
    This function is the core of SGM, modified based on the following source:
    link:
        https://github.com/csdongxian/skip-connections-matter
    citation:
        @inproceedings{wu2020skip,
            title={Skip connections matter: On the transferability of adversarial examples generated with resnets},
            author={Wu, Dongxian and Wang, Yisen and Xia, Shu-Tao and Bailey, James and Ma, Xingjun},
            booktitle={ICLR},
            year={2020}
        }
    The original license:
        MIT License

        Copyright (c) 2020 Dongxian Wu

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
    """
    def backward_hook(gamma):
        # implement SGM through grad through ReLU
        def _backward_hook(module, grad_in, grad_out):
            if isinstance(module, torch.nn.ReLU):
                return (gamma * grad_in[0],)

        return _backward_hook


    def backward_hook_mlp(gamma):
        # implement SGM through grad through dropout
        def _backward_hook(module, grad_in, grad_out):
            if isinstance(module, torch.nn.Dropout):
                return (gamma * grad_in[0],)

        return _backward_hook


    def backward_hook_norm(module, grad_in, grad_out):
        # normalize the gradient to avoid gradient explosion or vanish
        std = torch.std(grad_in[0])
        return (grad_in[0] / std,)

    def register_hook_for_resnet(model, arch, gamma):
        # There is only 1 ReLU in Conv module of ResNet-18/34
        # and 2 ReLU in Conv module ResNet-50/101/152
        if arch in ['resnet50', 'resnet101', 'resnet152']:
            gamma = np.power(gamma, 0.5)
        backward_hook_sgm = backward_hook(gamma)

        for name, module in model.named_modules():
            if 'relu' in name and not '0.relu' in name:
                module.register_backward_hook(backward_hook_sgm)

            # e.g., 1.layer1.1, 1.layer4.2, ...
            # if len(name.split('.')) == 3:
            if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
                module.register_backward_hook(backward_hook_norm)

    def register_hook_for_densenet(model, arch, gamma):
        # There are 2 ReLU in Conv module of DenseNet-121/169/201.
        gamma = np.power(gamma, 0.5)
        backward_hook_sgm = backward_hook(gamma)
        for name, module in model.named_modules():
            if 'relu' in name and not 'transition' in name:
                module.register_backward_hook(backward_hook_sgm)

    def register_hook_for_vit(model, arch, gamma):
        backward_hook_sgm = backward_hook_mlp(gamma)
        for name, module in model.named_modules():
            if 'dropout_1' in name and 'layer' in name and 'mlp' in name:
                module.register_backward_hook(backward_hook_sgm)
            elif 'dropout' in name and 'layer' in name and not 'mlp' in name:
                module.register_backward_hook(backward_hook_sgm)

    args = Registry._GLOBAL_REGISTRY['args']
    gammas = gamma.split('|')
    source_models = Registry._GLOBAL_REGISTRY['source_models']
    source_size = Registry._GLOBAL_REGISTRY['source_size']
    # assert len(args.source_model_path) == 1, "Skip-gradient-method doesn't support ensemble attack."
    for model_out_iter_idx in range(source_size // args.n_ensemble):
        models_in_iter = next(source_models)
        for model_in_iter_idx in range(args.n_ensemble):
            if args.n_ensemble * model_out_iter_idx + model_in_iter_idx < source_size:
                arch = guess_arch_from_path(args.source_model_path[args.n_ensemble * model_out_iter_idx + model_in_iter_idx])
                if arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                    register_hook_for_resnet(models_in_iter[model_in_iter_idx], arch=arch,
                                             gamma=float(gammas[args.n_ensemble * model_out_iter_idx + model_in_iter_idx]))
                elif arch in ['densenet121', 'densenet169', 'densenet201', 'densenet']:
                    register_hook_for_densenet(models_in_iter[model_in_iter_idx], arch=arch,
                                               gamma=float(gammas[args.n_ensemble * model_out_iter_idx + model_in_iter_idx]))
                elif arch in ['vit_b_16']:
                    register_hook_for_vit(models_in_iter[model_in_iter_idx], arch=arch,
                                          gamma=float(gammas[args.n_ensemble * model_out_iter_idx + model_in_iter_idx]))
                else:
                    raise ValueError('Current code only supports resnet/densenet. '
                                     'You can extend this code to other architectures.')

    def _skip_gradient(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                      input_trans_func, ensemble_models, loss_func):
        gradient = general_grad()(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                                  input_trans_func, ensemble_models, loss_func)
        return gradient

    return _skip_gradient


@Registry.register("gradient_calculation.cwa_new")
def cwa_new(reverse_step_size, inner_step_size):
    """
    This function is the core of CWA, modified based on the following source:
    link:
        https://github.com/huanranchen/AdversarialAttacks
    citation:
        @article{chen2023rethinking,
          title={Rethinking Model Ensemble in Transfer-based Adversarial Attacks},
          author={Chen, Huanran and Zhang, Yichi and Dong, Yinpeng and Zhu, Jun},
          journal={arXiv preprint arXiv:2303.09105},
          year={2023}
        }
    """

    args = Registry._GLOBAL_REGISTRY['args']
    inner_grad_accumulate = torch.zeros((args.batch_size,3,299,299)).cuda()
    # TI_func = Registry.global_registry()['gradient_calculation.convolved_grad'](kerlen=5)
    # SGM_grad = Registry.lookup('gradient_calculation.skip_gradient')(gamma='0.2|0.5|0.6')

    def _cwa_new(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                input_trans_func, ensemble_models, loss_func):
        n_samples = parse_name(difflib.get_close_matches('admix(strength=, n_samples=)', args.input_transformation.split('|'), 1, cutoff=0.1)[0])[2] \
            ['n_samples'] if 'admix' in args.input_transformation else 1
        n_copies = parse_name(difflib.get_close_matches('SI(n_copies=)', args.input_transformation.split('|'), 1, cutoff=0.1)[0])[2] \
            ['n_copies'] if 'SI' in args.input_transformation else 1

        grad_ls = []
        nonlocal inner_grad_accumulate
        if iter == 0:
            inner_grad_accumulate = torch.zeros(ori_img.size()).cuda() # reset inner_momentum for each batch

        if args.momentum_set_zero is not None and iter % args.momentum_set_zero == 0:
            inner_grad_accumulate = torch.zeros(ori_img.size()).cuda()

        for _ in range(n_samples):
            for n_copies_iter in range(n_copies):

                # first step
                ori_img_iter = adv_img.clone().detach()
                N = ori_img.shape[0]

                adv_img.requires_grad = True

                loss = loss_func(args, adv_img, true_label, target_label, ensemble_models)
                loss.backward()
                ensemble_gradient = adv_img.grad
                # ensemble_gradient, _ = TI_func(args, ensemble_gradient, 0, 0)

                adv_img.requires_grad = False
                adv_img = update_and_clip(args, adv_img, ori_img, -ensemble_gradient, step_size=reverse_step_size)

                # second step
                adv_img.grad = None
                for n_model_iter in range(len(ensemble_models)):

                    adv_img.requires_grad = True

                    # inner_img_grad = Registry.lookup('get_variance')()(args, adv_img, true_label, target_label,
                    #                                              grad_accumulate, grad_last, torch.zeros(ori_img.size()).cuda(),
                    #                                              input_trans_func, [ensemble_models[n_model_iter]], loss_func)

                    trans_img = input_trans_func(iter, adv_img, true_label, target_label, [ensemble_models[n_model_iter]],
                                                 grad_accumulate, grad_last, n_copies_iter)
                    loss = loss_func(args, trans_img, true_label, target_label, [ensemble_models[n_model_iter]])

                    loss.backward()
                    inner_img_grad = adv_img.grad
                    #
                    # inner_img_grad, _ = TI_func(args, inner_img_grad, 0, 0)

                    # inner_img_grad = SGM_grad(args, iter, ori_img,
                    #                     adv_img, true_label, target_label, grad_accumulate, grad_last,
                    #                     input_trans_func, [ensemble_models[n_model_iter]], loss_func)

                    adv_img.requires_grad = False

                    inner_grad_accumulate = args.decay_factor * inner_grad_accumulate + \
                                inner_img_grad / torch.norm(inner_img_grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)

                    # update inner adversarial images
                    if args.norm_type == "inf":
                        adv_img += inner_step_size * inner_grad_accumulate
                        adv_img = torch.where(adv_img > ori_img + args.epsilon, ori_img + args.epsilon, adv_img)
                        adv_img = torch.where(adv_img < ori_img - args.epsilon, ori_img - args.epsilon, adv_img)
                        adv_img = torch.clamp(adv_img, min=0, max=1)
                    elif args.norm_type == "2":
                        # adv_img = update_and_clip(args, adv_img, ori_img, inner_grad_accumulate, step_size=inner_step_size)
                        adv_img += inner_step_size * inner_grad_accumulate
                        perturb = adv_img - ori_img
                        perturb = perturb.renorm(p=2, dim=0, maxnorm=args.epsilon)
                        adv_img = ori_img + perturb
                        adv_img = torch.clamp(adv_img, min=0, max=1)
                outer_grad = adv_img - ori_img_iter
                grad_ls.append(outer_grad)

        multi_grad = torch.mean(torch.stack(grad_ls), dim=0)
        adv_img.requires_grad = True

        return multi_grad

    return _cwa_new


@Registry.register("gradient_calculation.flat_cwa")
def flat_cwa(reverse_step_size, inner_step_size):
    """
        This function is the ablation study FLAT-CWA in Paper 'Seeking Flat Minima over Diverse Surrogates for
        Improved Adversarial Transferability: A Theoretical Framework and Algorithmic Instantiation'
    """
    args = Registry._GLOBAL_REGISTRY['args']
    inner_grad_accumulate = torch.zeros((args.batch_size,3,299,299)).cuda()

    cfg_2_arch = {'ConvNeXt': 'convnext_t_tv',
                  'ResNet': 'resnet50',
                  'ResNet_AT': 'adv_resnet50_gelu',
                  'ViT': 'vit_b_16_google',
                  'Xcit': 'adv_xcit_s',
                  'DenseNet': 'densenet121',
                  'Inception3': 'inception_v3',
                  'VGG': 'vgg19_bn'}

    def get_arch_from_seq(dp_seq_model):
        seq_model = dp_seq_model.module
        while seq_model.__class__ is torch.nn.modules.container.Sequential:
            seq_model = seq_model._modules['1']
        arch = seq_model.__class__.__name__
        return arch

    def _flat_cwa(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                input_trans_func, ensemble_models, loss_func):

        # diverse model loading
        diverse_models = []
        for model in ensemble_models:
            arch = cfg_2_arch[get_arch_from_seq(model)]
            dataset = 'NIPS2017' if 'NIPS2017' in args.source_model_path[0] else 'CIFAR10'
            model_path_all = list_path([f'{dataset}/rap_new/{arch}'], verbose=False)
            model_path_iter = model_path_all[iter % len(model_path_all)]
            model_iter = guess_and_load_model(model_path_iter,
                                              load_as_ghost=args.ghost_attack,
                                              load_as_Styless=args.Styless_attack)
            diverse_models.append(model_iter)

        n_samples = parse_name(difflib.get_close_matches('admix(strength=, n_samples=)', args.input_transformation.split('|'), 1, cutoff=0.1)[0])[2] \
            ['n_samples'] if 'admix' in args.input_transformation else 1
        n_copies = parse_name(difflib.get_close_matches('SI(n_copies=)', args.input_transformation.split('|'), 1, cutoff=0.1)[0])[2] \
            ['n_copies'] if 'SI' in args.input_transformation else 1

        grad_ls = []
        nonlocal inner_grad_accumulate
        if iter == 0:
            inner_grad_accumulate = torch.zeros(ori_img.size()).cuda() # reset inner_momentum for each batch

        if args.momentum_set_zero is not None and iter % args.momentum_set_zero == 0:
            inner_grad_accumulate = torch.zeros(ori_img.size()).cuda()

        for _ in range(n_samples):
            for n_copies_iter in range(n_copies):

                # first step
                ori_img_iter = adv_img.clone().detach()
                N = ori_img.shape[0]

                adv_img.requires_grad = True
                loss = loss_func(args, adv_img, true_label, target_label, diverse_models)
                loss.backward()
                ensemble_gradient = adv_img.grad
                adv_img.requires_grad = False
                adv_img = update_and_clip(args, adv_img, ori_img, -ensemble_gradient, step_size=reverse_step_size)

                # second step
                adv_img.grad = None
                for n_model_iter in range(len(diverse_models)):

                    adv_img.requires_grad = True

                    trans_img = input_trans_func(iter, adv_img, true_label, target_label, [diverse_models[n_model_iter]],
                                                 grad_accumulate, grad_last, n_copies_iter)
                    loss = loss_func(args, trans_img, true_label, target_label, [diverse_models[n_model_iter]])

                    loss.backward()
                    inner_img_grad = adv_img.grad
                    adv_img.requires_grad = False

                    inner_grad_accumulate = args.decay_factor * inner_grad_accumulate + \
                                inner_img_grad / torch.norm(inner_img_grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)

                    # update inner adversarial images
                    if args.norm_type == "inf":
                        adv_img += inner_step_size * inner_grad_accumulate
                        adv_img = torch.where(adv_img > ori_img + args.epsilon, ori_img + args.epsilon, adv_img)
                        adv_img = torch.where(adv_img < ori_img - args.epsilon, ori_img - args.epsilon, adv_img)
                        adv_img = torch.clamp(adv_img, min=0, max=1)
                    elif args.norm_type == "2":
                        # adv_img = update_and_clip(args, adv_img, ori_img, inner_grad_accumulate, step_size=inner_step_size)
                        adv_img += inner_step_size * inner_grad_accumulate
                        perturb = adv_img - ori_img
                        perturb = perturb.renorm(p=2, dim=0, maxnorm=args.epsilon)
                        adv_img = ori_img + perturb
                        adv_img = torch.clamp(adv_img, min=0, max=1)
                outer_grad = adv_img - ori_img_iter
                grad_ls.append(outer_grad)

        multi_grad = torch.mean(torch.stack(grad_ls), dim=0)
        adv_img.requires_grad = True

        return multi_grad

    return _flat_cwa


@Registry.register("gradient_calculation.adaea")
def adaea(threshold, beta):
    """
    This function is the core of AdaEA, modified based on the following source:
    link:
        https://github.com/CHENBIN99/AdaEA
    citation:
        @InProceedings{Chen_2023_ICCV,
            author    = {Chen, Bin and Yin, Jiali and Chen, Shukai and Chen, Bohao and Liu, Ximeng},
            title     = {An Adaptive Model Ensemble Adversarial Attack for Boosting Adversarial Transferability},
            booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
            year      = {2023}
        }
    """
    def max_logit_loss():

        def _max_logit_loss(output, label):
            logits = output.gather(1, label.unsqueeze(1)).squeeze(1)
            loss = -1 * logits.sum()
            return loss

        return _max_logit_loss

    def agm(args, ori_data, cur_adv, grad, true_label, target_label, ensemble_models):
        """
        Adaptive gradient modulation
        :param ori_data: natural images
        :param cur_adv: adv examples in last iteration
        :param grad: gradient in this iteration
        :param label: ground truth
        :return: coefficient of each model
        """
        loss_func = torch.nn.CrossEntropyLoss()

        # generate adversarial example
        adv_exp = [update_and_clip(args, adv_img=cur_adv.detach(), ori_img=ori_data, update_dir=grad[idx])
                   for idx in range(args.n_ensemble)]
        loss_self = [loss_func(ensemble_models[idx](adv_exp[idx]), true_label) for idx in range(args.n_ensemble)] \
                    if not args.targeted else \
                    [-loss_func(ensemble_models[idx](adv_exp[idx]), target_label) for idx in range(args.n_ensemble)]
        w = torch.zeros(size=(args.n_ensemble,)).cuda()

        for j in range(args.n_ensemble):
            for i in range(args.n_ensemble):
                if i == j:
                    continue
                w[j] += loss_func(ensemble_models[i](adv_exp[j]), true_label) / loss_self[i] * beta \
                        if not args.targeted else \
                        loss_func(ensemble_models[i](adv_exp[j]), target_label) / loss_self[i] * beta

        w = torch.softmax(w, dim=0)

        return w

    def drf(args, grads, data_size):
        """
        disparity-reduced filter
        :param grads: gradients of each model
        :param data_size: size of input images
        :return: reduce map
        """
        reduce_map = torch.zeros(size=(args.n_ensemble, args.n_ensemble, data_size[0], data_size[-2], data_size[-1]),
                                 dtype=torch.float).cuda()
        sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        reduce_map_result = torch.zeros(size=(args.n_ensemble, data_size[0], data_size[-2], data_size[-1]),
                                        dtype=torch.float).cuda()
        for i in range(args.n_ensemble):
            for j in range(args.n_ensemble):
                if i >= j:
                    continue
                reduce_map[i][j] = sim_func(F.normalize(grads[i], dim=1), F.normalize(grads[j], dim=1))
            if i < j:
                one_reduce_map = (reduce_map[i, :].sum(dim=0) + reduce_map[:, i].sum(dim=0)) / (args.n_ensemble - 1)
                reduce_map_result[i] = one_reduce_map

        return reduce_map_result.mean(dim=0).view(data_size[0], 1, data_size[-2], data_size[-1])

    def _adaea(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
               input_trans_func, ensemble_models, loss_func):
        grads = []
        outputs = []
        B, C, H, W = ori_img.size()

        if args.loss_function == 'cross_entropy':
            loss_func = torch.nn.CrossEntropyLoss()
        elif args.loss_function == 'max_logit':
            loss_func = max_logit_loss()

        for model in ensemble_models:
            output = model(adv_img)
            loss = loss_func(output, true_label) if not args.targeted else -loss_func(output, target_label)
            grad = torch.autograd.grad(loss, adv_img, retain_graph=True, create_graph=False)[0]
            grads.append(grad)
            outputs.append(output)

        # AGM
        alpha = agm(args, ori_img, adv_img, grads, true_label, target_label, ensemble_models)

        # DRF
        cos_res = drf(args, grads, data_size=(B, C, H, W))
        cos_res[cos_res >= threshold] = 1.
        cos_res[cos_res < threshold] = 0.

        output = torch.stack(outputs, dim=0) * alpha.view(args.n_ensemble, 1, 1)
        output = output.sum(dim=0)
        loss = loss_func(output, true_label) if not args.targeted else -loss_func(output, target_label)
        grad = torch.autograd.grad(loss.sum(dim=0), adv_img)[0]
        grad = grad * cos_res

        return grad

    return _adaea


@Registry.register("gradient_calculation.PGN_grad")
def PGN_grad(zeta, delta, N):
    """
    This function is the core of PGN, modified based on the following source:
    link:
        https://github.com/Trustworthy-AI-Group/PGN
    citation:
        @inproceedings{ge2023boosting,
         title={{Boosting Adversarial Transferability by Achieving Flat Local Maxima}},
         author={Zhijin Ge and Hongying Liu and Xiaosen Wang and Fanhua Shang and Yuanyuan Liu},
         booktitle={Proceedings of the Advances in Neural Information Processing Systems},
         year={2023}}
    The original license:
        MIT License

        Copyright (c) 2023 Trustworthy-AI-Group

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

    :param zeta: Upper bound of neighborhood perturbation.
    :param delta: Balancing coefficient for gradient combination.
    :param N: Number of sampled examples for gradient estimation.
    """

    def _PGN_grad(args, iter,ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                  input_trans_func, ensemble_models, loss_func):
        """
        :param args: arguments
        :param iter: the current step index
        :param adv_img: a batch of adversarial examples to be updated
        :param true_label: true labels of [adv_img]
        :param target_label: target labels of [adv_img]
        :param grad_accumulate: accumulated gradient (not used in this function)
        :param grad_last: gradient at the last step (not used in this function)
        :param ensemble_models: a model list containing all ensembled surrogate models
        :param loss_func: loss function (cross-entropy, etc.)
        :param input_trans_func: input transformation function for pre-processing [adv_img] before feeding
                                 them into [loss_func]
        :return: gradient on adversarial examples [adv_img]
        """

        # Parameters specific to PGN
        eps = args.epsilon
        alpha = args.norm_step
        norm_type = args.norm_type  # l-inf or l2

        grad_ls = []

        for _ in range(N):
            # Apply image transformation (if any)
            trans_img = input_trans_func(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate,
                                         grad_last, _)

            if norm_type == "inf":
                # Perturb the input image within the epsilon-zeta neighborhood (l-inf norm)
                x_near = trans_img + torch.rand_like(trans_img).uniform_(-eps * zeta, eps * zeta)
            elif norm_type == "2":
                # Perturb the input image within the epsilon-zeta neighborhood (l2 norm)
                noise = torch.randn_like(trans_img)
                noise = noise / torch.norm(noise.view(noise.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                x_near = trans_img + eps * zeta * noise
            else:
                raise ValueError(f"Unsupported norm_type: {norm_type}")

            x_near = V(x_near, requires_grad=True)

            # Forward pass through the ensemble models
            out = 0
            for model in ensemble_models:
                out += model(x_near)
            out /= len(ensemble_models)

            # Calculate loss using cross-entropy
            loss = loss_func(args, x_near, true_label, target_label, ensemble_models)

            # First gradient g1
            loss.backward(retain_graph=True)
            g1 = x_near.grad.clone()
            x_near.grad.zero_()

            if norm_type == "inf":
                # Update x_star using g1 (l-inf norm)
                x_star = x_near.detach() + alpha * (-g1) / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            elif norm_type == "2":
                # Update x_star using g1 (l2 norm)
                g1_norm = g1 / torch.norm(g1.view(g1.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                x_star = x_near.detach() + alpha * (-g1_norm)

            x_star = V(x_star, requires_grad=True)

            # Forward pass through the ensemble models for x_star
            out = 0
            for model in ensemble_models:
                out += model(x_star)
            out /= len(ensemble_models)

            # Calculate loss for x_star
            loss_star = loss_func(args, x_star, true_label, target_label, ensemble_models)

            # Second gradient g2
            loss_star.backward(retain_graph=True)
            g2 = x_star.grad.clone()
            x_star.grad.zero_()

            if norm_type == "inf":
                avg_grad = (1 - delta) * g1 + delta * g2
            elif norm_type == "2":
                g2_norm = g2 / torch.norm(g2.view(g2.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1)
                avg_grad = (1 - delta) * g1_norm + delta * g2_norm

            grad_ls.append(avg_grad)

        # Combine accumulated gradients from multiple samples
        multi_grad = torch.mean(torch.stack(grad_ls), dim=0)

        return multi_grad

    return _PGN_grad


@Registry.register("gradient_calculation.ensemble_grad")
def ensemble_grad(n_ensemble):

    def _ensemble_grad(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                       input_trans_func, ensemble_models,loss_func):

        model_gradients = []
        for i in range(n_ensemble):
            model = guess_and_load_model(args.source_model_path[0], load_as_Styless=False)
            model_grad = general_grad()(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate,
                                         grad_last, input_trans_func, [model], loss_func)
            model_gradients.append(model_grad)
        averaged_gradient = torch.mean(torch.stack(model_gradients), dim=0)
        return averaged_gradient

    return _ensemble_grad


@Registry.register("gradient_calculation.flat_rap")
def flat_rap():
    """
        This function is the ablation study FLAT-RAP in Paper 'Seeking Flat Minima over Diverse Surrogates for
        Improved Adversarial Transferability: A Theoretical Framework and Algorithmic Instantiation'
    """
    cfg_2_arch = {'ConvNeXt': 'convnext_t_tv',
                  'ResNet': 'resnet50',
                  'ResNet_AT': 'adv_resnet50_gelu',
                  'ViT': 'vit_b_16_google',
                  'Xcit': 'adv_xcit_s',
                  'DenseNet': 'densenet121',
                  'Inception3': 'inception_v3',
                  'VGG': 'vgg19_bn'}

    def get_arch_from_seq(dp_seq_model):
        seq_model = dp_seq_model.module
        while seq_model.__class__ is torch.nn.modules.container.Sequential:
            seq_model = seq_model._modules['1']
        arch = seq_model.__class__.__name__
        return arch

    def _flat_rap(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                          input_trans_func, ensemble_models, loss_func):
        diverse_models = []
        for model in ensemble_models:
            arch = cfg_2_arch[get_arch_from_seq(model)]
            dataset = 'NIPS2017' if 'NIPS2017' in args.source_model_path[0] else 'CIFAR10'
            model_path_all = list_path([f'{dataset}/rap_new/{arch}'], verbose=False)
            model_path_iter = model_path_all[iter % len(model_path_all)]
            model_iter = guess_and_load_model(model_path_iter,
                                              load_as_ghost=args.ghost_attack,
                                              load_as_Styless=args.Styless_attack)
            diverse_models.append(model_iter)

        grad = general_grad()(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                              input_trans_func, diverse_models, loss_func)
        return grad

    return _flat_rap


@Registry.register("gradient_calculation.rap_new")
def rap_new(model_num, late_start, reverse_step, reverse_step_size):
    '''
    This function is the core of DRAP.
    citation:
        @article{zheng2025seeking,
          title={Seeking Flat Minima over Diverse Surrogates for Improved Adversarial Transferability: A Theoretical Framework and Algorithmic Instantiation},
          author={Zheng, Meixi and Wu, Kehan and Fan, Yanbo and Huang, Rui and Wu, Baoyuan},
          journal={arXiv preprint arXiv:2504.16474},
          year={2025}
        }
    '''

    args = Registry._GLOBAL_REGISTRY['args']
    assert args.n_iter == model_num * len(args.source_model_path)
    reverse_step_size /= 255
    late_start *= len(args.source_model_path)
    cfg_2_arch = {'ConvNeXt': 'convnext_t_tv',
                  'ResNet': 'resnet50',
                  'ResNet_AT': 'adv_resnet50_gelu',
                  'ViT': 'vit_b_16_google',
                  'Xcit': 'adv_xcit_s',
                  'DenseNet': 'densenet121',
                  'Inception3': 'inception_v3',
                  'VGG': 'vgg19_bn'}

    def get_arch_from_seq(dp_seq_model):
        seq_model = dp_seq_model[0].module
        while seq_model.__class__ is torch.nn.modules.container.Sequential:
            seq_model = seq_model._modules['1']
        arch = seq_model.__class__.__name__
        return arch

    def _rap_new(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                 input_trans_func, ensemble_models, loss_func):
        assert len(ensemble_models) == 1
        arch = cfg_2_arch[get_arch_from_seq(ensemble_models)]
        dataset = 'NIPS2017' if 'NIPS2017' in args.source_model_path[0] else 'CIFAR10'
        model_path_all = list_path([f'{dataset}/rap_new/{arch}'], verbose=False)
        model_path_iter = model_path_all[iter//len(args.source_model_path)]
        model_iter = guess_and_load_model(model_path_iter,
                                          load_as_ghost=args.ghost_attack,
                                          load_as_Styless=args.Styless_attack)

        n_samples = parse_name(difflib.get_close_matches('admix(strength=, n_samples=)', args.input_transformation.split('|'), 1,cutoff=0.1)[0])[2] \
            ['n_samples'] if 'admix' in args.input_transformation else 1
        n_copies = parse_name(difflib.get_close_matches('SI(n_copies=)', args.input_transformation.split('|'), 1, cutoff=0.1)[0])[2] \
            ['n_copies'] if 'SI' in args.input_transformation else 1
        args_notransform = copy.deepcopy(args)
        args_notransform.input_transformation = '' # do not apply input transformation at reverse step
        idfentity_trans = build_input_transformation(args_notransform.input_transformation)
        # TI_func = Registry.global_registry()['gradient_calculation.convolved_grad'](kerlen=5)
        rvs_loss_func = Registry._GLOBAL_REGISTRY['loss_function.max_logit']() if args.targeted \
            else Registry._GLOBAL_REGISTRY['loss_function.cross_entropy']()

        grad_ls = []
        for _ in range(n_samples):
            for n_copies_iter in range(n_copies):
                reversed_img = adv_img

                # reverse step
                if iter >= late_start:
                    for _ in range(reverse_step):
                        reversed_img.requires_grad = True
                        gradient = general_grad()(args_notransform, iter, ori_img, reversed_img, true_label, target_label, grad_accumulate, grad_last,
                                          idfentity_trans, [model_iter], rvs_loss_func)
                        # gradient, _ = TI_func(args, gradient, 0, 0)
                        reversed_img.requires_grad = False
                        # if args.targeted:
                        #     reversed_img = reversed_img + reverse_step_size * gradient.sign()
                        # else:
                        reversed_img = reversed_img - reverse_step_size * gradient.sign()

                # forword step
                reversed_img.requires_grad = True
                trans_img = input_trans_func(iter, reversed_img, true_label, target_label, [model_iter],
                                             grad_accumulate, grad_last, n_copies_iter)
                loss = loss_func(args, trans_img, true_label, target_label, [model_iter])
                loss.backward()
                gradient = reversed_img.grad.clone()
                # gradient, _ = TI_func(args, gradient, 0, 0)
                reversed_img.grad.zero_()
                reversed_img.requires_grad = False
                grad_ls.append(gradient)

        multi_grad = torch.mean(torch.stack(grad_ls), dim=0)

        return multi_grad


    return _rap_new


@Registry.register("gradient_calculation.rap_new_nodiverse")
def rap_new_nodiverse(late_start, reverse_step, reverse_step_size):
    '''
    This function is an ablation study in Paper 'Seeking Flat Minima over Diverse Surrogates for
    Improved Adversarial Transferability: A Theoretical Framework and Algorithmic Instantiation'
    '''

    args = Registry._GLOBAL_REGISTRY['args']
    reverse_step_size /= 255
    late_start *= len(args.source_model_path)

    def _rap_new_nodiverse(args, iter, ori_img, adv_img, true_label, target_label, grad_accumulate, grad_last,
                 input_trans_func, ensemble_models, loss_func):
        assert len(ensemble_models) == 1
        model_iter = ensemble_models[0]

        n_samples = parse_name(difflib.get_close_matches('admix(strength=, n_samples=)', args.input_transformation.split('|'), 1,cutoff=0.1)[0])[2] \
            ['n_samples'] if 'admix' in args.input_transformation else 1
        n_copies = parse_name(difflib.get_close_matches('SI(n_copies=)', args.input_transformation.split('|'), 1, cutoff=0.1)[0])[2] \
            ['n_copies'] if 'SI' in args.input_transformation else 1
        args_notransform = copy.deepcopy(args)
        args_notransform.input_transformation = '' # do not apply input transformation at reverse step
        idfentity_trans = build_input_transformation(args_notransform.input_transformation)
        # TI_func = Registry.global_registry()['gradient_calculation.convolved_grad'](kerlen=5)
        rvs_loss_func = Registry._GLOBAL_REGISTRY['loss_function.max_logit']() if args.targeted \
            else Registry._GLOBAL_REGISTRY['loss_function.cross_entropy']()

        grad_ls = []
        for _ in range(n_samples):
            for n_copies_iter in range(n_copies):
                reversed_img = adv_img

                # reverse step
                if iter >= late_start:
                    for _ in range(reverse_step):
                        reversed_img.requires_grad = True
                        gradient = general_grad()(args_notransform, iter, ori_img, reversed_img, true_label, target_label, grad_accumulate, grad_last,
                                          idfentity_trans, [model_iter], rvs_loss_func)
                        # gradient, _ = TI_func(args, gradient, 0, 0)
                        reversed_img.requires_grad = False
                        # if args.targeted:
                        #     reversed_img = reversed_img + reverse_step_size * gradient.sign()
                        # else:
                        reversed_img = reversed_img - reverse_step_size * gradient.sign()

                # forword step
                reversed_img.requires_grad = True
                trans_img = input_trans_func(iter, reversed_img, true_label, target_label, [model_iter],
                                             grad_accumulate, grad_last, n_copies_iter)
                loss = loss_func(args, trans_img, true_label, target_label, [model_iter])
                loss.backward()
                gradient = reversed_img.grad.clone()
                # gradient, _ = TI_func(args, gradient, 0, 0)
                reversed_img.grad.zero_()
                reversed_img.requires_grad = False
                grad_ls.append(gradient)

        multi_grad = torch.mean(torch.stack(grad_ls), dim=0)

        return multi_grad


    return _rap_new_nodiverse
