import torch
from utils.registry import Registry
from utils.registry import parse_name
import difflib
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import re


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
    def _general_grad(args, iter, adv_img, true_label, target_label, grad_accumulate, grad_last,
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
        assert args.n_var_sample is not None, "Please assign a value for argument 'n_var_sample' if calculation of " \
                                              "gradient variance is desired."
        for c in range(args.n_var_sample):
            neighbor_bound = 1.5 if args.norm_type == 'inf' else 0.015 if args.norm_type == '2' else None
            adv_img_noise = adv_img + adv_img.new(adv_img.size()).uniform_(-neighbor_bound * args.epsilon, neighbor_bound * args.epsilon)
            adv_img_noise.retain_grad()
            gradient = general_grad()(args, 0, adv_img_noise, true_label, target_label, grad_accumulate, grad_last,
                                      input_trans_func, ensemble_models, loss_func)
            global_grad = global_grad + gradient
        variance = global_grad / args.n_var_sample - grad_cur
        return variance

    return _get_variance


@Registry.register("gradient_calculation.skip_gradient")
def skip_gradient(gamma):
    """
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
    source_models = Registry._GLOBAL_REGISTRY['source_models']
    assert len(args.source_model_path) == 1, "Skip-gradient-method doesn't support ensemble attack."
    a = re.match(r"^.+/pretrained/(\w+)$", args.source_model_path[0])
    arch = a.groups()[0]
    if arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        register_hook_for_resnet(next(source_models)[0], arch=arch, gamma=gamma)
    elif arch in ['densenet121', 'densenet169', 'densenet201', 'densenet']:
        register_hook_for_densenet(next(source_models)[0], arch=arch, gamma=gamma)
    elif arch in ['vit_b_16']:
        register_hook_for_vit(next(source_models)[0], arch=arch, gamma=gamma)
    else:
        raise ValueError('Current code only supports resnet/densenet. '
                         'You can extend this code to other architectures.')

    def _skip_gradient(args, iter, adv_img, true_label, target_label, grad_accumulate, grad_last,
                      input_trans_func, ensemble_models, loss_func):
        gradient = general_grad()(args, iter, adv_img, true_label, target_label, grad_accumulate, grad_last,
                                  input_trans_func, ensemble_models, loss_func)
        return gradient


    return _skip_gradient
