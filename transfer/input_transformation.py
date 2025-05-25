import copy

from torch import nn
from utils.registry import Registry
import numpy as np
import torch.nn.functional as F
import torch
from utils.helper import update_and_clip
from surrogate_model.utils import guess_and_load_model, list_path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_input_transformation(input_trans_pipeline):
    """
    Transform an input string into the input transformation.

    The minilanguage is as follows:
        fn1|fn2(arg1, arg2, ...)|...
    which describes the successive transformation 'fn's to the input,
    each function can optionally have one or more args, which are either positional or key:value.

    The output transformation func expects a pipeline of input transformations.

    :param input_trans_pipeline: A string describing the pre-processing pipeline.
    :return: input transformation.
    :raises: ValueError: if input transformation name is unknown.
    """
    trans_pp = []
    if input_trans_pipeline:
        for trans_name in input_trans_pipeline.split("|"):
            try:
                trans_pp.append(Registry.lookup(f"input_transformation.{trans_name}")())
            except SyntaxError as err:
                raise ValueError(f"Syntax error on: {trans_name}") from err
    else:
        # if input_trans_pipeline is empty, apply identity transformation.
        trans_pp.append(Registry.lookup(f"input_transformation.identity")())

    def _trans_fn(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        """The input transformation function that is returned."""

        # Apply all the individual transformation in a sequence.
        for trans in trans_pp:
            adv_img = trans(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter)
        return adv_img

    return _trans_fn


@Registry.register("input_transformation.DI")
def DI(in_size, out_size):
    """
    This function is the core of DI-FGSM, modified based on the following source:
    link:
        https://github.com/cihangxie/DI-2-FGSM
    citation:
        @inproceedings{xie2019improving,
            title={Improving Transferability of Adversarial Examples with Input Diversity},
            author={Xie, Cihang and Zhang, Zhishuai and Zhou, Yuyin and Bai, Song and Wang, Jianyu and Ren, Zhou and Yuille, Alan},
            Booktitle = {Computer Vision and Pattern Recognition},
            year={2019},
            organization={IEEE}
        }
    The original license:
        MIT License

        Copyright (c) 2018 Cihang Xie

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
    def _DI(iter, img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        gg = torch.randint(0, 2, (1,)).item()
        if gg == 0:
            return img
        else:
            rnd = torch.randint(in_size, out_size+1, (1,)).item()
            rescaled = F.interpolate(img, (rnd, rnd), mode='nearest')
            h_rem = out_size - rnd
            w_hem = out_size - rnd
            pad_top = torch.randint(0, h_rem + 1, (1,)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_hem + 1, (1,)).item()
            pad_right = w_hem - pad_left
            padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
            padded = F.interpolate(padded, (in_size, in_size), mode='nearest')
        return padded

    return _DI


@Registry.register("input_transformation.identity")
def iden():
    """
    Identity function, which means doing no transformation on [adv_img].
    """
    def _iden(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        return adv_img

    return _iden


@Registry.register("input_transformation.TI")
def TI(kerlen=5):
    def _TI(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        """
        Even if Translation-Invariant Attack is input transformation-based method, it boosts efficiency by replacing
        calculating gradients of translated images by shifting gradient of untranslated images. So you could check
        'convolved_grad' operation in 'gradient_calculation.py'.
        """
        return adv_img

    return _TI


@Registry.register("input_transformation.admix")
def admix(strength, n_samples):
    """
    This function is the core of Admix, modified based on the following source:
    link:
        https://github.com/JHL-HUST/Admix
    citation:
        @inproceedings{wang2021admix,
          title={Admix: Enhancing the transferability of adversarial attacks},
          author={Wang, Xiaosen and He, Xuanran and Wang, Jingdong and He, Kun},
          booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
          pages={16158--16167},
          year={2021}
        }
    """
    args = Registry._GLOBAL_REGISTRY['args']
    def _admix(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        global addin
        do_admix = True if n_copies_iter == 0 else False
        if do_admix:
            stop = False
            random_indices = np.random.randint(0, adv_img.shape[0], adv_img.shape[0])
            while stop == False:
                if 'NIPS2017' in args.source_model_path[0]:
                    random_indices = np.random.randint(0, adv_img.shape[0], adv_img.shape[0])
                    addin_labels = true_label[random_indices]
                    if torch.sum(addin_labels == true_label).item() == 0:
                        stop = True
                elif 'CIFAR10' in args.source_model_path[0]:
                    label_flag = (true_label[random_indices] == true_label).cpu()
                    random_indices[label_flag] = np.random.randint(0, adv_img.shape[0], sum(label_flag).numpy())
                    if torch.sum(true_label[random_indices] == true_label).item() == 0:
                        stop = True
            addin = adv_img[random_indices].clone().detach()
        return adv_img + strength * addin

    return _admix


@Registry.register("input_transformation.SI")
def SI(n_copies, scale=2):
    """
    This function is the core of SI-FGSM, modified based on the following source:
    link:
        https://github.com/JHL-HUST/SI-NI-FGSM
    citation:
        @inproceedings{lin2019nesterov,
          title={Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks},
          author={Lin, Jiadong and Song, Chuanbiao and He, Kun and Wang, Liwei and Hopcroft, John E},
          booktitle={International Conference on Learning Representations},
          year={2019}
        }
    """
    def _SI(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        return adv_img / (scale ** n_copies_iter)

    return _SI


@Registry.register("input_transformation.look_ahead_NI")
def look_ahead_NI(step_size=1/255):
    """
    This function is the core of NI-FGSM, modified based on the following source:
    link:
        https://github.com/JHL-HUST/SI-NI-FGSM
    citation:
        @inproceedings{lin2019nesterov,
          title={Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks},
          author={Lin, Jiadong and Song, Chuanbiao and He, Kun and Wang, Liwei and Hopcroft, John E},
          booktitle={International Conference on Learning Representations},
          year={2019}
        }
    """
    args = Registry._GLOBAL_REGISTRY['args']
    def _look_ahead_NI(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        return adv_img + args.norm_step * args.decay_factor * grad_accumulate

    return _look_ahead_NI


@Registry.register("input_transformation.look_ahead_PI")
def look_ahead_PI(step_size=1/255):
    """
    This function is the core of PI-FGSM, modified based on the following source:
    link:
        https://github.com/Trustworthy-AI-Group/EMI
    citation:
        @article{wang2021boosting,
          title={Boosting adversarial transferability through enhanced momentum},
          author={Wang, Xiaosen and Lin, Jiadong and Hu, Han and Wang, Jingdong and He, Kun},
          journal={arXiv preprint arXiv:2103.10609},
          year={2021}
        }
    """
    args = Registry._GLOBAL_REGISTRY['args']
    def _look_ahead_PI(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        grad_last_norm = grad_last / (torch.mean(torch.abs(grad_last), (1, 2, 3), keepdim=True) + 1e-12)
        return adv_img + step_size * args.decay_factor * grad_last_norm

    return _look_ahead_PI


@Registry.register("input_transformation.add_reverse_perturbation")
def add_reverse_perturbation(late_start, inner_iter=10):
    """
    This function is the core of RAP, modified based on the following source:
    link:
        https://github.com/SCLBD/Transfer_attack_RAP
    citation:
        @article{qin2022boosting,
          title={Boosting the transferability of adversarial attacks with reverse adversarial perturbation},
          author={Qin, Zeyu and Fan, Yanbo and Liu, Yi and Shen, Li and Zhang, Yong and Wang, Jue and Wu, Baoyuan},
          journal={Advances in Neural Information Processing Systems},
          volume={35},
          pages={29845--29858},
          year={2022}
        }
    """
    # PGD configs
    args = Registry._GLOBAL_REGISTRY['args']
    rvs_args = copy.deepcopy(args)
    rvs_grad_calculator = Registry._GLOBAL_REGISTRY['gradient_calculation.general']()
    rvs_loss_func = Registry._GLOBAL_REGISTRY['loss_function.cross_entropy']() if args.targeted \
                    else Registry._GLOBAL_REGISTRY['loss_function.max_logit']()
    rvs_trans_func = Registry._GLOBAL_REGISTRY['input_transformation.identity']()
    rvs_args.input_transformation = 'identity'
    rvs_args.targeted = not args.targeted
    pgd_iter = inner_iter
    perturbation = 0

    def _add_reverse_perturbation(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        nonlocal perturbation

        if iter >= late_start:
            if n_copies_iter == 0:
                ori_adv_img = adv_img.clone().detach()
                rap_true_labels = true_label.clone().detach()
                rap_target_labels = target_label.clone().detach()
                rap_img = ori_adv_img.clone().detach()
                random_start_epsilon = args.epsilon if args.norm_type == 'inf' else args.epsilon / 100 if args.norm_type == '2' else None
                rap_img = rap_img + torch.empty_like(ori_adv_img).uniform_(-random_start_epsilon,
                                                                           random_start_epsilon)  # random start
                rap_img = torch.clamp(rap_img, min=0, max=1).detach()

                for i in range(pgd_iter):
                    rap_img.requires_grad_(True)
                    rvs_grad = rvs_grad_calculator(rvs_args, 0, 0, rap_img, rap_target_labels, rap_true_labels, 0, 0,
                                                   rvs_trans_func, ensemble_models, rvs_loss_func)
                    rap_img = update_and_clip(rvs_args, rap_img, ori_adv_img, rvs_grad)
                perturbation = (rap_img - ori_adv_img).detach()
        else:
            perturbation = torch.zeros(adv_img.size()).cuda()

        return adv_img + perturbation

    return _add_reverse_perturbation


@Registry.register("input_transformation.siA")
def siA(n_block, n_copies):
    """
    This function is the core of SIA, modified based on the following source:
    link:
        https://github.com/xiaosen-wang/SIT
    citation:
        @inproceedings{wang2023structure,
             title={{Structure Invariant Transformation for better Adversarial Transferability}},
             author={Xiaosen Wang and Zeliang Zhang and Jianping Zhang},
             booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
             year={2023}
        }
    The original license:
        MIT License

        Copyright (c) 2023 xiaosen-wang

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

    def vertical_shift(x):
        _, _, w, _ = x.shape
        step = np.random.randint(low=0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(x):
        _, _, _, h = x.shape
        step = np.random.randint(low=0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(x):
        return x.flip(dims=(2,))

    def horizontal_flip(x):
        return x.flip(dims=(3,))

    def rotate180(x):
        return x.rot90(k=2, dims=(2, 3))

    def scale(x):
        return torch.rand(1)[0] * x

    def resize(x):
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor) + 1
        new_w = int(w * scale_factor) + 1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
        return x

    def dct(x):
        # Apply 2D DCT using FFT
        dctx = torch.fft.fft2(x, norm='ortho')

        # Zero out the high-frequency components (similar to low-pass filtering)
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        dctx[:, :, -low_w:, :] = 0
        dctx[:, :, :, -low_h:] = 0

        # Apply the inverse 2D DCT using FFT
        return torch.fft.ifft2(dctx, norm='ortho').real

    def add_noise(x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255), 0, 1)

    def drop_out(x):
        return F.dropout2d(x, p=0.1, training=True)

    def block_transform(x, ops, num_block):
        _, _, w, h = x.shape
        y_axis = [0] + sorted(np.random.choice(list(range(1, h)), num_block - 1, replace=False).tolist()) + [h]
        x_axis = [0] + sorted(np.random.choice(list(range(1, w)), num_block - 1, replace=False).tolist()) + [w]

        x_copy = x.clone()
        for i in range(len(x_axis) - 1):
            for j in range(len(y_axis) - 1):
                chosen_op = np.random.choice(ops)
                x_copy[:, :, x_axis[i]:x_axis[i + 1], y_axis[j]:y_axis[j + 1]] = chosen_op(
                    x_copy[:, :, x_axis[i]:x_axis[i + 1], y_axis[j]:y_axis[j + 1]])

        return x_copy

    def _siA(iter, adv_img, true_label, target_label, ensemble_models, grad_accumulate, grad_last, n_copies_iter):
        ops = [vertical_shift, horizontal_shift, vertical_flip, horizontal_flip, rotate180, scale, resize, add_noise,
               dct, drop_out]
        x_copies = torch.cat([block_transform(adv_img, ops, n_block) for _ in range(n_copies)])
        return x_copies

    return _siA
