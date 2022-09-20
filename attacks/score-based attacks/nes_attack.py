'''

This file is copied from the following source:
link: https://github.com/ash-aldujaili/blackbox-adv-examples-signhunter

The original license is placed at the end of this file.

@inproceedings{
al-dujaili2020sign,
title={Sign Bits Are All You Need for Black-Box Attacks},
author={Abdullah Al-Dujaili and Una-May O'Reilly},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SygW0TEFwH}
}

basic structure for main:
1. config args, prior setup
2. implement NES Attack algorithm
3. return the output

'''

"""
Implements NES attacks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import Tensor as t

from attacks.score.score_black_box_attack import ScoreBlackBoxAttack
from utils.compute import lp_step


class NESAttack(ScoreBlackBoxAttack):
    """
    NES Attack
    """

    def __init__(self, max_loss_queries, epsilon, p, fd_eta, lr, q, lb, ub, batch_size, name):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lr: learning rate of NES step
        :param q: number of noise samples per NES step
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_extra_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size= batch_size,
                         name = "NES")
        self.q = q
        self.fd_eta = fd_eta
        self.lr = lr

    def _perturb(self, xs_t, loss_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        num_axes = len(_shape[1:])
        gs_t = torch.zeros_like(xs_t)
        for _ in range(self.q):
            # exp_noise = torch.randn_like(xs_t) / (dim ** 0.5)
            exp_noise = torch.randn_like(xs_t)
            fxs_t = xs_t + self.fd_eta * exp_noise
            bxs_t = xs_t - self.fd_eta * exp_noise
            est_deriv = (loss_fct(fxs_t) - loss_fct(bxs_t)) / (4. * self.fd_eta)
            gs_t += t(est_deriv.reshape(-1, *[1] * num_axes)) * exp_noise
        # perform the step
        new_xs = lp_step(xs_t, gs_t, self.lr, self.p)
        return new_xs, 2 * self.q * torch.ones(_shape[0])

    def _config(self):
        return {
            "name": self.name,
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_extra_queries": "inf" if np.isinf(self.max_extra_queries) else self.max_extra_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "lr": self.lr,
            "q": self.q,
            "fd_eta": self.fd_eta,
            "attack_name": self.__class__.__name__
        }
    
'''

Original License
   
MIT License

Copyright (c) 2019 Abdullah Al-Dujaili

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

'''
