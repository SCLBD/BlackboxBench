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
2. implement Bandit Attack algorithm
3. return the output

'''

"""
Implements Bandit Attack from Ilyas et al. 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from attacks.score.score_black_box_attack import ScoreBlackBoxAttack
from utils.compute import lp_step, step, eg_step, upsample_maker, l2_step


class BanditAttack(ScoreBlackBoxAttack):
    """
    Bandit Attack
    """

    def __init__(self,
                 max_loss_queries,
                 epsilon, p,
                 fd_eta, lr,
                 prior_exploration, prior_size, data_size, prior_lr,
                 lb, ub, batch_size, name):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lr: learning rate of NES step
        :param prior_exploration: exploration noise
        :param prior_size: prior height/width (this is applicable only to images), you can disable it by setting it to
            None (it is assumed to prior_size = prior_height == prior_width)
        :param data_size: data height/width (applicable to images of the from `c x h x w`, you can ignore it
            by setting it to none, it is assumed that data_size = height = width
        :param prior_lr: learning rate in the prior space
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
                         name = "Bandit")
        # other algorithmic parameters
        self.fd_eta = fd_eta
        # learning rate
        self.lr = lr
        # data size
        self.data_size = data_size

        # prior setup:
        # 1. step function
        if self.p == '2':
            self.prior_step = step
        elif self.p == 'inf':
            self.prior_step = eg_step
        else:
            raise Exception("Invalid p for l-p constraint")
        # 2. prior placeholder
        self.prior = None
        # prior size
        self.prior_size = prior_size
        # prior exploration
        self.prior_exploration = prior_exploration
        # 3. prior upsampler
        self.prior_upsample_fct = None if self.prior_size is None else upsample_maker(data_size, data_size)
        self.prior_lr = prior_lr

    def _perturb(self, xs_t, loss_fct):
        """
        The core of the bandit algorithm
        since this is compute intensive, it is implemented with torch support to push ops into gpu (if available)
        however, the input / output are numpys
        :param xs: numpy
        :return new_xs: returns a torch tensor
        """
        if xs_t.dim() != 2 and self.prior_size is not None:  # for cifar10 and imagenet data needs to be transpose into c x  h x w for upsample method
            xs_t = xs_t.transpose(1, 3)
        _shape = list(xs_t.shape)
        eff_shape = list(xs_t.shape)
        # since the upsampling assumes xs_t is batch_size x c x h x w. This is not the case for mnist,
        # which is batch_size x dim, let's take care of that below
        if len(_shape) == 2:
            eff_shape = [_shape[0], 1, self.data_size, self.data_size]
        if self.prior_size is None:
            prior_shape = eff_shape
        else:
            prior_shape = eff_shape[:-2] + [self.prior_size] * 2
        # reset the prior if xs  is a new batch
        if self.is_new_batch:
            self.prior = torch.zeros(prior_shape)
        # create noise for exploration, estimate the gradient, and take a PGD step
        # exp_noise = torch.randn(prior_shape) / (np.prod(prior_shape[1:]) ** 0.5)  # according to the paper
        exp_noise = torch.randn(prior_shape)
        # Query deltas for finite difference estimator
        if self.prior_size is None:
            q1 = step(self.prior, exp_noise, self.prior_exploration)
            q2 = step(self.prior, exp_noise, - self.prior_exploration)
        else:
            q1 = self.prior_upsample_fct(step(self.prior, exp_noise, self.prior_exploration))
            q2 = self.prior_upsample_fct(step(self.prior, exp_noise, - self.prior_exploration))
        # Loss points for finite difference estimator
        if xs_t.dim() != 2 and self.prior_size is not None:
            l1 = loss_fct(l2_step(xs_t, q1.view(_shape), self.fd_eta).transpose(1, 3))
            l2 = loss_fct(l2_step(xs_t, q2.view(_shape), self.fd_eta).transpose(1, 3))
        else:
            l1 = loss_fct(l2_step(xs_t, q1.view(_shape), self.fd_eta))
            l2 = loss_fct(l2_step(xs_t, q2.view(_shape), self.fd_eta))
        # finite differences estimate of directional derivative
        est_deriv = (l1 - l2) / (self.fd_eta * self.prior_exploration)
        # 2-query gradient estimate
        # Note: Ilyas' implementation multiply the below by self.prior_exploration (different from pseudocode)
        # This should not affect the result as the `self.prior_lr` can be adjusted accordingly
        est_grad = torch.Tensor(est_deriv.reshape(-1, *[1] * len(prior_shape[1:]))) * exp_noise
        # update prior with the estimated gradient:
        self.prior = self.prior_step(self.prior, est_grad, self.prior_lr)
        # gradient step in the data space
        if self.prior_size is None:
            gs = self.prior.clone()
        else:
            gs = self.prior_upsample_fct(self.prior)
        if xs_t.dim() != 2 and self.prior_size is not None:
            gs = gs.transpose(1, 3)
            xs_t = xs_t.transpose(1, 3)
            _shape = list(xs_t.shape)
        # perform the step
        new_xs = lp_step(xs_t, gs.view(_shape), self.lr, self.p)
        return new_xs, 2 * torch.ones(_shape[0])

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
            "prior_lr": self.prior_lr,
            "prior_exploration": self.prior_exploration,
            "prior_size": self.prior_size,
            "data_size": self.data_size,
            "fd_eta": self.fd_eta,
            "attack_name": self.__class__.__name__
        }

'''

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
