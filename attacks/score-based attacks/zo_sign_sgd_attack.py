'''
@inproceedings{
al-dujaili2020sign,
title={Sign Bits Are All You Need for Black-Box Attacks},
author={Abdullah Al-Dujaili and Una-May O'Reilly},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SygW0TEFwH}
}

rewrite from   https://github.com/ash-aldujaili/blackbox-adv-examples-signhunter
'''

"""
Implements ZO-SIGN-SGD attacks from
"SignSGD via Zeroth-Order Oracle"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch 
from torch import Tensor as t
import pdb


from attacks.score.score_black_box_attack import ScoreBlackBoxAttack
from utils.compute import lp_step


class ZOSignSGDAttack(ScoreBlackBoxAttack):
    """
    ZOSignSGD Attack
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
                         batch_size=batch_size,
                         name = "zosignsgd")
        self.q = q
        self.fd_eta = fd_eta
        self.lr = lr

    def _perturb(self, xs_t, loss_fct):
        #pdb.set_trace()
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        num_axes = len(_shape[1:])
        gs_t = torch.zeros_like(xs_t)
        for _ in range(self.q):
            # exp_noise = torch.randn_like(xs_t) / (dim ** 0.5)
            exp_noise = torch.randn_like(xs_t)
            fxs_t = xs_t + self.fd_eta * exp_noise
            bxs_t = xs_t
            est_deriv = (loss_fct(fxs_t) - loss_fct(bxs_t)) / self.fd_eta
            gs_t += est_deriv.reshape(-1, *[1] * num_axes) * exp_noise
        # perform the sign step regardless of the lp-ball constraint
        # this is the main difference in the method.
        new_xs = lp_step(xs_t, gs_t, self.lr, 'inf')
        # the number of queries required for forward difference is q (forward sample) + 1 at xs_t
        return new_xs, (self.q + 1) * torch.ones(_shape[0])

    def _config(self):
        return {
            'name': self.name,
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
