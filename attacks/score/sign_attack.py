from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
    1. config args and prior setup
    2. define functions that insert perturbation and return the estimated gradient
    3. return results

'''

"""
Implements SignHunter
"""
import numpy as np
import torch 

from attacks.score.score_black_box_attack import ScoreBlackBoxAttack
from utils.compute import lp_step, sign, norm


class SignAttack(ScoreBlackBoxAttack):
    """
    SignHunter
    """

    def __init__(self, max_loss_queries, epsilon, p, fd_eta, lb, ub, batch_size, name):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
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
                         name="Sign")


        self.fd_eta = fd_eta
        self.best_est_deriv = None
        self.xo_t = None
        self.sgn_t = None
        self.h = 0
        self.i = 0

    def _perturb(self, xs_t, loss_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        # additional queries at the start
        add_queries = 0
        if self.is_new_batch:
            self.xo_t = xs_t.clone()
            self.h = 0
            self.i = 0
        if self.i == 0 and self.h == 0:
            self.sgn_t = sign(torch.ones(_shape[0], dim))
            fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
            bxs_t = self.xo_t  
            est_deriv = (loss_fct(fxs_t) - loss_fct(bxs_t)) / self.epsilon
            self.best_est_deriv = est_deriv
            add_queries = 3  # because of bxs_t and the 2 evaluations in the i=0, h=0, case.
        chunk_len = np.ceil(dim / (2 ** self.h)).astype(int)
        istart = self.i * chunk_len
        iend = min(dim, (self.i + 1) * chunk_len)
        self.sgn_t[:, istart:iend] *= - 1.
        fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
        bxs_t = self.xo_t
        est_deriv = (loss_fct(fxs_t) - loss_fct(bxs_t)) / self.epsilon
        
        ### sign here
        self.sgn_t[[i for i, val in enumerate(est_deriv < self.best_est_deriv) if val], istart: iend] *= -1.
        

        self.best_est_deriv = (est_deriv >= self.best_est_deriv) * est_deriv + (
                est_deriv < self.best_est_deriv) * self.best_est_deriv
        # perform the step
        new_xs = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
        # update i and h for next iteration
        self.i += 1
        if self.i == 2 ** self.h or iend == dim:
            self.h += 1
            self.i = 0
            # if h is exhausted, set xo_t to be xs_t
            if self.h == np.ceil(np.log2(dim)).astype(int) + 1:
                self.xo_t = xs_t.clone()
                self.h = 0
                print("new change")
                
        if self.p == '2':
            import pdb
            pdb.set_trace()

        return new_xs, torch.ones(_shape[0]) + add_queries

    def get_gs(self):
        """
        return the current estimated of the gradient sign
        :return:
        """
        return self.sgn_t

    def _config(self):
        return {
            "name": self.name, 
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_extra_queries": "inf" if np.isinf(self.max_extra_queries) else self.max_extra_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
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
