from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''

This file is copied from the following source:
link: https://github.com/ash-aldujaili/blackbox-adv-examples-signhunter/blob/master/src/attacks/blackbox/black_box_attack.py

@inproceedings{
al-dujaili2020sign,
title={Sign Bits Are All You Need for Black-Box Attacks},
author={Abdullah Al-Dujaili and Una-May O'Reilly},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SygW0TEFwH}
}

The original license is placed at the end of this file.

basic structure for main:
    1. config args and prior setup
    2. define funtion that returns a summary of the attack results
    3. set the decision-based attacking
    4. return the logs
    
'''

"""
Implements the base class for decision-based black-box attacks
"""
import numpy as np
import torch
from torch import Tensor as t

import sys

class DecisionBlackBoxAttack(object):
    def __init__(self, max_queries=np.inf, epsilon=0.5, p='inf', lb=0., ub=1., batch_size=1):
        """
        :param max_queries: max number of calls to model per data point
        :param epsilon: perturbation limit according to lp-ball
        :param p: norm for the lp-ball constraint
        :param lb: minimum value data point can take in any coordinate
        :param ub: maximum value data point can take in any coordinate
        """
        assert p in ['inf', '2'], "L-{} is not supported".format(p)

        self.p = p
        self.max_queries = max_queries
        self.total_queries = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_distance = 0
        self.sigma = 0
        self.EOT = 1
        self.lb = lb
        self.ub = ub
        self.epsilon = epsilon / ub
        self.batch_size = batch_size
        self.list_loss_queries = torch.zeros(1, self.batch_size)

    def result(self):
        """
        returns a summary of the attack results (to be tabulated)
        :return:
        """
        list_loss_queries = self.list_loss_queries[1:].view(-1)
        mask = list_loss_queries > 0
        list_loss_queries = list_loss_queries[mask]
        self.total_queries = int(self.total_queries)
        self.total_successes = int(self.total_successes)
        self.total_failures = int(self.total_failures)
        return {
            "total_queries": self.total_queries,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "average_num_queries": "NaN" if self.total_successes == 0 else self.total_queries / self.total_successes,
            "failure_rate": "NaN" if self.total_successes + self.total_failures == 0 else self.total_failures / (self.total_successes + self.total_failures),
            "median_num_loss_queries": "NaN" if self.total_successes == 0 else torch.median(list_loss_queries).item(), 
            "config": self._config()
        }

    def _config(self):
        """
        return the attack's parameter configurations as a dict
        :return:
        """
        raise NotImplementedError

    def distance(self, x_adv, x = None):
        if x is None:
            diff = x_adv.reshape(x_adv.size(0), -1)
        else:
            diff = (x_adv - x).reshape(x.size(0), -1)
        if self.p == '2':
            out = torch.sqrt(torch.sum(diff * diff)).item()
        elif self.p == 'inf':
            out = torch.sum(torch.max(torch.abs(diff), 1)[0]).item()
        return out
    
    def is_adversarial(self, x, y):
        '''
        check whether the adversarial constrain holds for x
        '''
        if self.targeted:
            return self.predict_label(x) == y
        else:
            return self.predict_label(x) != y

    def predict_label(self, xs):
        if type(xs) is torch.Tensor:
            x_eval = xs.permute(0,3,1,2)
        else:
            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2))
        x_eval = torch.clamp(x_eval, 0, 1)
        x_eval = x_eval + self.sigma * torch.randn_like(x_eval)
        if self.ub == 255:
            out = self.model(x_eval)[1]
        else:
            out = self.model(x_eval)
        l = out.argmax(dim=1)
        return l.detach()

    def _perturb(self, xs_t, ys):
        raise NotImplementedError

    def run(self, xs, ys_t, model, targeted, dset):
        
        self.model = model
        self.targeted = targeted
        self.train_dataset = dset

        self.logs = {
            'iteration': [0],
            'query_count': [0]
        }

        xs = xs / self.ub
        xs_t = t(xs)

        # initialize
        if self.targeted:
            check = self.is_adversarial(xs_t, ys_t)
            if torch.any(check):
                print('Some original images already belong to the target class!')
                return self.logs
        else:
            check = self.is_adversarial(xs_t, ys_t)
            if torch.any(check):
                print('Some original images do not belong to the original class!')
                return self.logs

        adv, q = self._perturb(xs_t, ys_t)

        success = self.distance(adv,xs_t) < self.epsilon
        self.total_queries += np.sum(q * success)
        self.total_successes += np.sum(success)
        self.total_failures += ys_t.shape[0] - success
        self.list_loss_queries = torch.cat([self.list_loss_queries, torch.zeros(1, self.batch_size)], dim=0)
        if type(q) is np.ndarray:
            self.list_loss_queries[-1] = t(q * success)
        else:
            self.list_loss_queries[-1] = q * success
        # self.total_distance += self.distance(adv,xs_t)

        return self.logs
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
