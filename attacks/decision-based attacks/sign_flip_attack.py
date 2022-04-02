'''

This file is modified based on the following source:
link: https://github.com/wubaoyuan/Sign-Flip-Attack

The original license is placed at the end of this file

@inproceedings{Chen2020boosting,
    title={Boosting Decision-based Black-box Adversarial Attacks with Random Sign Flip},
    author={Chen, Weilun and Zhang, Zhaoxiang and Hu, Xiaolin and Wu, Baoyuan},
    Booktitle = {Proceedings of the European Conference on Computer Vision},
    year={2020}
}

The update include:
    1. model setting
    2. args and config
    3. functions modification

basic structure for main:
    1. config args and prior setup
    2. define functions that implement sign-flip attack and linf binary search
    
'''


"""
Implements Sign Flip
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import Tensor as t
import torch.nn.functional as F

from attacks.decision.decision_black_box_attack import DecisionBlackBoxAttack

class SignFlipAttack(DecisionBlackBoxAttack):
    """
    SignFlip
    """

    def __init__(self, epsilon, p, resize_factor, max_queries, lb, ub, batch_size):
        super().__init__(max_queries = max_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size = batch_size)
        self.resize_factor = resize_factor


    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "attack_name": self.__class__.__name__
        }

    def _perturb(self, x, y):
        '''
        Sign Flip Attack: linf decision-based adversarial attack
        '''
        b, c, h, w = x.size()
        # Q: query number for each image
        Q = torch.zeros(b)
        # q_num: current queries
        q_num = 0
        
        # initialize
        if self.targeted:
            iters = 0
            while True:
                x_a, _ = self.train_dataset.get_eval_data(iters, iters + 1)
                yi_pred = self.predict_label(x_a)
                if yi_pred == y:
                    break
                iters += 1
                if iters > 10000:
                    print('Initialization Failed!')
                    return x, iters
            x_a = t(x_a/self.ub, dtype=torch.float)
        else:
            x_a = torch.rand_like(x)
            iters = 0
            check = self.is_adversarial(x_a, y)
            while check.sum() < y.size(0):
                x_a[check < 1] = torch.rand_like(x_a[check < 1])
                Q[check < 1] += 1
                check = self.is_adversarial(x_a, y)
                iters += 1
                if iters > 10000:
                    print('Initialization Failed!')
                    return x, iters

        # linf binary search
        x_a = self.binary_infinity(x_a, x, y, 10)
        delta = x_a - x
        del x_a

        assert self.resize_factor >= 1.
        h_dr, w_dr = int(h // self.resize_factor), int(w // self.resize_factor)

        # 10 queries for binary search
        q_num, Q = q_num + 10, Q + 10

        # indices for unsuccessful images
        unsuccessful_indices = torch.ones(b) > 0

        # hyper-parameter initialization
        alpha = torch.ones(b) * 0.004
        prob = torch.ones_like(delta) * 0.999
        prob = self.resize(prob, h_dr, w_dr)

        # additional counters for hyper-parameter adjustment
        reset = 0
        proj_success_rate = torch.zeros(b)
        flip_success_rate = torch.zeros(b)

        while q_num < self.max_queries:
            reset += 1
            b_cur = unsuccessful_indices.sum()

            # the project step
            eta = torch.randn([b_cur, c, h_dr, w_dr]).sign() * alpha[unsuccessful_indices][:, None, None, None]
            eta = self.resize(eta, h, w)
            l, _ = delta[unsuccessful_indices].abs().reshape(b_cur, -1).max(1)
            delta_p = self.project_infinity(delta[unsuccessful_indices] + eta, torch.zeros_like(eta),
                                    l - alpha[unsuccessful_indices])
            check = self.is_adversarial((x[unsuccessful_indices] + delta_p).clamp(0, 1), y[unsuccessful_indices])
            delta[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] = delta_p[
                check.nonzero().squeeze(1)]
            proj_success_rate[unsuccessful_indices] += check.float()

            # the random sign flip step
            s = torch.bernoulli(prob[unsuccessful_indices]) * 2 - 1
            delta_s = delta[unsuccessful_indices] * self.resize(s, h, w).sign()
            check = self.is_adversarial((x[unsuccessful_indices] + delta_s).clamp(0, 1), y[unsuccessful_indices])
            prob[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] -= s[check.nonzero().squeeze(
                1)] * 1e-4
            prob.clamp_(0.99, 0.9999)
            flip_success_rate[unsuccessful_indices] += check.float()
            delta[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] = delta_s[
                check.nonzero().squeeze(1)]

            # hyper-parameter adjustment
            if reset % 10 == 0:
                proj_success_rate /= reset
                flip_success_rate /= reset
                alpha[proj_success_rate > 0.7] *= 1.5
                alpha[proj_success_rate < 0.3] /= 1.5
                prob[flip_success_rate > 0.7] -= 0.001
                prob[flip_success_rate < 0.3] += 0.001
                prob.clamp_(0.99, 0.9999)
                reset *= 0
                proj_success_rate *= 0
                flip_success_rate *= 0

            # query count
            q_num += 2
            Q[unsuccessful_indices] += 2

            # update indices for unsuccessful perturbations
            l, _ = delta[unsuccessful_indices].abs().reshape(b_cur, -1).max(1)
            unsuccessful_indices[unsuccessful_indices.nonzero().squeeze(1)[(l <= self.epsilon).nonzero().squeeze(1)]] = 0

            # print attack information
            if q_num % 10000 == 0:
                print(f"Queries: {q_num}/{self.max_queries} Successfully attacked images: {b - unsuccessful_indices.sum()}/{b}")

            if unsuccessful_indices.sum() == 0:
                break

        print('attack finished!')
        print(f"Queries: {q_num}/{self.max_queries} Successfully attacked images: {b - unsuccessful_indices.sum()}/{b}")
        return (x + delta).clamp(0, 1), Q.cpu().numpy()


    def resize(self, x, h, w):
        return F.interpolate(x, size=[h, w], mode='bilinear', align_corners=False)


    def binary_infinity(self, x_a, x, y, k):
        '''
        linf binary search
        :param k: the number of binary search iteration
        '''
        b = x_a.size(0)
        l = torch.zeros(b)
        u, _ = (x_a - x).reshape(b, -1).abs().max(1)
        for _ in range(k):
            mid = (l + u) / 2
            adv = self.project_infinity(x_a, x, mid).clamp(0, 1)
            check =self.is_adversarial(adv, y)
            u[check.nonzero().squeeze(1)] = mid[check.nonzero().squeeze(1)]
            check = check < 1
            l[check.nonzero().squeeze(1)] = mid[check.nonzero().squeeze(1)]
        return self.project_infinity(x_a, x, u).clamp(0, 1)


    def project_infinity(self, x_a, x, l):
        '''
        linf projection
        '''
        return torch.max(x - l[:, None, None, None], torch.min(x_a, x + l[:, None, None, None]))
    
'''

Original License

MIT License

Copyright (c) 2020 cwllenny

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
