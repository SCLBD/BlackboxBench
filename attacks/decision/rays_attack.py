from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''

This file is copied from the following source:
link: https://github.com/uclaml/RayS

The original license is placed at the end of this file.

@inproceedings{chen2020rays,
  title={RayS: A Ray Searching Method for Hard-label Adversarial Attack},
  author={Chen, Jinghui and Gu, Quanquan},
  booktitle={Proceedings of the 26rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2020}
}

basic structure for main:
  1. config args and prior setup
  2. define functions that implement rays attack and binary search for decision boundary along sgn direction
  3. define purturbation
  
'''

"""
Implements RayS Attack
"""
import numpy as np
import torch
from torch import Tensor as t

from attacks.decision.decision_black_box_attack import DecisionBlackBoxAttack

class RaySAttack(DecisionBlackBoxAttack):
    """
    RayS
    """

    def __init__(self, epsilon, p, max_queries, lb, ub, batch_size):
        super().__init__(max_queries = max_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size = batch_size)
        self.lin_search_rad = 10
        self.pre_set = {1, -1}


    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "attack_name": self.__class__.__name__
        }

    def get_xadv(self, x, v, d):
        if isinstance(d, int):
            d = torch.tensor(d).repeat(len(x))
        out = x + d.view(len(x), 1, 1, 1) * v
        out = torch.clamp(out, self.lb, self.ub)
        return out

    def attack_hard_label(self, x, y, target=None):
        """ 
        Attack the original image and return adversarial example
        """
        shape = list(x.shape)
        dim = np.prod(shape[1:])

        # init variables
        self.queries = torch.zeros_like(y)
        self.sgn_t = torch.sign(torch.ones(shape))
        self.d_t = torch.ones_like(y).float().fill_(float("Inf"))
        working_ind = (self.d_t > self.epsilon).nonzero().flatten()

        stop_queries = self.queries.clone()
        dist = self.d_t.clone()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
 
        block_level = 0
        block_ind = 0
        for _ in range(self.max_queries):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            valid_mask = (self.queries < self.max_queries) 
            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, target, attempt, valid_mask)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm((self.x_final - x).view(shape[0], -1), np.inf, 1)
            stop_queries[working_ind] = self.queries[working_ind]
            working_ind = (dist > self.epsilon).nonzero().flatten()
            if working_ind.size(0) == 0:
                break

            if torch.sum(self.queries >= self.max_queries) == shape[0]:
                print('out of queries')
                break

            # print('d_t: %.4f | adbd: %.4f | queries: %.4f | rob acc: %.4f | iter: %d' % (torch.mean(self.d_t), torch.mean(dist), torch.mean(self.queries.float()), len(working_ind) / len(x), i + 1))
 

        stop_queries = torch.clamp(stop_queries, 0, self.max_queries)
        return self.x_final, stop_queries.cpu().numpy()

    # check whether solution is found
    def search_succ(self, x, y, target, mask):
        self.queries[mask] += 1
        return self.is_adversarial(x[mask], y[mask])

    # binary search for decision boundary along sgn direction
    def binary_search(self, x, y, target, sgn, valid_mask, tol=1e-3):
        sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

        d_start = torch.zeros_like(y).float()
        d_end = self.d_t.clone()

        initial_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target, valid_mask)
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            search_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target, to_search_ind)
            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]


    def _perturb(self, xs_t, ys_t):
        if self.targeted:
            adv, q = self.attack_hard_label(xs_t, ys_t, target=ys_t)
        else:
            adv, q = self.attack_hard_label(xs_t, ys_t, target=None)

        return adv, q
      
'''
 
Original License
 
MIT License

Copyright (c) 2020 Jinghui Chen, Quanquan Gu

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
