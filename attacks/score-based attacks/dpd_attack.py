"""
Implements DPD attacks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch 
from torch import Tensor as t

from attacks.score.score_black_box_attack import ScoreBlackBoxAttack
from utils.compute import lp_step
from models.rsd import ResnetGenerator, weights_init

def norm(t):
    assert len(t.shape) == 4
    norm_vec = ch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float() * 1e-8
    return norm_vec

class DPDAttack(ScoreBlackBoxAttack):
    """
    DPD Attack
    """

    def __init__(self, max_loss_queries, epsilon, p, lb, ub):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_extra_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub)
        self.net_lr = 0.01
        self.alpha = 0.001
        self.beta = 0.1
        self.gradient_iters = 1
        self.image_lr = 0.001
        self.norm_weight = 0.00001
        self.gene_net1 = None
        self.optimizer_g1 = None
        self.gene_net2 = None
        self.optimizer_g2 = None

    def gene_out1(self):
        inp = torch.randn_like(self.in_pattern)
        gene_delta = self.gene_net1(inp) * 0.5
        return gene_delta, inp

    def gene_out2(self):
        inp = torch.randn_like(self.in_pattern)
        gene_delta = self.gene_net2(inp) * 0.5
        return gene_delta, inp


    def grad_est(self,x, u, v, n_u, n_v, loss_fct):
        n_u = n_u / norm(n_u) * norm(u.detach()) * 0.5
        n_v = n_v / norm(n_v) * norm(v.detach()) * 0.5

        ud = u.detach() + n_u
        vd = v.detach() + n_v
        xd = x.detach()

        f_x = t(loss_fct(xd.cpu().numpy().transpose(0,2,3,1)))
        f_u = t(loss_fct((xd + self.alpha * ud).cpu().numpy().transpose(0,2,3,1)))
        f_v = t(loss_fct((xd + self.alpha * vd).cpu().numpy().transpose(0,2,3,1)))
        f_uv = t(loss_fct((xd + self.alpha * ud + self.alpha * self.beta * vd).cpu().numpy().transpose(0,2,3,1)))
        f_vu = t(loss_fct((xd + self.alpha * vd + self.alpha * self.beta * ud).cpu().numpy().transpose(0,2,3,1)))

        g_u = -(f_u - f_x) / self.alpha * (f_uv - f_u) / (self.alpha * self.beta) * vd + (f_uv - f_x) / self.alpha * (f_v - f_x) / self.alpha * vd
        g_v = -(f_v - f_x) / self.alpha * (f_vu - f_v) / (self.alpha * self.beta) * ud + (f_vu - f_x) / self.alpha * (f_u - f_x) / self.alpha * ud
        g_x = (f_u - f_x) / self.alpha * ud + (f_v - f_x) / self.alpha * vd + \
              (f_uv - f_x) / self.alpha * (ud + self.beta * vd) + (f_vu - f_x) / self.alpha * (vd + self.beta * ud) + \
              (f_u - f_v) / self.alpha * (ud - vd) + (f_uv - f_u) / (self.alpha * self.beta) * vd + \
              (f_vu - f_v) / (self.alpha * self.beta) * ud + (f_uv - f_vu) / self.alpha * (ud + self.beta * vd - vd - self.beta * ud)

        return g_x.detach() / 8.0, g_u.detach(), g_v.detach(), f_x.detach().item()

    def _perturb(self, xs_t, loss_fct):
        xs_t = xs_t.permute(0,3,1,2)

        if self.is_new_batch:
            self.in_pattern = xs_t.clone()
            self.gene_net1 = ResnetGenerator(3, 3, 8, norm_type='instance', act_type='relu', gpu_ids=[int(0)])
            self.gene_net1.apply(weights_init)
            self.optimizer_g1 = torch.optim.Adam(self.gene_net1.parameters(), lr=self.net_lr, betas=(0.5, 0.999), weight_decay=0.0001)
            self.gene_net1.train()

            self.gene_net2 = ResnetGenerator(3, 3, 8, norm_type='instance', act_type='relu', gpu_ids=[int(0)])
            self.gene_net2.apply(weights_init)
            self.optimizer_g2 = torch.optim.Adam(self.gene_net2.parameters(), lr=self.net_lr, betas=(0.5, 0.999), weight_decay=0.0001)
            self.gene_net2.train()

            self.l_vals, self.last_u, self.lu_ort, self.last_v, self.lv_ort = [], None, 0, None, 0

        prior_x = torch.zeros_like(xs_t)
        for _ in range(self.gradient_iters):
            # generate perturbations
            u, n_u = self.gene_out1()
            v, n_v = self.gene_out2()

            # estimate gradients
            grad_x, grad_u, grad_v, l_val = self.grad_est(x=xs_t, u=u, v=v, n_u=n_u, n_v=n_v, loss_fct= loss_fct)

            # update distributions 1
            if self.last_u is not None:
                self.lu_ort = (torch.sum(u * self.last_u, (1, 2, 3)) / (norm(self.last_u) * torch.sqrt(torch.sum(u * u, (1, 2, 3))))) ** 2
            lu_norm = self.norm_weight * torch.mean(torch.sum(u * u, (1, 2, 3)), 0)
            lu_grad = torch.sum(u * grad_u, (1, 2, 3)) / (norm(grad_u) * torch.sqrt(torch.sum(u * u, (1, 2, 3))))

            loss_u = lu_grad + lu_norm + self.lu_ort
            loss_u.backward()

            # update distributions 2
            detach_u = u.detach()
            lvu_ort = (torch.sum(v * detach_u, (1, 2, 3)) / (norm(detach_u) * torch.sqrt(torch.sum(v * v, (1, 2, 3))))) ** 2
            if self.last_v is not None:
                self.lv_ort = (torch.sum(v * self.last_v, (1, 2, 3)) / (norm(self.last_v) * torch.sqrt(torch.sum(v * v, (1, 2, 3))))) ** 2
            lv_norm = self.norm_weight * torch.mean(torch.sum(v * v, (1, 2, 3)), 0)
            lv_grad = torch.sum(v * grad_v, (1, 2, 3)) / (norm(grad_v) * torch.sqrt(torch.sum(v * v, (1, 2, 3))))

            loss_v = lv_grad + lv_norm + self.lv_ort + lvu_ort
            loss_v.backward()

            # update gradients
            prior_x += grad_x

            # record
            self.last_u = u.detach()
            self.last_v = v.detach()
            self.l_vals.append(l_val)

        self.optimizer_g1.step()
        self.gene_net1.zero_grad()
        self.optimizer_g2.step()
        self.gene_net2.zero_grad()
        
        return (xs_t + prior_x * self.image_lr).permute(0,3,1,2), 5 * self.gradient_iters * np.ones(xs_t.size(0))

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_extra_queries": "inf" if np.isinf(self.max_extra_queries) else self.max_extra_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "attack_name": self.__class__.__name__
        }
