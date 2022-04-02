'''

This file is modified based on the following source:
link: https://github.com/thu-ml/ares/blob/main/pytorch_ares/pytorch_ares/attack_torch/evolutionary.py

The original license is placed at the end of this file.

@inproceedings{dong2019efficient,
  title={Efficient decision-based black-box adversarial attacks on face recognition},
  author={Dong, Yinpeng and Su, Hang and Wu, Baoyuan and Li, Zhifeng and Liu, Wei and Zhang, Tong and Zhu, Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7714--7722},
  year={2019}
}

The update include:
  1. args and config
  2. functions modification
  3. save process
  
basic structure for main:
  1. config args and prior setup
  2. define functions that implement evolutionary attack
  3. define perturbations
  
'''

"""
Implements Evolutionary Attack
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import scipy.misc
import scipy.ndimage
import scipy
from torch import Tensor as t

from attacks.decision.decision_black_box_attack import DecisionBlackBoxAttack

class EvolutionaryAttack(DecisionBlackBoxAttack):
    """
    Evolutionary Attack
    """
    def __init__(self, epsilon, p, max_queries, lb, ub, batch_size, sub):
        super().__init__(max_queries = max_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size = batch_size)
        self.sub = sub

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "attack_name": self.__class__.__name__
        }

    def loss(self, x0, y0, x_):
        if ~self.is_adversarial(x_, y0):
            return np.inf
        else:
            return self.distance(x_, x0)
        
    def evolutionary(self, x0, y0):
        num_directions = 100
        best_dir, best_dist = None, float('inf')
            
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        for _ in range(num_directions):
            theta = torch.randn_like(x0)
            x_ = torch.clamp(x0 + theta, 0, 1)
            if self.is_adversarial(x_, y0):
                if self.distance(x_, x0) < best_dist:
                    best_dir, best_dist = x_, self.distance(x_, x0)
                    print("--------> Found distortion %.4f" % best_dist)

        if best_dist == float('inf'):    
            print("Couldn't find valid initial, failed")
            return x0, num_directions

        print("==========> Found best distortion %.4f "
              "using %d queries" % (best_dist, num_directions))

        # Hyperparameters
        #sigma = 0.01
        f = self.sub
        cc = 0.01
        cconv = 0.001
        shape = list(x0.shape)
        s = int(shape[1] / f)
        m = s * s * 3
        m_shape = (s, s, 3)
        k = int(m/20)
        mu = 0.1
        
        # Hyperparameter tuning - 1/5 success rule
        MAX_PAST_TRIALS = 15
        success_past_trials = 0
        
        # Initializations
        C = np.identity(m)
        x_ = best_dir
        pc = np.zeros(m)
        
        prev_loss = best_dist
        q = 0
        old_it = 0
        for it in range(self.max_queries):

            if it%10 == 0:
                print("Iteration: ", it, " mu: ", mu, "norm: ", prev_loss)

            # Update hyperparameters
            if it > MAX_PAST_TRIALS and it%5==0:
                p = success_past_trials/MAX_PAST_TRIALS
                mu = mu*np.exp(p - 0.2)
            sigma = 0.01 * self.distance(x_, x0)
            
            z = np.random.multivariate_normal(np.zeros(m), (sigma**2)*C)
            
            
            # Select k coordinates with probability proportional to C
            probs = np.exp(np.diagonal(C))
            probs /= sum(probs)
            indices = np.random.choice(m, size=k, replace=False, p=probs)
            
            # Set non selected coordinates to 0
            indices_zero = np.setdiff1d(np.arange(m), indices)
            z[indices_zero] = 0
            
            z_ = scipy.ndimage.zoom(z.reshape(m_shape), [f, f, 1], order=1)
            z_ = t(z_) + mu*(x0 - x_)
            
            x_new = torch.clamp(x_ + z_, 0, 1)
            new_loss = self.loss(x0, y0, x_new)
            q += 1
            success = new_loss < prev_loss
            
            if success:
                # Update x_
                x_ = x_new
                if it - old_it >= 200:
                    break
                norm = new_loss
                old_it = it
                print("Found adv with distortion {0} Queries {1}".format(norm, it))
                if norm < self.epsilon:
                    break
                
                # Update pc and C
                pc = (1-cc)*pc + z*np.sqrt(cc*(2-cc))/sigma

                c_diag = np.diagonal(C)
                c_diag = (1-cconv)*c_diag + cconv*np.square(pc)
                C = np.diag(c_diag)
                
                # Update loss
                prev_loss = new_loss
            
            # Update past success trials.
            if success:
                success_past_trials += 1
            else:
                success_past_trials -= 1
            success_past_trials = np.clip(success_past_trials, 0, MAX_PAST_TRIALS)

        return x_, num_directions + q


    def _perturb(self, xs_t, ys_t):
        if self.targeted:
            raise NotImplementedError
        else:
            adv, q = self.evolutionary(xs_t, ys_t)
        return adv, q
