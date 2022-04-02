'''

This file is modified based on the following source:
link: https://github.com/cmhcbb/attackbox/blob/master/attack

The original license is placed at the end of this file.

@article{cheng2019sign,
  title={Sign-opt: A query-efficient hard-label adversarial attack},
  author={Cheng, Minhao and Singh, Simranjit and Chen, Patrick and Chen, Pin-Yu and Liu, Sijia and Hsieh, Cho-Jui},
  journal={arXiv preprint arXiv:1909.10773},
  year={2019}
}

The update include:
  1. args and configs
  2. functions modification
 
basic structure for main:
    1. config args and prior setup
    2. define functions that implement sign-opt attack, gradient evaluation with two formulas and binary search
    3. define perturbation
    
'''

"""
Implements Sign_OPT
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import time
import scipy.spatial
from scipy.linalg import qr
import random
from torch import Tensor as t

from attacks.decision.decision_black_box_attack import DecisionBlackBoxAttack

start_learning_rate = 1.0

def quad_solver(Q, b):
    """
    Solve min_a  0.5*aQa + b^T a s.t. a>=0
    """
    K = Q.shape[0]
    alpha = torch.zeros((K,))
    g = b
    Qdiag = torch.diag(Q)
    for _ in range(20000):
        delta = torch.maximum(alpha - g/Qdiag,0) - alpha
        idx = torch.argmax(torch.abs(delta))
        val = delta[idx]
        if abs(val) < 1e-7: 
            break
        g = g + val*Q[:,idx]
        alpha[idx] += val
    return alpha

def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = torch.sign(y)
    y_sign[y_sign==0] = 1
    return y_sign


class SignOPTAttack(DecisionBlackBoxAttack):
    """
    Sign_OPT
    """

    def __init__(self, epsilon, p, alpha, beta, svm, momentum, max_queries, k, lb, ub, batch_size, sigma):
        super().__init__(max_queries = max_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size = batch_size)
        self.alpha = alpha
        self.beta = beta
        self.svm = svm
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.query_count = 0


    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "attack_name": self.__class__.__name__
        }

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001):
        """ 
        Attack the original image and return adversarial example
        """

        y0 = y0[0]
        self.query_count = 0

        # Calculate a good starting point.
        num_directions = 10
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        for i in range(num_directions):
            self.query_count += 1
            theta = torch.randn_like(x0)
            if self.predict_label(x0+theta)!=y0:
                initial_lbd = torch.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(x0, y0, theta, initial_lbd, g_theta)
                self.query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)

        timeend = time.time()        
        if g_theta == float('inf'):    
            print("Couldn't find valid initial, failed")
            return x0, self.query_count
        print("==========> Found best distortion %.4f in %.4f seconds "
              "using %d queries" % (g_theta, timeend-timestart, self.query_count))

        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        vg = torch.zeros_like(xg)

        for i in range(1500):
            if self.svm == True:
                sign_gradient, grad_queries = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta)
            else:
                sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)
            self.query_count += grad_queries
            # Line search
            min_theta = xg
            min_g2 = gg
            min_vg = vg
            for _ in range(15):
                if self.momentum > 0:
                    new_vg = self.momentum*vg - alpha*sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta /= torch.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                self.query_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    if self.momentum > 0:
                        min_vg = new_vg
                else:
                    break
            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    if self.momentum > 0:
                        new_vg = self.momentum*vg - alpha*sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta /= torch.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    self.query_count += count
                    if new_g2 < gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        if self.momentum > 0:
                            min_vg = new_vg
                        break
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            xg, gg = min_theta, min_g2
            vg = min_vg
            

            if self.query_count > self.max_queries:
               break

            dist = self.distance(gg*xg)
            if dist < self.epsilon:
                break

            print("Iteration %3d distortion %.4f num_queries %d" % (i+1, dist, self.query_count))

        dist = self.distance(gg*xg)
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f queries %d \nTime: %.4f seconds" % (dist, self.query_count, timeend-timestart))
        return x0 + gg*xg, self.query_count

    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k
        sign_grad = torch.zeros_like(theta)
        queries = 0
        for _ in range(K):          
            u = torch.randn_like(theta)
            u /= torch.norm(u)
            
            sign = 1
            new_theta = theta + h*u
            new_theta /= torch.norm(new_theta)
            
            # Targeted case.
            if (target is not None and 
                self.predict_label(x0+initial_lbd*new_theta) == target):
                sign = -1
                
            # Untargeted case
            if (target is None and
                self.predict_label(x0+t(initial_lbd*new_theta)) != y0):
                sign = -1
            queries += 1
            sign_grad += u*sign
        
        sign_grad /= K    
        
        return sign_grad, queries

    def sign_grad_svm(self, x0, y0, theta, initial_lbd, h=0.001, K=100, lr=5.0, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = torch.zeros_like(theta)
        queries = 0
        dim = np.prod(theta.shape)
        X = torch.zeros((dim, K))
        for iii in range(K):
            u = torch.randn_like(theta)
            u /= torch.norm(u)
            
            sign = 1
            new_theta = theta + h*u
            new_theta /= torch.norm(new_theta)            
            
            # Targeted case.
            if (target is not None and 
                self.predict_label(x0+t(initial_lbd*new_theta)) == target):
                sign = -1
                
            # Untargeted case
            if (target is None and
                self.predict_label(x0+t(initial_lbd*new_theta)) != y0):
                sign = -1
                
            queries += 1
            X[:,iii] = sign*u.reshape((dim,))
        
        Q = X.transpose().dot(X)
        q = -1*torch.ones((K,))
        # G = torch.diag(-1*torch.ones((K,)))
        # h = torch.zeros((K,))
        ### Use quad_qp solver 
        #alpha = solve_qp(Q, q, G, h)
        ### Use coordinate descent solver written by myself, avoid non-positive definite cases
        alpha = quad_solver(Q, q)
        sign_grad = (X.dot(alpha)).reshape(theta.shape)
        
        return sign_grad, queries

    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        if self.predict_label(x0+lbd*theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.predict_label(x0+lbd_hi*theta) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.predict_label(x0+lbd_lo*theta) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1
                if nquery + self.query_count> self.max_queries:
                    break

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if nquery + self.query_count> self.max_queries:
                break
            if self.predict_label(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if self.predict_label(x0+t(current_best*theta)) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if nquery + self.query_count> self.max_queries:
                break
            if self.predict_label(x0 + t(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def attack_targeted(self, x0, target, alpha = 0.2, beta = 0.001):
        """ 
        Attack the original image and return adversarial example
        """

        target = target[0]

        num_samples = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        ls_total = 0
        sample_count = 0
        print("Searching for the initial direction on %d samples: " % (num_samples))
        timestart = time.time()


        # Iterate through training dataset. Find best initial point for gradient descent.
        for i in range(500):
            xi, _ = self.train_dataset.get_eval_data(i,i+1)
            yi_pred = self.predict_label(xi)
            query_count += 1
            if yi_pred != target:
                continue
            theta = xi - x0
            initial_lbd = torch.norm(theta)
            theta /= initial_lbd
            lbd, count = self.fine_grained_binary_search_targeted(x0, target, theta, initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print("--------> Found distortion %.4f" % g_theta)

            sample_count += 1
            if sample_count >= num_samples:
                break
     

        timeend = time.time()
        if g_theta == np.inf:
            return x0, query_count
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" %
              (g_theta, timeend-timestart, query_count))
    
        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta

        for i in range(1000):
            if self.svm == True:
                sign_gradient, grad_queries = self.sign_grad_svm(x0, 0, xg, initial_lbd=gg, h=beta, target=target)
            else:
                sign_gradient, grad_queries = self.sign_grad_v1(x0, 0, xg, initial_lbd=gg, h=beta, target=target)

            
            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * sign_gradient
                new_theta /= torch.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local_targeted(x0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * sign_gradient
                    new_theta /= torch.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local_targeted(x0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break
                        
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            xg, gg = min_theta, min_g2
            
            query_count += (grad_queries + ls_count)
            ls_total += ls_count

            if query_count > self.max_queries:
                break

            dist = self.distance(gg*xg)
            if dist < self.epsilon:
                break
            
            if i%5==0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, dist, query_count))

        adv_target = self.predict_label(x0 + t(gg*xg))
        if (adv_target == target):
            timeend = time.time()
            print("\nAdversarial Example Found Successfully: distortion %.4f target"
                  " %d queries %d LS queries %d \nTime: %.4f seconds" % (dist, target, query_count, ls_total, timeend-timestart))

            return x0 + t(gg*xg), query_count
        else:
            print("Failed to find targeted adversarial example.")
            return x0, query_count
    
    def fine_grained_binary_search_local_targeted(self, x0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if self.predict_label(x0 + t(lbd*theta)) != t:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.predict_label(x0 + t(lbd_hi*theta)) != t:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100: 
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.predict_label(x0 + t(lbd_lo*theta)) == t:
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.predict_label(x0 + t(lbd_mid*theta)) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, x0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if self.predict_label(x0 + t(current_best*theta)) != t:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.predict_label(x0 + t(lbd_mid*theta)) != t:
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
        return lbd_hi, nquery


    def _perturb(self, xs_t, ys):
        if self.targeted:
            adv, q = self.attack_targeted(xs_t, ys, self.alpha, self.beta)
        else:
            adv, q = self.attack_untargeted(xs_t, ys, self.alpha, self.beta)

        return adv, q
      
'''

Original License

MIT License

Copyright (c) 2022 Minhao Cheng
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
