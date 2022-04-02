'''

This file is modified based on the following source:
link: https://github.com/thisisalirah/GeoDA

@inproceedings{rahmati2020geoda,
  title={Geoda: a geometric framework for black-box adversarial attacks},
  author={Rahmati, Ali and Moosavi-Dezfooli, Seyed-Mohsen and Frossard, Pascal and Dai, Huaiyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8446--8455},
  year={2020}
}

The update include:
  1. args and config
  2. functions modification
  
basic structure for main:
  1. config args and prior setup
  2. define functions that generate sub-noises and implement GeoDA algorithm
  3. define perturbations and return results 
  
'''

"""
Implements GeoDA
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os
import time
from torch import Tensor as t
from numpy import linalg 
import math

from attacks.decision.decision_black_box_attack import DecisionBlackBoxAttack

from math import cos, sqrt, pi


def dct(x, y, v, u, n):
    # Normalisation
    def alpha(a):
        if a == 0:
            return sqrt(1.0 / n)
        else:
            return sqrt(2.0 / n)

    return alpha(u) * alpha(v) * cos(((2 * x + 1) * (u * pi)) / (2 * n)) * cos(((2 * y + 1) * (v * pi)) / (2 * n))


def generate_2d_dct_basis(sub_dim, n, path):
    # Assume square image, so we don't have different xres and yres

    # We can get different frequencies by setting u and v
    # Here, we have a max u and v to loop over and display
    # Feel free to adjust
    maxU = sub_dim
    maxV = sub_dim

    dct_basis = []
    for u in range(0, maxU):
        for v in range(0, maxV):
            basisImg = np.zeros((n, n))
            for y in range(0, n):
                for x in range(0, n):
                    basisImg[y, x] = dct(x, y, v, u, max(n, maxV))
            dct_basis.append(basisImg)
    dct_basis = np.mat(np.reshape(dct_basis, (maxV*maxU, n*n))).transpose()
    np.save(path, dct_basis)
    return dct_basis

class SubNoise(torch.nn.Module):
    """given subspace x and the number of noises, generate sub noises"""
    # x is the subspace basis
    def __init__(self, num_noises, x):
        self.num_noises = num_noises
        self.x = x
        self.size = int(self.x.shape[0] ** 0.5)
        super(SubNoise, self).__init__()

    def forward(self):
        r = torch.zeros([self.size ** 2, 3*self.num_noises], dtype=torch.float32)
        noise = torch.randn([self.x.shape[1], 3*self.num_noises], dtype=torch.float32).cuda()
        sub_noise = torch.transpose(torch.mm(self.x, noise), 0, 1)
        r = sub_noise.view([self.num_noises, 3, self.size, self.size])
        r_list = r.permute(0,2,3,1)
        return r_list

class GeoDAttack(DecisionBlackBoxAttack):
    """
    GeoDA
    """
    def __init__(self, epsilon, p, max_queries, sub_dim, tol, alpha, mu, search_space, grad_estimator_batch_size, lb, ub, batch_size, sigma):
        super().__init__(max_queries = max_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size = batch_size)
        self.sub_dim = sub_dim
        self.tol = tol
        self.alpha = alpha
        self.mu = mu
        self.search_space = search_space
        self.grad_estimator_batch_size = grad_estimator_batch_size
        self.sigma = sigma

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "attack_name": self.__class__.__name__
        }

    def opt_query_iteration(self, Nq, T, eta): 
        coefs=[eta**(-2*i/3) for i in range(0,T)]
        coefs[0] = 1*coefs[0]
        sum_coefs = sum(coefs)
        opt_q=[round(Nq*coefs[i]/sum_coefs) for i in range(0,T)]
        if opt_q[0]>80:
            T = T + 1
            opt_q, T = self.opt_query_iteration(Nq, T, eta)
        elif opt_q[0]<50:
            T = T - 1
            opt_q, T = self.opt_query_iteration(Nq, T, eta)

        return opt_q, T

    def find_random_adversarial(self, x0, y0, epsilon=1000):

        num_calls = 1
        
        step = 0.02
        perturbed = x0
        
        while self.predict_label(perturbed) == y0[0]: 
            pert = torch.randn(x0.shape)

            perturbed = x0 + num_calls*step* pert
            perturbed = torch.clamp(perturbed, 0, 1)
            num_calls += 1
            
        return perturbed, num_calls 

    def bin_search(self, x_0, y_0, x_random, tol):
        num_calls = 0
        adv = x_random
        cln = x_0
        while True:
            
            mid = (cln + adv) / 2.0
            num_calls += 1
            
            if self.predict_label(mid) != y_0[0]:
                adv = mid
            else:
                cln = mid

            if torch.norm(adv-cln)<tol:
                break

        return adv, num_calls 

    def go_to_boundary(self, x_0, y_0, grad):

        epsilon = 1

        num_calls = 1
        perturbed = x_0 

        if self.p == '2':   
            grads = grad
        elif self.p == 'inf':            
            grads = torch.sign(grad)/torch.norm(grad)
            
        while self.predict_label(perturbed) == y_0[0]:
            perturbed = x_0 + (num_calls*epsilon*grads[0])
            perturbed = torch.clamp(perturbed, 0, 1)

            num_calls += 1
            
            if num_calls > 100:
                print('falied ... ')
                break
        return perturbed, num_calls

    def black_grad_batch(self, x_boundary, q_max, random_noises, y_0, sub_basis_torch):
        grad_tmp = [] # estimated gradients in each estimate_batch
        z = []        # sign of grad_tmp
        outs = []
        batch_size = self.grad_estimator_batch_size

        num_batchs = math.ceil(q_max/batch_size)
        last_batch = q_max - (num_batchs-1)*batch_size
        EstNoise = SubNoise(batch_size, sub_basis_torch)
        all_noises = []
        
        for j in range(num_batchs):
            if j == num_batchs-1:
                EstNoise_last = SubNoise(last_batch, sub_basis_torch)
                current_batch = EstNoise_last()
                current_batch_np = current_batch.cpu().numpy()
                noisy_boundary = [x_boundary[0,:,:,:].cpu().numpy()]*last_batch +self.alpha*current_batch.cpu().numpy()

            else:
                current_batch = EstNoise()
                current_batch_np = current_batch.cpu().numpy()
                noisy_boundary = [x_boundary[0,:,:,:].cpu().numpy()]*batch_size +self.alpha*current_batch.cpu().numpy()
            
            all_noises.append(current_batch_np) 
            
            noisy_boundary_tensor = torch.tensor(noisy_boundary)
            
            predict_labels = self.predict_label(noisy_boundary_tensor)
            
            outs.append(predict_labels.cpu())

        all_noise = np.concatenate(all_noises, axis=0)
        outs = np.concatenate(outs, axis=0)
            

        for i, predict_label in enumerate(outs):
            if predict_label == y_0:
                z.append(1)
                grad_tmp.append(all_noise[i])
            else:
                z.append(-1)
                grad_tmp.append(-all_noise[i])
        
        grad = -(1/q_max)*sum(grad_tmp)
        
        grad_f = torch.tensor(grad)[None, :,:,:]

        return grad_f, sum(z)
    
    

    def GeoDA(self, x_0, y_0, x_b, iteration, q_opt, sub_basis_torch):
        q_num = 0
        grad = 0
        
        for i in range(iteration):
        
            t1 = time.time()
            random_vec_o = torch.randn(q_opt[i], x_0.shape[1], x_0.shape[2], x_0.shape[3])

            grad_oi, _ = self.black_grad_batch(x_b, q_opt[i], random_vec_o, y_0[0], sub_basis_torch)
            q_num = q_num + q_opt[i]

            grad = grad_oi + grad
            x_adv, qs = self.go_to_boundary(x_0, y_0, grad)
            q_num = q_num + qs

            x_adv, bin_query = self.bin_search(x_0, y_0, x_adv, self.tol)
            q_num = q_num + bin_query

            x_b = x_adv
            
            t2 = time.time()
            
            norm = self.distance(x_adv, x_0)
            message = ' (took {:.5f} seconds)'.format(t2 - t1)
            print('iteration -> ' + str(i) + str(message) + '     -- ' + self.p + ' norm is -> ' + str(norm))
            if norm < self.epsilon:
                break
            if q_num > self.max_queries:
                break
            
            
        x_adv = torch.clamp(x_adv, 0, 1)
            
        return x_adv, q_num, grad

    def _perturb(self, xs_t, ys):

        if self.search_space == 'sub':
            print('Check if DCT basis available ...')
            
            path = os.path.join(os.path.dirname(__file__), '2d_dct_basis_{}_{}.npy'.format(self.sub_dim,xs_t.size(2)))
            if os.path.exists(path):
                print('Yes, we already have it ...')
                sub_basis = np.load(path).astype(np.float32)
            else:
                print('Generating dct basis ......')
                sub_basis = generate_2d_dct_basis(self.sub_dim, xs_t.size(2), path).astype(np.float32)
                print('Done!\n')


            sub_basis_torch = torch.from_numpy(sub_basis).cuda()


        x_random, query_random_1 = self.find_random_adversarial(xs_t, ys, epsilon=100)     
        
        # Binary search
        
        x_boundary, query_binsearch_2 = self.bin_search(xs_t, ys, x_random, self.tol)
        x_b = x_boundary
        
        
        query_rnd = query_binsearch_2 + query_random_1


        iteration = round(self.max_queries/500) 
        q_opt_it = int(self.max_queries  - (iteration)*25)
        q_opt_iter, iterate = self.opt_query_iteration(q_opt_it, iteration, self.mu)
        q_opt_it = int(self.max_queries  - (iterate)*25)
        q_opt_iter, iterate = self.opt_query_iteration(q_opt_it, iteration, self.mu)
        print('Start: The GeoDA will be run for:' + ' Iterations = ' + str(iterate) + ', Query = ' + str(self.max_queries) + ', Norm = ' + str(self.p)+ ', Space = ' + str(self.search_space) )


        t3 = time.time()
        adv, query_o, _= self.GeoDA(xs_t, ys, x_b, iterate, q_opt_iter, sub_basis_torch)
        t4 = time.time()
        message = ' took {:.5f} seconds'.format(t4 - t3)
        qmessage = ' with query = ' + str(query_o + query_rnd)


        print('End: The GeoDA algorithm' + message + qmessage )

            
        return adv, query_o + query_rnd
