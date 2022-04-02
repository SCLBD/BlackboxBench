'''

This file is copied from the following source:
link: https://github.com/cmhcbb/attackbox/blob/master/attack/HSJA.py

The original license is placed at the end of this file.

@inproceedings{chen2020hopskipjumpattack,
  title={Hopskipjumpattack: A query-efficient decision-based attack},
  author={Chen, Jianbo and Jordan, Michael I and Wainwright, Martin J},
  booktitle={2020 ieee symposium on security and privacy (sp)},
  pages={1277--1294},
  year={2020},
  organization={IEEE}
}

basic structure for main:
  1. config args and prior setup
  2. define functions that implement HSJA algorithm, boundary decision, 
     gradient estimation, binary search to approach boundary and geometric progression
  
'''

"""
Implements HSJA
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import Tensor as t

from attacks.decision.decision_black_box_attack import DecisionBlackBoxAttack


class HSJAttack(DecisionBlackBoxAttack):
    """
    HSJA
    """
    def __init__(self, epsilon, p, max_queries, gamma, stepsize_search, max_num_evals, init_num_evals, EOT, sigma, lb, ub, batch_size):
        super().__init__(max_queries = max_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size = batch_size)
        self.gamma = gamma
        self.stepsize_search = stepsize_search
        self.max_num_evals = max_num_evals
        self.init_num_evals = init_num_evals
        self.verbose = True
        self.EOT = EOT
        self.sigma = sigma

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "attack_name": self.__class__.__name__
        }

    def hsja(self,input_xi,label_or_target):

        d = int(np.prod(input_xi.shape))
        # Set binary search threshold.
        if self.p == '2':
                theta = self.gamma / (np.sqrt(d) * d)
        else:
                theta = self.gamma / (d ** 2)

        self.query = 0
        # Initialize.
        perturbed = self.initialize(input_xi, label_or_target)
        

        # Project the initialization to the boundary.
        perturbed, dist_post_update = self.binary_search_batch(input_xi, perturbed, label_or_target, theta)
        dist = self.compute_distance(perturbed, input_xi)

        for j in np.arange(10000):
                # Choose delta.
                if j==1:
                    delta = 0.1 * (self.ub - self.lb)
                else:
                    if self.p == '2':
                            delta = np.sqrt(d) * theta * dist_post_update
                    elif self.p == 'inf':
                            delta = d * theta * dist_post_update        

                # Choose number of evaluations.
                num_evals = int(self.init_num_evals * np.sqrt(j+1))
                num_evals = int(min([num_evals, self.max_num_evals]))

                # approximate gradient.
                gradf = self.approximate_gradient(perturbed, label_or_target, num_evals, delta)
                if self.p == 'inf':
                        update = np.sign(gradf)
                else:
                        update = gradf

                # search step size.
                if self.stepsize_search == 'geometric_progression':
                        # find step size.
                        epsilon = self.geometric_progression_for_stepsize(perturbed, label_or_target, 
                                update, dist, j+1)

                        # Update the sample. 
                        perturbed = self.clip_image(perturbed + epsilon * update, 
                                self.lb, self.ub)

                        # Binary search to return to the boundary. 
                        perturbed, dist_post_update = self.binary_search_batch(input_xi, 
                                perturbed[None], label_or_target, theta)

                elif self.stepsize_search == 'grid_search':
                        # Grid search for stepsize.
                        epsilons = np.logspace(-4, 0, num=20, endpoint = True) * dist
                        epsilons_shape = [20] + len(input_xi.shape) * [1]
                        perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                        perturbeds = self.clip_image(perturbeds, self.lb, self.ub)
                        idx_perturbed = self.decision_function(perturbeds, label_or_target)

                        if np.sum(idx_perturbed) > 0:
                                # Select the perturbation that yields the minimum distance # after binary search.
                                perturbed, dist_post_update = self.binary_search_batch(input_xi, 
                                        perturbeds[idx_perturbed], label_or_target, theta)

                # compute new distance.
                dist = self.compute_distance(perturbed, input_xi)
                if self.verbose:
                        print('iteration: {:d}, {:s} distance {:.4f}'.format(j+1, self.p, dist))
                if self.query > self.max_queries:
                        break
                if dist < self.epsilon:
                        break

        return t(perturbed).unsqueeze(0)

    def decision_function(self, images, label):
        """
        Decision function output 1 on the desired side of the boundary,
        0 otherwise.
        """
        images = torch.from_numpy(images).float()
        if len(images.shape) == 3:
                images = images.unsqueeze(0)
        self.query += images.shape[0]
        la = self.predict_label(images).cpu().numpy()
        if self.targeted:
                return (la==label)
        else:
                return (la!=label)

    def clip_image(self, image, clip_min, clip_max):
        # Clip an image, or an image batch, with upper and lower threshold.
        return np.minimum(np.maximum(clip_min, image), clip_max) 


    def compute_distance(self, x_ori, x_pert):
        # Compute the distance between two images.
        if self.p == '2':
                return np.linalg.norm(x_ori - x_pert)
        elif self.p == 'inf':
                return np.max(abs(x_ori - x_pert))


    def approximate_gradient(self, sample, label_or_target, num_evals, delta):
        # Generate random vectors.
        noise_shape = [num_evals] + list(sample.shape)
        if self.p == '2':
                rv = np.random.randn(*noise_shape)
        elif self.p == 'inf':
                rv = np.random.uniform(low = -1, high = 1, size = noise_shape)

        rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1,2,3), keepdims = True))
        perturbed = sample + delta * rv
        perturbed = self.clip_image(perturbed, self.lb, self.ub)
        rv = (perturbed - sample) / delta

        # query the model.
        decisions = self.decision_function(perturbed, label_or_target)
        decision_shape = [len(decisions)] + [1] * len(sample.shape)
        fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        if np.mean(fval) == 1.0: # label changes. 
                gradf = np.mean(rv, axis = 0)
        elif np.mean(fval) == -1.0: # label not change.
                gradf = - np.mean(rv, axis = 0)
        else:
                fval -= np.mean(fval)
                gradf = np.mean(fval * rv, axis = 0) 

        # Get the gradient direction.
        gradf = gradf / np.linalg.norm(gradf)

        return gradf


    def project(self, original_image, perturbed_images, alphas):
        alphas_shape = [1] * len(original_image.shape)
        alphas = alphas.reshape(alphas_shape)
        if self.p == '2':
                #print(alphas.shape,original_image.shape, perturbed_images.shape)
                return (1-alphas) * original_image + alphas * perturbed_images
        elif self.p == 'inf':
                out_images = self.clip_image(
                        perturbed_images, 
                        original_image - alphas, 
                        original_image + alphas
                        )
                return out_images


    def binary_search_batch(self, original_image, perturbed_images, label_or_target, theta):
        """ Binary search to approach the boundary. """

        # Compute distance between each of perturbed image and original image.
        dists_post_update = np.array([
                        self.compute_distance(
                                original_image, 
                                perturbed_image 
                        ) 
                        for perturbed_image in perturbed_images])
        #print(dists_post_update)
        # Choose upper thresholds in binary searchs based on constraint.
        if self.p == 'inf':
                highs = dists_post_update
                # Stopping criteria.
                thresholds = np.minimum(dists_post_update * theta, theta)
        else:
                highs = np.ones(len(perturbed_images))
                thresholds = theta

        lows = np.zeros(len(perturbed_images))

        

        # Call recursive function. 
        while np.max((highs - lows) / thresholds) > 1:
                # projection to mids.
                mids = (highs + lows) / 2.0
                mid_images = self.project(original_image, perturbed_images, mids)
                # Update highs and lows based on model decisions.
                decisions = self.decision_function(mid_images, label_or_target)
                lows = np.where(decisions == 0, mids, lows)
                highs = np.where(decisions == 1, mids, highs)
                if self.query > self.max_queries:
                        break

        out_images = self.project(original_image, perturbed_images, highs)

        # Compute distance of the output image to select the best choice. 
        # (only used when stepsize_search is grid_search.)
        dists = np.array([
                self.compute_distance(
                        original_image, 
                        out_image 
                ) 
                for out_image in out_images])
        idx = np.argmin(dists)

        dist = dists_post_update[idx]
        out_image = out_images[idx]
        return out_image, dist


    def initialize(self, input_xi, label_or_target):
        """ 
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0
        # Find a misclassified random noise.
        while True:
                random_noise = np.random.uniform(self.lb, self.ub, size = input_xi.shape)
                success = self.decision_function(random_noise, label_or_target)[0]
                if success:
                        break
                if self.query > self.max_queries:
                        break
                assert num_evals < 1e4,"Initialization failed! "
                "Use a misclassified image as `target_image`" 

        # Binary search to minimize l2 distance to original image.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * input_xi + mid * random_noise 
                success = self.decision_function(blended, label_or_target)
                if success:
                        high = mid
                else:
                        low = mid
                if self.query > self.max_queries:
                        break

        initialization = (1 - high) * input_xi + high * random_noise 

        return initialization


    def geometric_progression_for_stepsize(self, x, label_or_target, update, dist, j):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching 
        the desired side of the boundary,
        """
        epsilon = dist / np.sqrt(j) 

        def phi(epsilon):
                new = x + epsilon * update
                success = self.decision_function(new, label_or_target)
                return success

        while not phi(epsilon):
                epsilon /= 2.0
                if self.query > self.max_queries:
                        break

        return epsilon


    def _perturb(self, xs_t, ys_t):
        xs = xs_t.cpu().numpy()
        ys = ys_t.cpu().numpy()
        adv = self.hsja(xs, ys)
        return t(adv), self.query

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
