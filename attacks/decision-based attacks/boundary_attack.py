'''

This file is modified based on the following source:
link: https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/boundary_attack.py

The original license is placed at the end of this file.

@article{rauber2017foolboxnative,
  doi = {10.21105/joss.02607},
  url = {https://doi.org/10.21105/joss.02607},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {53},
  pages = {2607},
  author = {Jonas Rauber and Roland Zimmermann and Matthias Bethge and Wieland Brendel},
  title = {Foolbox Native: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX},
  journal = {Journal of Open Source Software}
}

@inproceedings{rauber2017foolbox,
  title={Foolbox: A Python toolbox to benchmark the robustness of machine learning models},
  author={Rauber, Jonas and Brendel, Wieland and Bethge, Matthias},
  booktitle={Reliable Machine Learning in the Wild Workshop, 34th International Conference on Machine Learning},
  year={2017},
  url={http://arxiv.org/abs/1707.04131},
}

The update include:
  1. model setting
  2. args and config
  3. functions modification
  
basic structure for main:
  1. config args and prior setup
  2. define functions that produce candidates and spherical candidates and implement boundary attack algorithm
  3. insert perturbation
  4. return results
  
'''

"""
Implements Boundary Attack
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

class BoundaryAttack(DecisionBlackBoxAttack):
    """
    Boundary Attack
    """
    def __init__(self, epsilon, p, max_queries, lb, ub, batch_size, steps, spherical_step, source_step, source_step_convergance, step_adaptation, update_stats_every_k):
        super().__init__(max_queries = max_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size = batch_size)
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.source_step_convergance = source_step_convergance
        self.step_adaptation = step_adaptation
        self.update_stats_every_k = update_stats_every_k
        self.query = 0


    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "attack_name": self.__class__.__name__
        }

    def atleast_kd(self, x, k):
        shape = x.shape + (1,) * (k - x.ndim)
        return x.reshape(shape)
    
    def flatten(self, x, keep = 1):
        return x.flatten(start_dim=keep)

    def draw_proposals(
        self,
        originals,
        perturbed,
        unnormalized_source_directions,
        source_directions,
        source_norms,
        spherical_steps,
        source_steps,
    ):
        # remember the actual shape
        shape = originals.shape
        assert perturbed.shape == shape
        assert unnormalized_source_directions.shape == shape
        assert source_directions.shape == shape

        # flatten everything to (batch, size)
        originals = self.flatten(originals)
        perturbed = self.flatten(perturbed)
        unnormalized_source_directions = self.flatten(unnormalized_source_directions)
        source_directions = self.flatten(source_directions)
        N, D = originals.shape

        assert source_norms.shape == (N,)
        assert spherical_steps.shape == (N,)
        assert source_steps.shape == (N,)

        # draw from an iid Gaussian (we can share this across the whole batch)
        eta = torch.normal(mean=torch.ones(D, 1))

        # make orthogonal (source_directions are normalized)
        eta = eta.T - torch.matmul(source_directions, eta) * source_directions
        assert eta.shape == (N, D)

        # rescale
        norms = torch.norm(eta, dim=-1, p=2)
        assert norms.shape == (N,)
        eta = eta * self.atleast_kd(spherical_steps * source_norms / norms, eta.ndim)

        # project on the sphere using Pythagoras
        distances = self.atleast_kd((spherical_steps ** 2 + 1).sqrt(), eta.ndim)
        directions = eta - unnormalized_source_directions
        spherical_candidates = originals + directions / distances

        # clip
        min_, max_ = self.lb, self.ub
        spherical_candidates = spherical_candidates.clamp(min_, max_)

        # step towards the original inputs
        new_source_directions = originals - spherical_candidates
        assert new_source_directions.ndim == 2
        new_source_directions_norms = torch.norm(self.flatten(new_source_directions), dim=-1, p=2)

        # length if spherical_candidates would be exactly on the sphere
        lengths = source_steps * source_norms

        # length including correction for numerical deviation from sphere
        lengths = lengths + new_source_directions_norms - source_norms

        # make sure the step size is positive
        lengths = torch.max(lengths, torch.zeros_like(lengths))

        # normalize the length
        lengths = lengths / new_source_directions_norms
        lengths = self.atleast_kd(lengths, new_source_directions.ndim)

        candidates = spherical_candidates + lengths * new_source_directions

        # clip
        candidates = candidates.clamp(min_, max_)

        # restore shape
        candidates = candidates.reshape(shape)
        spherical_candidates = spherical_candidates.reshape(shape)
        return candidates, spherical_candidates
    
    def initialize(self, input_xi, label_or_target):
        """ 
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0
        # Find a misclassified random noise.
        while True:
                random_noise = t(np.random.uniform(self.lb, self.ub, size = input_xi.shape))
                success = self.is_adversarial(random_noise, label_or_target)[0]
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
                success = self.is_adversarial(blended, label_or_target)
                if success:
                        high = mid
                else:
                        low = mid
                if self.query > self.max_queries:
                        break

        initialization = (1 - high) * input_xi + high * random_noise 

        return initialization

    def boundary_attack(self, target_sample, y):
        self.query = 0

        best_advs = self.initialize(target_sample, y)
        shape = list(target_sample.shape)
        N = shape[0]
        ndim = target_sample.ndim
        spherical_steps = torch.ones(N) * self.spherical_step
        source_steps = torch.ones(N) * self.source_step

        stats_spherical_adversarial = ArrayQueue(maxlen=100, N=N)
        stats_step_adversarial = ArrayQueue(maxlen=30, N=N)


        for step in range(1, self.steps + 1):
            converged = source_steps < self.source_step_convergance
            if converged.all():
                break  # pragma: no cover
            converged = self.atleast_kd(converged, ndim)


            unnormalized_source_directions = target_sample - best_advs
            source_norms = torch.norm(self.flatten(unnormalized_source_directions), dim = -1, p = 2)
            source_directions = unnormalized_source_directions / self.atleast_kd(
                source_norms, ndim
            )

            # only check spherical candidates every k steps
            check_spherical_and_update_stats = step % self.update_stats_every_k == 0

            candidates, spherical_candidates = self.draw_proposals(
                target_sample,
                best_advs,
                unnormalized_source_directions,
                source_directions,
                source_norms,
                spherical_steps,
                source_steps,
            )


            is_adv = self.is_adversarial(candidates, y)
            self.query += N

            if check_spherical_and_update_stats:
                spherical_is_adv = self.is_adversarial(spherical_candidates, y)
                self.query += N
                stats_spherical_adversarial.append(spherical_is_adv)
                stats_step_adversarial.append(is_adv)
            else:
                spherical_is_adv = None

            # in theory, we are closer per construction
            # but limited numerical precision might break this
            distances = torch.norm(self.flatten(target_sample - candidates), dim=-1, p=2)
            closer = distances < source_norms
            is_best_adv = is_adv & closer
            is_best_adv = self.atleast_kd(is_best_adv, ndim)

            cond = (~converged)&(is_best_adv)
            best_advs = torch.where(cond, candidates, best_advs)

            if self.query > self.max_queries:
                break

            diff = self.distance(best_advs, target_sample)
            if diff <= self.epsilon:
                print("{} steps".format(self.query))
                print("Mean Squared Error: {}".format(diff))
                break
            if is_best_adv:
                print("Mean Squared Error: {}".format(diff))
                print("Calls: {}".format(self.query))


            if check_spherical_and_update_stats:
                full = stats_spherical_adversarial.isfull().cuda()
                if full.any():
                    probs = stats_spherical_adversarial.mean().cuda()
                    cond1 = (probs > 0.5) & full
                    spherical_steps = torch.where(
                        cond1, spherical_steps * self.step_adaptation, spherical_steps
                    )
                    source_steps = torch.where(
                        cond1, source_steps * self.step_adaptation, source_steps
                    )
                    cond2 = (probs < 0.2) & full
                    spherical_steps = torch.where(
                        cond2, spherical_steps / self.step_adaptation, spherical_steps
                    )
                    source_steps = torch.where(
                        cond2, source_steps / self.step_adaptation, source_steps
                    )
                    stats_spherical_adversarial.clear(cond1 | cond2)


                full = stats_step_adversarial.isfull().cuda()
                if full.any():
                    probs = stats_step_adversarial.mean().cuda()
                    cond1 = (probs > 0.25) & full
                    source_steps = torch.where(
                        cond1, source_steps * self.step_adaptation, source_steps
                    )
                    cond2 = (probs < 0.1) & full
                    source_steps = torch.where(
                        cond2, source_steps / self.step_adaptation, source_steps
                    )
                    stats_step_adversarial.clear(cond1 | cond2)
        return best_advs, self.query

    def _perturb(self, xs_t, ys_t):
        adv, q = self.boundary_attack(xs_t, ys_t)
        return adv, q

class ArrayQueue:
    def __init__(self, maxlen: int, N: int):
        # we use NaN as an indicator for missing data
        self.data = np.full((maxlen, N), np.nan)
        self.next = 0
        # used to infer the correct framework because this class uses NumPy
        self.tensor = None

    @property
    def maxlen(self) -> int:
        return int(self.data.shape[0])

    @property
    def N(self) -> int:
        return int(self.data.shape[1])

    def append(self, x) -> None:
        if self.tensor is None:
            self.tensor = x
        x = x.cpu().numpy()
        assert x.shape == (self.N,)
        self.data[self.next] = x
        self.next = (self.next + 1) % self.maxlen

    def clear(self, dims) -> None:
        if self.tensor is None:
            self.tensor = dims  # pragma: no cover
        dims = dims.cpu().numpy()
        assert dims.shape == (self.N,)
        assert dims.dtype == np.bool
        self.data[:, dims] = np.nan

    def mean(self):
        assert self.tensor is not None
        result = np.nanmean(self.data, axis=0)
        return torch.from_numpy(result)

    def isfull(self):
        assert self.tensor is not None
        result = ~np.isnan(self.data).any(axis=0)
        return torch.from_numpy(result)
'''

Original License

MIT License

Copyright (c) 2020 Jonas Rauber et al.

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
