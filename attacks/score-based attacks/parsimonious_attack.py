'''

This file is modified based on the following source:
link: https://github.com/machanic/MetaSimulator/tree/master/parsimonious_attack

@InProceedings{ma2021simulator,
    author    = {Ma, Chen and Chen, Li and Yong, Jun-Hai},
    title     = {Simulating Unknown Target Models for Query-Efficient Black-Box Attacks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {11835-11844}
}

The update include:
    1. data preprocess
    2. args and config
    3. functions modification
    
basic structure for main:
    1. config args, prior setup
    2. define some functions, including image processinf, noise setting, implementation of local search algorithm, and perturbation.
    3. return output
'''

"""
Implements Parsimonious attack
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch import Tensor as t
import heapq
import math
import itertools
import torch

from attacks.score.score_black_box_attack import ScoreBlackBoxAttack
from utils.compute import lp_step


class ParsimoniousAttack(ScoreBlackBoxAttack):
    """
    Parsimonious Attack
    """

    def __init__(self, max_loss_queries, epsilon, p, block_size, block_batch_size, EOT, lb, ub, batch_size, name):
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
                         ub=ub,
                         batch_size=batch_size,
                         name='ECO')
        self.block_sizeo = block_size
        self.batch_sizeo = block_batch_size
        self.no_hier = False
        self.max_iters = 1
        self.EOT = EOT
        
    def _split_block(self, upper_left, lower_right, block_size):
        """
        Split an image into a set of blocks.
        Note that a block consists of [upper_left, lower_right, channel]

        Args:
          upper_left: [x, y], the coordinate of the upper left of an image
          lower_right: [x, y], the coordinate of the lower right of an image
          block_size: int, the size of a block

        Return:
          blocks: list, a set of blocks
        """
        blocks = []
        xs = torch.arange(upper_left[0], lower_right[0], block_size)
        ys = torch.arange(upper_left[1], lower_right[1], block_size)
        for x, y in itertools.product(xs, ys):
            for c in range(3):
                blocks.append([[x, y], [x + block_size, y + block_size], c])
        return blocks

    def _perturb_image(self, image, noise):
        adv_image = image + noise
        adv_image = torch.clamp(adv_image, 0, self.ub)
        return adv_image


    def _flip_noise(self, noise, block):
        """Filp the sign of perturbation on a block.
            Args:
              noise: numpy array of size [1, 3, 32, 32], a noise
              block: [upper_left, lower_right, channel], a block

            Returns:
              noise_new: numpy array with size [1,3, 32, 32], an updated noise
        """
        noise_new = noise.clone()
        upper_left, lower_right, channel = block
        noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1],channel] *= -1
        return noise_new
    
    def local_search(self, image, noise, loss_fct, blocks):

        # Local variables
        priority_queue = []
        num_queries = 0

        # Check if a block is in the working set or not
        A = torch.zeros((len(blocks)), dtype=torch.int32)
        for i, block in enumerate(blocks):
            upper_left, _, channel = block
            x = upper_left[0]
            y = upper_left[1]
            # If the sign of perturbation on the block is positive,
            # which means the block is in the working set, then set A to 1
            if noise[0, x, y, channel] > 0:
                A[i] = 1
        # Calculate the current loss
        image_batch = self._perturb_image(image, noise)
        correct, losses = loss_fct(image_batch, es = True)
        num_queries += 1
        curr_loss = -losses[0]
        # Early stopping
        if torch.all(correct):
            return noise, num_queries*self.EOT, curr_loss, True
        # Main loop
        for _ in range(self.max_iters):
            # Lazy greedy insert
            indices = torch.nonzero(A==0).view(-1)
            batch_size =  100
            num_batches = int(math.ceil(indices.size(0) / batch_size))
            for ibatch in range(num_batches):
                bstart = ibatch * batch_size
                bend = min(bstart + batch_size, indices.size(0))
                image_batch = torch.zeros([bend - bstart,self.img_size,self.img_size,self.in_channels]).float()
                noise_batch = torch.zeros([bend - bstart,self.img_size,self.img_size,self.in_channels]).float()
                for i, idx in enumerate(indices[bstart:bend]):
                    idx = idx.item()
                    noise_batch[i:i + 1, ...] = self._flip_noise(noise, blocks[idx])
                    image_batch[i:i + 1, ...] = self._perturb_image(image, noise_batch[i:i + 1, ...])
                correct, losses = loss_fct(image_batch, es = True)
                # Early stopping
                success_indices = torch.nonzero(correct.long()).view(-1)
                if success_indices.size(0) > 0:
                    noise[0, ...] = noise_batch[success_indices[0], ...]
                    curr_loss = -losses[success_indices[0]]
                    num_queries += success_indices[0].item() + 1
                    return noise, num_queries*self.EOT, curr_loss, True

                num_queries += bend - bstart
                # Push into the priority queue
                for i in range(bend - bstart):
                    idx = indices[bstart + i]
                    margin = -losses[i] - curr_loss
                    heapq.heappush(priority_queue, (margin.item(), idx))
            # Pick the best element and insert it into the working set
            if len(priority_queue) > 0:
                best_margin, best_idx = heapq.heappop(priority_queue)
                curr_loss += best_margin
                noise = self._flip_noise(noise, blocks[best_idx])
                A[best_idx] = 1
            # Add elements into the working set
            while len(priority_queue) > 0:
                # Pick the best element
                cand_margin, cand_idx = heapq.heappop(priority_queue)
                # Re-evalulate the element
                image_batch = self._perturb_image(
                    image, self._flip_noise(noise, blocks[cand_idx]))
                correct, losses = loss_fct(image_batch, es = True)
                num_queries += 1
                margin = -losses[0] - curr_loss
                # If the cardinality has not changed, add the element
                if len(priority_queue) == 0 or margin.item() <= priority_queue[0][0]:
                    # If there is no element that has negative margin, then break
                    if margin.item() > 0:
                        break
                    # Update the noise
                    curr_loss = -losses[0]
                    noise = self._flip_noise(noise, blocks[cand_idx])
                    A[cand_idx] = 1
                    # Early stopping
                    if torch.all(correct):
                        return noise, num_queries*self.EOT, curr_loss, True
                # If the cardinality has changed, push the element into the priority queue
                else:
                    heapq.heappush(priority_queue, (margin.item(), cand_idx))
            priority_queue = []

            # Lazy greedy delete
            indices  = torch.nonzero(A == 1).view(-1)
            batch_size = 100
            num_batches = int(math.ceil(indices.size(0) / batch_size))
            for ibatch in range(num_batches):
                bstart = ibatch * batch_size
                bend = min(bstart + batch_size, indices.size(0))

                image_batch = torch.zeros([bend - bstart, self.img_size, self.img_size,self.in_channels]).float()
                noise_batch = torch.zeros([bend - bstart, self.img_size, self.img_size,self.in_channels]).float()
                for i, idx in enumerate(indices[bstart:bend]):
                    noise_batch[i:i + 1, ...] = self._flip_noise(noise, blocks[idx])
                    image_batch[i:i + 1, ...] = self._perturb_image(image, noise_batch[i:i+1, ...])
                correct, losses = loss_fct(image_batch, es = True)
                # Early stopping
                success_indices = torch.nonzero(correct.long()).view(-1)
                if success_indices.size(0) > 0:
                    noise[0, ...] = noise_batch[success_indices[0], ...]
                    curr_loss = -losses[success_indices[0]]
                    num_queries += success_indices[0].item() + 1
                    return noise, num_queries*self.EOT, curr_loss, True
                num_queries += bend - bstart
                # Push into the priority queue
                for i in range(bend - bstart):
                    idx = indices[bstart + i].item()
                    margin = -losses[i] - curr_loss
                    heapq.heappush(priority_queue, (margin.item(), idx))

            # Pick the best element and remove it from the working set
            if len(priority_queue) > 0:
                best_margin, best_idx = heapq.heappop(priority_queue)
                curr_loss += best_margin
                noise = self._flip_noise(noise, blocks[best_idx])
                A[best_idx] = 0
            # Delete elements from the working set
            while len(priority_queue) > 0:
                # pick the best element
                cand_margin, cand_idx = heapq.heappop(priority_queue)
                # Re-evalulate the element
                image_batch = self._perturb_image(image, self._flip_noise(noise, blocks[cand_idx]))
                correct, losses = loss_fct(image_batch, es = True)
                num_queries += 1
                margin = -losses[0] - curr_loss
                # If the cardinality has not changed, remove the element
                if len(priority_queue) == 0 or margin.item() <= priority_queue[0][0]:
                    # If there is no element that has negative margin, then break
                    if margin.item() > 0:
                        break
                    # Update the noise
                    curr_loss = -losses[0]
                    noise = self._flip_noise(noise, blocks[cand_idx])
                    A[cand_idx] = 0
                    # Early stopping
                    if torch.all(correct):
                        return noise, num_queries*self.EOT, curr_loss, True
                else:
                    heapq.heappush(priority_queue, (margin.item(), cand_idx))
                    
            priority_queue = []
        return noise, num_queries*self.EOT, curr_loss, False


    def _perturb(self, xs_t, loss_fct):

        self.img_size = xs_t.size(1)
        self.in_channels = xs_t.size(3)
        upper_left = [0, 0]
        lower_right = [self.img_size, self.img_size]
        num_queries = 0

        if self.is_new_batch:
            self.query = 0
            self.block_size = self.block_sizeo
            self.blocks = self._split_block(upper_left, lower_right, self.block_size)
            self.noise = -self.epsilon * torch.ones_like(xs_t).float()
            self.num_blocks = len(self.blocks)
            self.batch_size = self.batch_sizeo if self.batch_sizeo > 0 else self.num_blocks
            self.curr_order = torch.randperm(self.num_blocks)

        num_batches = int(math.ceil(self.num_blocks / self.batch_size))
        def loss_fct2(x, es):
            correct, loss = loss_fct(x, es = True)
            for _ in range(self.EOT - 1):
                _, new_loss = loss_fct(x, es = True)
                loss += new_loss
            return correct, loss
        
        for i in range(num_batches):
            # Pick a mini-batch
            bstart = i * self.batch_size
            bend = min(bstart + self.batch_size, self.num_blocks)
            blocks_batch = [self.blocks[self.curr_order[idx].item()]
                            for idx in range(bstart, bend)]
            self.noise, queries, loss, success = self.local_search(xs_t, self.noise, loss_fct2, blocks_batch)
            num_queries += queries
            self.query += queries
            print("Block size: {}, batch: {}, loss: {:.4f}, num queries: {}".format(self.block_size, i, -loss, self.query))
            # If query count exceeds the maximum queries, then return False
            if self.query > self.max_loss_queries:
                return torch.clamp(xs_t + self.noise, 0, self.ub), num_queries
            if success:
                return torch.clamp(xs_t + self.noise, 0, self.ub), num_queries

        # If block size is >= 2, then split the image into smaller blocks and reconstruct a batch
        if not self.no_hier and self.block_size >= 2:
            self.block_size //= 2
            self.blocks = self._split_block(upper_left, lower_right, self.block_size)
            self.num_blocks = len(self.blocks)
            self.batch_size = self.batch_sizeo if self.batch_sizeo > 0 else self.num_blocks
            self.curr_order = torch.randperm(self.num_blocks)
        # Otherwise, shuffle the order of the batch
        else:
            self.curr_order = torch.randperm(self.num_blocks)
        
        return torch.clamp(xs_t + self.noise, 0, self.ub), num_queries


    def _config(self):
        return {
            "name": self.name,
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_extra_queries": "inf" if np.isinf(self.max_extra_queries) else self.max_extra_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "attack_name": self.__class__.__name__
        }
