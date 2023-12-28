"""
A wrapper for datasets
    mnist, cifar10, imagenet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.examples.tutorials.mnist import input_data

import datasets.cifar10 as cifar10_input
from datasets.imagenet import ImagenetValidData
from utils.misc import data_path_join
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_imagenet(n_ex = 1000, size=224):
    IMAGENET_SL = size
    IMAGENET_PATH = data_path_join('imagenet/Sample_1000')
    imagenet = ImageFolder(IMAGENET_PATH,
                           transforms.Compose([
                               transforms.Resize(IMAGENET_SL),
                               transforms.CenterCrop(IMAGENET_SL),
                               transforms.ToTensor()
                           ]))
    torch.manual_seed(0)

    imagenet_loader = DataLoader(imagenet, batch_size=n_ex, shuffle=True, num_workers=1)
    x_test, y_test = next(iter(imagenet_loader))
    return np.array(x_test, dtype=np.float32), np.array(y_test)

class Dataset(object):
    def __init__(self, name, config):
        """
        :param name: dataset name
        :param config: dictionary whose keys are dependent on the dataset created
         (see source code below)
        """
        assert name in ['mnist', 'cifar10', 'cifar10aug', 'imagenet', 'imagenet_sub'], "Invalid dataset"
        self.name = name
        model_name = config['modeln']
        # config = config['dset_config']

        if self.name == 'cifar10':
            data_path = data_path_join('/home/DATA/ITWM/cifar10')
            self.data = cifar10_input.CIFAR10Data(data_path)
        elif self.name == 'cifar10aug':
            data_path = data_path_join('cifar10_data')
            # raw_cifar = cifar10_input.CIFAR10Data(data_path)
            # sess = config['sess']
            # model = config['model']
            # self.data = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)
        elif self.name == 'mnist':
            self.data = input_data.read_data_sets(data_path_join('mnist_data'), one_hot=False)
        elif self.name == 'imagenet':
            data_path = data_path_join('imagenet_data')
            self.data = ImagenetValidData(data_dir=data_path)
        elif self.name == 'imagenet_sub':
            if model_name == 'Inception':
                self.x_test, self.y_test = load_imagenet(n_ex=1000, size=299)
            else:
                self.x_test, self.y_test = load_imagenet()
            self.x_test = self.x_test.transpose(0, 2, 3, 1)

    def get_next_train_batch(self, batch_size):
        """
        Returns a tuple of (x_batch, y_batch)
        """
        if self.name in ['cifar10', 'cifar10aug']:
            return self.data.train_data.get_next_batch(batch_size, multiple_passes=True)
        elif self.name == 'mnist':
            return self.data.train.next_batch(batch_size)
        elif self.name in ['imagenet', 'imagenet_sub']:
            raise Exception(
                'No training data for imagenet is needed (provided), the models are assumed to be pretrained!')

    def get_eval_data(self, bstart, bend):
        """
        :param bstart: batch start index
        :param bend: batch end index
        """
        if self.name in ['cifar10', 'cifar10aug']:
            return self.data.eval_data.xs[bstart:bend, :], \
                   self.data.eval_data.ys[bstart:bend]
        elif self.name == 'mnist':
            return self.data.test.images[bstart:bend, :], \
                   self.data.test.labels[bstart:bend]
        elif self.name == 'imagenet':
            return self.data.get_eval_data(bstart, bend)
        elif self.name == 'imagenet_sub':
            return self.x_test[bstart:bend, :], \
                    self.y_test[bstart:bend]


    @property
    def min_value(self):
        if self.name in ['cifar10', 'cifar10aug', 'mnist', 'imagenet', 'imagenet_sub']:
            return 0.

    @property
    def max_value(self):
        if self.name in ['cifar10', 'cifar10aug']:
            return 255.
        if self.name in ['mnist', 'imagenet', 'imagenet_sub']:
            return 1.
