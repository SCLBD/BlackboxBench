"""
A wrapper for imagenet validation set, this is a simple loader
with the appropriate transforms, it does not support shuffling, nor batching
"""
import random

import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

IMAGENET_SL = 299


# put your imagenet validation path here
# where the directory is
# label_1/
#     ILS....JPEG
#     ILS....JPEG
# label_2/
#     ILS....JPEG
#     ILS....JPEG
#
# The labels are indexed based on 1001 classes, where 0 is the "I dont know" label


class ImagenetValidData():
    def __init__(self, data_dir="/storage/imagenet/tf_val_set"):
        imgnet_transform = transforms.Compose([
            transforms.Resize(IMAGENET_SL),
            transforms.CenterCrop(IMAGENET_SL),
            transforms.Lambda(lambda _: np.array(_) / 255.)])
        self.dset = ImageFolder(root=data_dir, transform=imgnet_transform)
        # shuffle the dataset
        self.idxs = list(range(len(self.dset)))
        random.Random(1).shuffle(self.idxs)

    def get_eval_data(self, bstart, bend):
        images, labels = zip(*[self.dset[self.idxs[i]] for i in range(bstart, bend)])

        return np.array(images), np.array(labels) + 1  # to consider "IDK" label
