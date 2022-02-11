import torch
import torch.nn as nn
import numpy as np
import os
import json
from PIL import Image
import csv

##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',') 
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1) 
    return image_id_list, label_ori_list, label_tar_list
## simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std): 
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

##define TI
def gkern(kernel_size=5, nsig=3):
    #calculate the kernel
    x = np.linspace(-nsig, nisg, kernel_size)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    #calculate the gaussian kernel
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
    return gaussian_kernel

##define DI
def DI(X_in):
    rnd = np.random.randint(299, 330, size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        X_out = F.pad(F.interpolate(X_in, size=(rnd, rnd)), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return X_out
    else:
        return X_in

def pgd(model, data, labels, targeted, epsilon, k, a, random_start=True):
    
    data_max = data + epsilon
    data_min = data - epsilon
    data_max.clamp_(0, 1)
    data_min.clamp_(0, 1)

    data = data.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    perturbed_data = data.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        perturbed_data = perturbed_data + torch.empty_like(perturbed_data).uniform_(-epsilon, epsilon)
        perturbed_data = torch.clamp(perturbed_data, min=0, max=1).detach()

    for _ in range(k):
        perturbed_data.requires_grad = True
        outputs = model(norm(perturbed_data))
        if arg.adv_loss_function == 'CE':
            loss = nn.CrossEntropyLoss(reduction='sum')
            if targeted:
                cost = loss(outputs, labels)
            else:
                cost = -1 * loss(outputs, labels)
        elif arg.adv_loss_function == 'MaxLogit':
            if targeted:
                real = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
                cost = -1 * real.sum()
            else:
                real = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
                cost = real.sum()
        # Update adversarial images
        cost.backward()

        gradient = perturbed_data.grad.clone().to(device)
        perturbed_data.grad.zero_()
        with torch.no_grad():
            perturbed_data.data -= a * torch.sign(gradient)
            perturbed_data.data = torch.max(torch.min(perturbed_data.data, data_max), data_min)
    return perturbed_data.detach()











