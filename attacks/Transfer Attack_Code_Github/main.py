import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.autograd as autograd
import torchvision
import torchvision.datasets as td
import torch.distributions as tdist
import argparse
from torchvision import models, transforms

from PIL import Image
import csv
import numpy as np
import os
import scipy.stats as st
from tqdm import tqdm

from utils import *
from flags import parse_handle

#Set random seed for reproduce
torch.manual_seed(42)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

#Parsing input parameters
parser = parse_handle()
arg = parser.parse_args()
arg.adv_alpha = arg.adv_epsilon / arg.adv_steps
print(arg)

model_1 = models.inception_v3(pretrained=True, transform_input=True).eval()
model_2 = models.resnet50(pretrained=True).eval()
model_3 = models.densenet121(pretrained=True).eval()
model_4 = models.vgg16_bn(pretrained=True).eval()


for param in model_1.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)

if arg.source_model == 'inception-v3':
    model_source = model_1
elif arg.source_model == 'resnet50':
    model_source = model_2
elif arg.source_model == 'densenet121':
    model_source = model_3
elif arg.source_model == 'vgg16bn':
    model_source = model_4



# values are standard normalization for ImageNet images,
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])
image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')

img_size = 299
input_path = './dataset/images/'
lr = 2 / 255  # step size
epsilon = 16  # L_inf norm bound
num_batches = np.int(np.ceil(len(image_id_list) / arg.batch_size)) 

n = tdist.Normal(0.0, 15/255)

adv_activate = 0
noise_activate = 0 
pos = np.zeros((4, arg.max_iterations)) 

for k in tqdm(range(0, num_batches)):
    batch_size_cur = min(arg.batch_size, len(image_id_list) - k * arg.batch_size) 
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device) 
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * arg.batch_size + i] + '.png'))
    labels = torch.tensor(label_ori_list[k * arg.batch_size:k * arg.batch_size + batch_size_cur]).to(device)
    target_labels = torch.tensor(label_tar_list[k * arg.batch_size:k * arg.batch_size + batch_size_cur]).to(device)
    grad_pre = 0
    prev = float('inf')

    for t in range(arg.max_iterations): 
        if t < arg.transpoint:
            adv_activate = 0
            noise_activate = 0
        else:
            if arg.adv_perturbation: 
                adv_activate = 1
            else:
                adv_activate = 0
            if arg.gaussian_noise:  
                noise_activate = 1  
            else:
                noise_activate = 0
        grad_list = []
        for q in range(arg.m1): 
            delta.requires_grad_(False) 
            if arg.strength == 0:  #adMix
                X_addin = torch.zeros_like(X_ori).to(device)
            else: 
                #add X_addin
                X_addin = torch.zeros_like(X_ori).to(device)
                random_labels = torch.zeros(batch_size_cur).to(device) 
                stop = False
                while stop == False:
                    random_indices = np.random.randint(0, 1000, batch_size_cur) 
                    for i in range(batch_size_cur): 
                        X_addin[i] = trn(Image.open(input_path + image_id_list[random_indices[i]] + '.png'))
                        random_labels[i] = label_ori_list[random_indices[i]] 
                    if torch.sum(random_labels==labels).item() == 0: 
                        stop = True
                X_addin = arg.strength * X_addin 
                X_addin = torch.clamp(X_ori+delta+X_addin, min=0, max=1) - (X_ori+delta)
            
            #add gaussian noise
            if noise_activate: 
                gaussian_noise = n.sample((X_ori.shape)).to(device) 
                gaussian_noise = torch.clamp(X_ori+delta+X_addin+gaussian_noise, min=0, max=1) - (X_ori+delta+X_addin) 
            else:
                gaussian_noise = torch.zeros_like(X_ori).to(device) 

            #add adversarial perturbation
            if adv_activate:   
                top_values_1, top_indices_1 = model_source(norm(X_ori+delta+X_addin)).topk(arg.m1+1, dim=1, largest=True, sorted=True) 
                if arg.adv_targeted: 
                    if arg.adv_label == 'pred':
                        label_pred = top_indices_1[:, q+1] 
                    else:
                        label_pred = labels 
                else:
                    if arg.adv_label == 'pred':
                        label_pred = top_indices_1[:, 0]
                    elif arg.adv_label == 'target':
                        label_pred = target_labels
                X_advaug = pgd(model_source, X_ori+delta+X_addin, label_pred, arg.adv_targeted, arg.adv_epsilon, arg.adv_steps, arg.adv_alpha)
                X_aug = X_advaug - (X_ori+delta+X_addin)
            else:
                X_aug = torch.zeros_like(X_ori).to(device)
           
            delta.requires_grad_(True) 

            for j in range(arg.m2): 
                if arg.DI:  # DI
                    if arg.NI:
                        if arg.SI:
                            logits = model_source(norm(DI((X_ori + delta + X_addin + X_aug + gaussian_noise - lr * grad_pre)/2**j))) 
                        else: 
                            logits = model_source(norm(DI(X_ori + delta + X_addin + X_aug + gaussian_noise - lr * grad_pre)))
                    else: 
                        if arg.SI: 
                            logits = model_source(norm(DI((X_ori + delta + X_addin + X_aug + gaussian_noise)/2**j)))
                        else: 
                            logits = model_source(norm(DI(X_ori + delta + X_addin + X_aug + gaussian_noise)))
                else:
                    if arg.NI: 
                        if arg.SI: 
                            logits = model_source(norm((X_ori + delta + X_addin + X_aug + gaussian_noise - lr * grad_pre)/2**j))
                        else: 
                            logits = model_source(norm(X_ori + delta + X_addin + X_aug + gaussian_noise - lr * grad_pre))
                    else: 
                        if arg.SI: 
                             logits = model_source(norm((X_ori + delta + X_addin + X_aug + gaussian_noise)/2**j))
                        else: 
                            logits = model_source(norm(X_ori + delta + X_addin + X_aug + gaussian_noise))
        
                if arg.loss_function == 'CE': 
                    loss_func = nn.CrossEntropyLoss(reduction='sum') 
                    if arg.targeted:
                        loss = loss_func(logits, target_labels)
                    else:
                        loss = -1 * loss_func(logits, labels)
                elif arg.loss_function == 'MaxLogit': 
                    if arg.targeted:
                        real = logits.gather(1,target_labels.unsqueeze(1)).squeeze(1)
                        loss = -1 * real.sum()
                    else:
                        real = logits.gather(1,labels.unsqueeze(1)).squeeze(1)
                        loss = real.sum()
                loss.backward()
                grad_cc = delta.grad.clone().to(device) 
                if arg.TI:  # TI
                    gaussian_kernel = gkern(kernel_size=5, nsig=3) 
                    grad_cc = functional.conv2d(grad_cc, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3) 
                grad_list.append(grad_cc) 
                delta.grad.zero_() 
        grad_c = 0
        for j in range(arg.m1 * arg.m2):
            grad_c += grad_list[j]
        grad_c = grad_c / (arg.m1 * arg.m2) 
        if arg.MI or arg.NI:  # MI
            grad_c = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
        grad_pre = grad_c
        delta.data = delta.data - lr * torch.sign(grad_c) 
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori 

        if arg.targeted:
            pos[0, t] = pos[0, t] + sum(torch.argmax(model_1(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
            pos[1, t] = pos[1, t] + sum(torch.argmax(model_2(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
            pos[2, t] = pos[2, t] + sum(torch.argmax(model_3(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
            pos[3, t] = pos[3, t] + sum(torch.argmax(model_4(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
        else:
            pos[0, t] = pos[0, t] + sum(torch.argmax(model_1(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()
            pos[1, t] = pos[1, t] + sum(torch.argmax(model_2(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()
            pos[2, t] = pos[2, t] + sum(torch.argmax(model_3(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()
            pos[3, t] = pos[3, t] + sum(torch.argmax(model_4(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()


torch.cuda.empty_cache()
print(f'{arg.source_model} --> inception-v3 | resnet50 | densenet121 | vgg16bn')
tmp_pos = np.zeros((4, arg.max_iterations // 10))
for i in range(arg.max_iterations):
    if i % 10 == 9:
        tmp_pos[:, i // 10] = pos[:, i]
print(tmp_pos)

