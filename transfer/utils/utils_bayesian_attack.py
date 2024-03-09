import argparse
import copy
import math
import logging
import os
from collections import OrderedDict
from utils.registry import Registry
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.backends import cudnn
from torch.utils.data import DataLoader
from surrogate_model.utils import guess_and_load_model, list_path
from utils.helper import makedir

args = Registry._GLOBAL_REGISTRY['args']

class RandomResizedCrop(T.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = torchvision.transforms.functional.get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w


def assign_grad(model, grad_dict):
    names_in_grad_dict = grad_dict.keys()
    for name, param in model.named_parameters():
        if name in names_in_grad_dict:
            if param.grad != None:
                param.grad.data.mul_(0).add_(grad_dict[name])
            else:
                param.grad = grad_dict[name]


def get_grad(model):
    grad_dict = OrderedDict()
    for name, param in model.named_parameters():
        grad_dict[name] = param.grad.data+0
    return grad_dict


def update_bn_imgnet(loader, model, device=None):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
    if not momenta:
        return
    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0
    for i, input in enumerate(loader):
        # using 10% of the training data to update batch-normalization statistics
        if i > len(loader)//10:
            break
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)
        with torch.no_grad():
            model(input)
    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def update_swag_model(model, mean_model, sqmean_model, n):
    for param, param_mean, param_sqmean in zip(model.parameters(), mean_model.parameters(), sqmean_model.parameters()):
        param_mean.data.mul_(n / (n+1.)).add_(param, alpha=1./(n+1.))
        param_sqmean.data.mul_(n / (n+1.)).add_(param**2, alpha=1./(n+1.))


def eval_imgnet(val_loader, model, device):
    loss_eval = 0
    # grad_norm_eval = 0
    acc_eval = 0
    for i, (img, label) in enumerate(val_loader):
        img, label = img.to(device), label.to(device)
        model.eval()
        with torch.no_grad():
            output = model(img).logits if 'inception_v3' in args.source_model_path[0] \
                                        and 'NIPS2017' in args.source_model_path[0] \
                                        and model.training == True \
                                        else model(img)
        loss = F.cross_entropy(output, label)
        acc = 100*(output.argmax(1) == label).sum() / len(img)
        loss_eval += loss.item()
        acc_eval += acc
        if i == 4:
            loss_eval/=(i+1)
            acc_eval/=(i+1)
            break
    # loss_eval/=(i+1)
    # acc_eval/=(i+1)
    return loss_eval, acc_eval


def add_into_weights(model, grad_on_weights, gamma):
    names_in_gow = grad_on_weights.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_gow:
                param.add_(gamma * grad_on_weights[name])


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def cat_grad(grad_dict):
    dls = []
    for name, d in grad_dict.items():
        dls.append(d)
    return _concat(dls)


def finetune_imagenet(pretrain_ckpt, batch_size, lr, wd, momentum, epochs, lam, gamma,
                      swa_start, swa_n, seed, data_path, save_dir):
    """

    :param arch:
    :param batch_size:
    :param lam:
    :param lr:
    :param epochs:
    :param swa_start:
    :param swa_n: update swag model every `swa_n` iterations.
    :param seed:
    :param data_path:
    :param save_dir: path to save checkpoints(regular model and swag model) during the finetuning process.
    :return: mean model and mean square model
    """
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    makedir(save_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if 'inception_v3' in pretrain_ckpt:
        transform_train = T.Compose([
            RandomResizedCrop(224, interpolation=3),
            T.Resize(size=299),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = T.Compose([
            T.Resize(256, interpolation=3),
            T.CenterCrop(224),
            T.Resize(size=299),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform_train = T.Compose([
            RandomResizedCrop(224, interpolation=3),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = T.Compose([
            T.Resize(256, interpolation=3),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = torchvision.datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform_train)
    dataset_val = torchvision.datasets.ImageFolder(os.path.join(data_path, 'val'), transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    model = guess_and_load_model(list_path([pretrain_ckpt])[0], norm_layer=False, parallel=False, require_grad=True, load_as_ghost=False)
    model = nn.DataParallel(model)

    print("SWAG training ...")
    mean_model = copy.deepcopy(model)
    sqmean_model = copy.deepcopy(model)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    n_collected = 0
    n_ensembled = 0
    for epoch in range(epochs):
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            model.train()

            output_cln = model(img).logits if 'inception_v3' in pretrain_ckpt \
                                               and 'NIPS2017' in pretrain_ckpt \
                                               and model.training == True \
                                               else model(img)

            loss_normal = F.cross_entropy(output_cln, label)
            optimizer.zero_grad()
            loss_normal.backward()
            grad_normal = get_grad(model)   # 返回模型参数的梯度的字典
            norm_grad_normal = cat_grad(grad_normal).norm()    # 将所有参数的梯度拼接起来 求模

            add_into_weights(model, grad_normal, gamma=+gamma / (norm_grad_normal + 1e-20))    # 给model的参数加上gamma扰动
            loss_add = F.cross_entropy(model(img).logits if 'inception_v3' in pretrain_ckpt \
                                               and 'NIPS2017' in pretrain_ckpt \
                                               and model.training == True \
                                               else model(img), label)
            optimizer.zero_grad()
            loss_add.backward()
            grad_add = get_grad(model)    # 返回扰动后模型参数的梯度
            add_into_weights(model, grad_normal, gamma=-gamma / (norm_grad_normal + 1e-20))    # 给model的参数减去gamma扰动

            optimizer.zero_grad()
            grad_new_dict = OrderedDict()
            for (name, g_normal), (_, g_add) in zip(grad_normal.items(), grad_add.items()):
                grad_new_dict[name] = g_normal + (lam / 0.1) * (g_add - g_normal)   # outer gradient的解
            assign_grad(model, grad_new_dict)    # 将grad_new_dict赋给model参数的梯度
            optimizer.step()

            if i % 100 == 0:
                acc = 100 * (output_cln.argmax(1) == label).sum() / len(img)
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc:{:.2f}'.format(
                    epoch, i * len(img), len(train_loader.dataset),
                           100. * i / len(train_loader), loss_normal.item(), acc))

            # SWAG
            if ((epoch + 1) > swa_start
                    and ((epoch - swa_start) * len(train_loader) + i) % (((epochs - swa_start) * len(train_loader)) // swa_n) == 0):
                update_swag_model(model, mean_model, sqmean_model, n_ensembled)    # 更新参数的均值和参数平方的均值
                n_ensembled += 1

        loss_cln_eval, acc_eval = eval_imgnet(val_loader, model, device)
        logging.info('CURRENT EVAL Loss: {:.6f}\tAcc:{:.2f}'.format(loss_cln_eval, acc_eval))
        print("updating BN statistics ... ")
        update_bn_imgnet(train_loader, mean_model)
        loss_cln_eval, acc_eval = eval_imgnet(val_loader, mean_model, device)
        logging.info('SWA EVAL Loss: {:.6f}\tAcc:{:.2f}'.format(loss_cln_eval, acc_eval))

        torch.save({"state_dict": model.state_dict(),
                    "opt_state_dict": optimizer.state_dict(),
                    "epoch": epoch},
                   os.path.join(save_dir, 'ep_{}.pt'.format(epoch)))
        torch.save({"mean_state_dict": mean_model.state_dict(),
                    "sqmean_state_dict": sqmean_model.state_dict(),
                    "epoch": epoch},
                   os.path.join(save_dir, 'swag_ep_{}.pt'.format(epoch)))

    return mean_model, sqmean_model

def finetune_cifar10(pretrain_ckpt, batch_size, lr, wd, momentum, epochs, lam, gamma,
                      swa_start, swa_n, seed, data_path, save_dir):
    # logging.info('Save dir: '+save_dir)
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    makedir(save_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)

    model = guess_and_load_model(list_path([pretrain_ckpt])[0], norm_layer=False, parallel=False, require_grad=True, load_as_ghost=False)
    model = nn.DataParallel(model)
    model = model.cuda()

    print("SWAG training ...")
    mean_model = copy.deepcopy(model)
    sqmean_model = copy.deepcopy(model)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    n_collected = 0
    n_ensembled = 0
    for epoch in range(epochs):
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            model.train()

            output_cln = model(img)
            loss_normal = F.cross_entropy(output_cln, label)
            optimizer.zero_grad()
            loss_normal.backward()
            grad_normal = get_grad(model)   # 返回模型参数的梯度的字典
            norm_grad_normal = cat_grad(grad_normal).norm()    # 将所有参数的梯度拼接起来 求模

            add_into_weights(model, grad_normal, gamma=+gamma / (norm_grad_normal + 1e-20))    # 给model的参数加上gamma扰动
            loss_add = F.cross_entropy(model(img), label)
            optimizer.zero_grad()
            loss_add.backward()
            grad_add = get_grad(model)    # 返回扰动后模型参数的梯度
            add_into_weights(model, grad_normal, gamma=-gamma / (norm_grad_normal + 1e-20))    # 给model的参数减去gamma扰动

            optimizer.zero_grad()
            grad_new_dict = OrderedDict()
            for (name, g_normal), (_, g_add) in zip(grad_normal.items(), grad_add.items()):
                grad_new_dict[name] = g_normal + (lam / 0.1) * (g_add - g_normal)   # outer gradient的解
            assign_grad(model, grad_new_dict)    # 将grad_new_dict赋给model参数的梯度
            optimizer.step()

            if i % 100 == 0:
                acc = 100 * (output_cln.argmax(1) == label).sum() / len(img)
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc:{:.2f}'.format(
                    epoch, i * len(img), len(train_loader.dataset),
                           100. * i / len(train_loader), loss_normal.item(), acc))

            # SWAG
            if ((epoch + 1) > swa_start
                    and ((epoch - swa_start) * len(train_loader) + i) % (((epochs - swa_start) * len(train_loader)) // swa_n) == 0):
                update_swag_model(model, mean_model, sqmean_model, n_ensembled)    # 更新参数的均值和参数平方的均值
                n_ensembled += 1

        loss_cln_eval, acc_eval = eval_imgnet(val_loader, model, device)
        logging.info('CURRENT EVAL Loss: {:.6f}\tAcc:{:.2f}'.format(loss_cln_eval, acc_eval))
        print("updating BN statistics ... ")
        update_bn_imgnet(train_loader, mean_model)
        loss_cln_eval, acc_eval = eval_imgnet(val_loader, mean_model, device)
        logging.info('SWA EVAL Loss: {:.6f}\tAcc:{:.2f}'.format(loss_cln_eval, acc_eval))

        torch.save({"state_dict": model.state_dict(),
                    "opt_state_dict": optimizer.state_dict(),
                    "epoch": epoch},
                   os.path.join(save_dir, 'ep_{}.pt'.format(epoch)))
        torch.save({"mean_state_dict": mean_model.state_dict(),
                    "sqmean_state_dict": sqmean_model.state_dict(),
                    "epoch": epoch},
                   os.path.join(save_dir, 'swag_ep_{}.pt'.format(epoch)))

    return mean_model, sqmean_model
