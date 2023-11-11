
from __future__ import division
from __future__ import absolute_import

import torch
import os
import json
import torch.nn as nn
from utils.misc import data_path_join
# from utils_.utils import RecorderMeter
# from utils_ import utils
from models import (resnet_preact, resnet, wrn, vgg, densenet)# pyramidnet wrn_adv
# , wrn_sap, wrn_adv_sap, model_zoo, vgg_rse, pni_model
# from models.Resnet import resnet152_denoise, resnet101_denoise
import numpy as np
from torchvision import models, datasets
import timm

def load_torch_models(model_name):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if model_name == "pyramidnet":
        pass
        # TRAINED_MODEL_PATH = data_path_join('pretrained_models/pyramidnet_basic_110_84/00/')
        # filename = 'model_best_state.pth'
        # with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
        #     pretrained_model = pyramidnet.Network(json.load(fr)['model_config'])
        #     pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'resnet_adv_4':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/resnet_adv_4/cifar-10_linf/')
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = resnet.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
    elif model_name == 'resnet':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/resnet_basic_110/00/')
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = resnet.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    elif model_name == 'wrn':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_28_10/00/')
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = wrn.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])

    elif model_name == 'dense':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/densenet_BC_100_12/00/')
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = densenet.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])


    elif model_name == 'vgg_rse':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/rse_model/')
        filename = 'cifar10_vgg_rse_005.pth'
        # with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            # pretrained_model = vgg.Network(json.load(fr)['model_config'])
        noise_init = 0
        noise_inner = 0
        pretrained_model = nn.DataParallel(vgg_rse.VGG_RSE('VGG16', 10, noise_init, noise_inner, img_width=32), device_ids=range(1))
        pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))

    elif model_name == 'rse':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/rse_model/')
        filename = 'cifar10_vgg_rse.pth'
        # with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            # pretrained_model = vgg.Network(json.load(fr)['model_config'])
        noise_init = 0.2
        noise_inner = 0.1
        pretrained_model = nn.DataParallel(vgg_rse.VGG_RSE('VGG16', 10, noise_init, noise_inner, img_width=32), device_ids=range(1))
        pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))



    elif model_name == 'vgg':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/vgg_15_BN_64/00/')
        filename = 'model_best_state.pth'
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = vgg.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename),map_location=device)['state_dict'])


    elif model_name == 'wrn_adv':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
        filename = 'adv_wrn16_linf.pth'
        pretrained_model = wrn_adv.create_model(name='wide', num_classes = 10)
        if hasattr(pretrained_model, 'module'):
            pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename),map_location=device)['state_dict'])
    
    elif model_name == 'wrn28':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
        filename = 'cifar_wrn_28.pth'
        pretrained_model = wrn.WideNet()
        pretrained_model = torch.nn.DataParallel(pretrained_model)
        checkpoint = torch.load(os.path.join(TRAINED_MODEL_PATH, filename))
        # if hasattr(pretrained_model, 'module'):
        #     pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(checkpoint['net'])
    
    # elif model_name == 'wrn28_sap':
    #     TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
    #     filename = 'cifar_wrn_28.pth'
    #     pretrained_model = wrn_sap.WideNet()
    #     pretrained_model = torch.nn.DataParallel(pretrained_model)
    #     checkpoint = torch.load(os.path.join(TRAINED_MODEL_PATH, filename))
    #     # if hasattr(pretrained_model, 'module'):
    #     #     pretrained_model = pretrained_model.module
    #     pretrained_model.load_state_dict(checkpoint['net'])
    
    elif model_name == 'wrn16_clean':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
        filename = 'wrn_clean.pth'
        pretrained_model = wrn_adv.create_model(name='wide', num_classes = 10)
        if hasattr(pretrained_model, 'module'):
            pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename),map_location=device)['state_dict'])


    elif model_name == 'wrn16_fine':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
        filename = 'wrn16_fine.pth'
        pretrained_model = wrn_adv.create_model(name='wide16', num_classes = 10)
        # if hasattr(pretrained_model, 'module'):
        #     pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename),map_location=device))


    elif model_name == 'wrn28_fine':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
        filename = 'wrn28_fine.pth'
        pretrained_model = wrn.WideNet()
        pretrained_model = torch.nn.DataParallel(pretrained_model)
        # if hasattr(pretrained_model, 'module'):
        #     pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename),map_location=device))



    elif model_name == 'wrn16_clean_sap':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
        filename = 'wrn_clean.pth'
        pretrained_model = wrn_adv_sap.create_model(name='wide', num_classes = 10)
        if hasattr(pretrained_model, 'module'):
            pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename),map_location=device)['state_dict'])

 
    # elif model_name == 'wrn_gau':
    #     TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
    #     filename = 'wrn_005.pth'
    #     # pretrained_model = nn.DataParallel(adv_model.create_model(name='wide_gau', num_classes = 10), device_ids=range(1))
    #     # pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))        
    #     pretrained_model = wrn_adv.create_model(name='wide', num_classes = 10)
    #     if hasattr(pretrained_model, 'module'):
    #         pretrained_model = pretrained_model.module
    #     pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])

    elif model_name == 'wrn_01':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
        filename = '01checkpoint_200.pth'
        # pretrained_model = nn.DataParallel(adv_model.create_model(name='wide_gau', num_classes = 10), device_ids=range(1))
        # pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))        
        pretrained_model = wrn_adv.create_model(name='wide', num_classes = 10)
        if hasattr(pretrained_model, 'module'):
            pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])

    elif model_name == 'res18':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/resnet/')
        filename = 'cifar_res18.pth'
        pretrained_model = resnet.ResNet18(pretrained=True)
        pretrained_model = torch.nn.DataParallel(pretrained_model)
        checkpoint = torch.load(os.path.join(TRAINED_MODEL_PATH, filename))
        if hasattr(pretrained_model, 'module'):
            pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(checkpoint['net'])


    elif model_name == 'res50':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/resnet/')
        filename = 'cifar_res50.pth'
        pretrained_model = resnet.ResNet50()
        pretrained_model = torch.nn.DataParallel(pretrained_model)
        checkpoint = torch.load(os.path.join(TRAINED_MODEL_PATH, filename))
        # if hasattr(pretrained_model, 'module'):
        #     pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(checkpoint['net'])


    elif model_name == 'res101':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/resnet/')
        filename = 'cifar_res101.pth'
        pretrained_model = resnet.ResNet101()
        pretrained_model = torch.nn.DataParallel(pretrained_model)
        checkpoint = torch.load(os.path.join(TRAINED_MODEL_PATH, filename))
        # if hasattr(pretrained_model, 'module'):
        #     pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(checkpoint['net'])

    elif model_name == 'dense121':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/densenet/')
        filename = 'cifar_dense121.pth'
        pretrained_model = densenet.DenseNet121()
        pretrained_model = torch.nn.DataParallel(pretrained_model)
        checkpoint = torch.load(os.path.join(TRAINED_MODEL_PATH, filename))
        # if hasattr(pretrained_model, 'module'):
        #     pretrained_model = pretrained_model.module
        pretrained_model.load_state_dict(checkpoint['net'])


    elif model_name == 'best_adv_wrn34':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
        filename = 'cifar10_linf_wrn34-20.pt'
        pretrained_model = model_zoo.WideResNet(
            num_classes=10, depth=34, width=20,
            activation_fn=model_zoo.Swish, mean=model_zoo.CIFAR10_MEAN,
            std=model_zoo.CIFAR10_STD)
        # dataset_fn = datasets.CIFAR10
        params = torch.load(os.path.join(TRAINED_MODEL_PATH, filename))
        pretrained_model.load_state_dict(params)

    elif model_name == 'best_adv_wrn28':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/wrn_adv/')
        filename = 'cifar10_linf_wrn28-10_with.pt'
        pretrained_model = model_zoo.WideResNet(
            num_classes=10, depth=28, width=10,
            activation_fn=model_zoo.Swish, mean=model_zoo.CIFAR10_MEAN,
            std=model_zoo.CIFAR10_STD)
        # dataset_fn = datasets.CIFAR10
        params = torch.load(os.path.join(TRAINED_MODEL_PATH, filename))
        pretrained_model.load_state_dict(params)


    elif model_name == 'pni':
        TRAINED_MODEL_PATH = data_path_join('pretrained_models/pni_model/')
        filename = 'checkpoint.pth.tar'
        recorder = RecorderMeter(200)  # count number of epoches
        pretrained_model = pni_model.noise_resnet20()
        checkpoint = torch.load(os.path.join(TRAINED_MODEL_PATH, filename))
        state_tmp = pretrained_model.state_dict()
        if 'state_dict' in checkpoint.keys():
            state_tmp.update(checkpoint['state_dict'])
        else:
            state_tmp.update(checkpoint)

        pretrained_model.load_state_dict(state_tmp)

    elif model_name == 'timm_res18':
        pretrained_model = timm.create_model("resnet18", num_classes=10, pretrained=False)

        # override model
        pretrained_model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        pretrained_model.maxpool = nn.Identity()  # type: ignore
        # model.fc = nn.Linear(512,  10)

        pretrained_model.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                            "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
                            map_location="cpu", 
                            file_name="resnet18_cifar10.pth",
                    )
        )



    from advertorch.utils import NormalizeByChannelMeanStd

    if  (model_name == 'vgg_plain') or (model_name == 'vgg_rse') or (model_name == 'rse') or (model_name == 'pni') or (model_name == 'wrn_adv') or (model_name == 'wrn16_clean') or (model_name == 'wrn16_fine') or (model_name == 'wrn16_clean_sap') or (model_name == 'wrn_01') or (model_name == 'best_adv_wrn28') or (model_name == 'best_adv_wrn34') or (model_name == 'wrn_stop') or (model_name == 'sat_wrn16_l2') or (model_name == 'sat_wrn16_linf'):
        net = pretrained_model

    elif 'wrn28' in model_name:
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])

        # std = np.array([0.2470, 0.2435, 0.2616])
        normalize = NormalizeByChannelMeanStd(
                mean=mean.tolist(), std=std.tolist())

        net = nn.Sequential(
            normalize,
            pretrained_model
        )

    else:
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])

        normalize = NormalizeByChannelMeanStd(
                mean=mean.tolist(), std=std.tolist())

        net = nn.Sequential(
            normalize,
            pretrained_model
        )

    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    return net


class Permute(nn.Module):

    def __init__(self, permutation = [2,1,0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        
        return input[:, self.permutation]


def load_torch_models_imagesub(model_name):
    if model_name == "VGG16":
        pretrained_model = models.vgg16_bn(pretrained=True)
    elif model_name == 'Resnet18':
        pretrained_model = models.resnet18(pretrained=True)
    elif model_name == 'Resnet34':
        pretrained_model = models.resnet34(pretrained=True)
    elif model_name == 'Resnet50':
        pretrained_model = models.resnet50(pretrained=True)
    elif model_name == 'Resnet101':
        pretrained_model = models.resnet101(pretrained=True)
    # elif model_name == 'Squeezenet':
    #     pretrained_model = models.squeezenet1_1(pretrained=True)
    elif model_name == 'Googlenet':
        pretrained_model = models.googlenet(pretrained=True)
    elif model_name == 'Inception':
        pretrained_model = models.inception_v3(pretrained=True)
    elif model_name == 'Widenet':
        pretrained_model = models.wide_resnet50_2(pretrained=True)
    # elif model_name == 'Adv_Denoise_Resnext101':
    #     pretrained_model = resnet101_denoise()
    #     loaded_state_dict = torch.load(os.path.join('./results/denoise/', model_name+".pytorch"))
    #     pretrained_model.load_state_dict(loaded_state_dict)
    
    # if 'defense' in state and state['defense']:
    #     net = nn.Sequential(
    #         Normalize(mean, std),
    #         Permute([2,1,0]),
    #         pretrained_model
    #     )
    # else:
    # net = nn.Sequential(
    #     Normalize(mean, std),
    #     pretrained_model
    # )

    from advertorch.utils import NormalizeByChannelMeanStd

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalize = NormalizeByChannelMeanStd(
            mean=mean.tolist(), std=std.tolist())

    if 'Denoise' in model_name:
        net = nn.Sequential(
            # Normalize(mean, std),
            normalize,
            Permute([2,1,0]),
            pretrained_model
        )
    
    else:
        net = nn.Sequential(
            # Normalize(mean, std),
            normalize,
            pretrained_model
        )
            

    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    return net