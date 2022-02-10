from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision.models import resnet18, resnet34, resnet50


class Noise(nn.Module):
    def __init__(self, std):
        super(Noise, self).__init__()
        self.std = std
        self.buffer = None
    # the std is fixed by the defender. 
    def forward(self, x):
        if self.std > 0:
            if self.buffer is None:
                self.buffer = torch.Tensor(x.size()).normal_(0, self.std).cuda()
            else:
                self.buffer.data.resize_(x.size()).normal_(0, self.std)
            return x + self.buffer
        return x



class WideResNetBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(WideResNetBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class WideResNetBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(WideResNetBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNet_Gau(nn.Module):
    def __init__(self, depth=16, num_classes=10, widen_factor=10, dropRate=0.0, noise_init = 0.0, noise_inner = 0.0):
        super(WideResNet_Gau, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WideResNetBasicBlock
        # 1st conv before any network block
        
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = WideResNetBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = WideResNetBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = WideResNetBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.noise_init = noise_init
        self.noise_inner = noise_inner
        self.noise_layer = Noise(self.noise_init)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                # init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                tmp = np.sqrt(3. / m.weight.data.shape[0])
                m.weight.data.uniform_(-tmp, tmp)
                m.bias.data.zero_()
                # init.kaiming_normal_(m.weight)
                # init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.noise_layer(x)
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.nChannels)
        out = self.fc(x)
        return x, out


class WideResNet(nn.Module):
    def __init__(self, depth=16, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WideResNetBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = WideResNetBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = WideResNetBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = WideResNetBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                # init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                tmp = np.sqrt(3. / m.weight.data.shape[0])
                m.weight.data.uniform_(-tmp, tmp)
                m.bias.data.zero_()
                # init.kaiming_normal_(m.weight)
                # init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.nChannels)
        out = self.fc(x)

        return x, out

def wideresnet16(**kwargs):
    return WideResNet(depth=16, **kwargs)

def wideresnet22(**kwargs):
    return WideResNet(depth=22, **kwargs)

def wideresnet_gau(**kwargs):
    return WideResNet_Gau(depth=16, **kwargs)


class MnistModel(nn.Module):
    """ Construct basic MnistModel for mnist adversal attack """
    def __init__(self, re_init=False, has_dropout=False):
        super(MnistModel, self).__init__()
        self.re_init = re_init
        self.has_dropout = has_dropout
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        if self.has_dropout:
            self.dropout = nn.Dropout()

        if self.re_init:
            self._init_params(self.conv1)
            self._init_params(self.conv2)
            self._init_params(self.fc1)
            self._init_params(self.fc2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)

        if self.has_dropout:
            x = self.dropout(x)

        x = self.fc2(x)

        return x

    def _init_params(self, module, mean=0.1, std=0.1):
        init.normal_(module.weight, std=0.1)
        if hasattr(module, 'bias'):
            init.constant_(module.bias, mean)

__factory = {
    # resnet series, kwargs: num_classes
    'resnet': resnet18, 
    'resnet18': resnet18, 
    'resnet34': resnet34, 
    'resnet50': resnet50, 
    # wideresnet series, kwargs: num_classes, widen_factor, dropRate
    'wide': wideresnet16, 
    'wide_gau': wideresnet_gau, 
    'wideresnet': wideresnet16, 
    'wideresnet16': wideresnet16, 
    'wideresnet22': wideresnet22, 
    # mnist, kwargs: has_dropout
    'mnist': MnistModel, 
}

def create_model(name, **kwargs):
    assert(name in __factory), 'invalid network'
    return __factory[name](**kwargs)


if __name__ == '__main__':
    net = create_model('wide')
    import pdb; pdb.set_trace()  # breakpoint 2e2204d9 //