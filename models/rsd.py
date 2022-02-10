import torch.nn as nn
import math
# n_w = 0


def weights_init(m, act_type='relu'):
    # global n_w
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if act_type == 'selu':
            n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
            m.weight.data.normal_(0.0, 1.0 / math.sqrt(n))
        else:
            m.weight.data.normal_(0.0, 0.02)
            # n_w += m.in_channels * m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
            # n_w += m.out_channels
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    #     n_w += m.out_channels
    # print(n_w)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type='instance', act_type='selu', use_dropout=False, n_blocks=1,
                 padding_type='reflect', gpu_ids=[]):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpulist = gpu_ids
        self.num_gpus = len(self.gpulist)

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        # initial layer
        model0 = [nn.Conv2d(input_nc, 3 * ngf, kernel_size=1, padding=0, bias=use_bias),
                  self.act]

        model0 += [nn.ReflectionPad2d(3),
                   nn.Conv2d(3 * ngf, ngf, kernel_size=7, padding=0, bias=use_bias),
                   norm_layer(ngf),
                   self.act]

        # down-sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2), self.act]
        self.model0 = nn.Sequential(*model0)
        # self.model0.cuda(self.gpulist[0])

        # block
        model1 = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                   use_dropout=use_dropout, use_bias=use_bias)]
        self.model1 = nn.Sequential(*model1)
        # self.model1.cuda(self.gpulist[0])

        # up-sampling
        model2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                          output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), self.act]
        model2 += [nn.ReflectionPad2d(3)]
        model2 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model2 += [nn.Tanh()]

        self.model2 = nn.Sequential(*model2)
        # self.model2.cuda(self.gpulist[0])

    def forward(self, x):
        # input
        # x = x.cuda(self.gpulist[0])

        # down-sampling
        x = self.model0(x)

        # block
        x = self.model1(x)

        # up-sampling
        x = self.model2(x)

        return x


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
            nn.Alpha

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
