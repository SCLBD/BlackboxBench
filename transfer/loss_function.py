import copy

import torch
import torch.nn.functional as F
from utils.registry import Registry
import torch.nn as nn
import os
import numpy as np
from utils.helper import get_source_layers, InsideCounter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loss_function(loss_fn_name):
    """
    Transform an input string into the loss function.
    :param loss_fn_name: A string describing the loss function.
    :return: loss function.
    :raises: ValueError: if loss function name is unknown.
    """
    if loss_fn_name:
        try:
            loss_fn = Registry.lookup(f"loss_function.{loss_fn_name}")()
        except SyntaxError as err:
            raise ValueError(f"Syntax error on: {loss_fn_name}") from err

    return loss_fn


@Registry.register("loss_function.cross_entropy")
def CE():
    """
    Cross-entropy loss, which supports ensemble on logits or losses.
    """
    def _CE(args, img, true_label, target_label, ensemble_models, reduction='mean', ensemble='logit'):
        if ensemble == 'logit':
            # ensemble on logits
            out = 0
            for n_model_iter in range(len(ensemble_models)):
                # To avoid OOM in the main(0) GPU, we allocate models on other GPUs.
                model_device = next(ensemble_models[n_model_iter].parameters()).device
                out += ensemble_models[n_model_iter].to(DEVICE)(img)
                ensemble_models[n_model_iter].to(model_device)
                torch.cuda.empty_cache()
            out /= len(ensemble_models)

            loss = torch.nn.CrossEntropyLoss(reduction=reduction)(out, true_label) \
                if not args.targeted else -torch.nn.CrossEntropyLoss()(out, target_label)

        elif ensemble == 'loss':
            # ensemble on losses
            loss = 0
            for n_model_iter in range(len(ensemble_models)):
                # To avoid OOM in the main(0) GPU, we allocate models on other GPUs.
                model_device = next(ensemble_models[n_model_iter].parameters()).device
                out = ensemble_models[n_model_iter].to(DEVICE)(img)
                ensemble_models[n_model_iter].to(model_device)
                torch.cuda.empty_cache()
                loss += torch.nn.CrossEntropyLoss(reduction=reduction)(out, true_label) \
                        if not args.targeted else -torch.nn.CrossEntropyLoss()(out, target_label)

            loss /= len(ensemble_models)
        else:
            raise ValueError('Invalid ensemble method.')

        return loss

    return _CE


@Registry.register("loss_function.max_logit")
def ML():
    """
    Max-logit loss, which supports ensemble on logits or losses.
    """
    def _ML(args, img, true_label, target_label, ensemble_models, ensemble='logit'):
        if ensemble == 'logit':
            # ensemble on logits
            logits = 0
            for n_model_iter in range(len(ensemble_models)):
                # To avoid OOM in the main(0) GPU, we allocate models on other GPUs.
                model_device = next(ensemble_models[n_model_iter].parameters()).device
                logits += ensemble_models[n_model_iter].to(DEVICE)(img)
                ensemble_models[n_model_iter].to(model_device)
            logits /= len(ensemble_models)

            if args.targeted:
                real = logits.gather(1, target_label.unsqueeze(1)).squeeze(1)
                loss = real.sum()
            else:
                real = logits.gather(1, true_label.unsqueeze(1)).squeeze(1)
                loss = -1 * real.sum()

        elif ensemble == 'loss':
            # ensemble on losses
            loss = 0
            for n_model_iter in range(len(ensemble_models)):
                # To avoid OOM in the main(0) GPU, we allocate models on other GPUs.
                model_device = next(ensemble_models[n_model_iter].parameters()).device
                logits = ensemble_models[n_model_iter].to(DEVICE)(img)
                ensemble_models[n_model_iter].to(model_device)

                if args.targeted:
                    real = logits.gather(1, target_label.unsqueeze(1)).squeeze(1)
                    loss += real.sum()
                else:
                    real = logits.gather(1, true_label.unsqueeze(1)).squeeze(1)
                    loss += -1 * real.sum()

            loss /= len(ensemble_models)

        else:
            raise ValueError('Invalid ensemble method.')

        return loss

    return _ML


@Registry.register("loss_function.linbp")
def LinBP(linbp_layer):
    """
    This is the forward function for LinBP, modified based on the following source. Build upon I-FGSM framework,
    the complete LinBP algorithm also includes a novel backward function---linbp_backw_resnet50(), defined in
    'gradient_calculation.py'.
    link:
        https://github.com/qizhangli/linbp-attack
    citation:
        @inproceedings{guo2020backpropagating,
            title={Backpropagating Linearly Improves Transferability of Adversarial Examples.},
            author={Guo, Yiwen and Li, Qizhang and Chen, Hao},
            booktitle={NeurIPS},
            year={2020}
        }
    """
    args = Registry._GLOBAL_REGISTRY['args']

    def linbp_relu(x):
        x_p = F.relu(-x)
        x = x + x_p.data
        return x

    def block_func(block, x, linbp):
        identity = x
        conv_in = x + 0
        out = block.conv1(conv_in)
        out = block.bn1(out)
        out_0 = out + 0
        if linbp:
            out = linbp_relu(out_0)
        else:
            out = block.relu(out_0)
        ori_mask_0 = out.data.bool().int()

        out = block.conv2(out)
        out = block.bn2(out)
        out_1 = out + 0
        if linbp:
            out = linbp_relu(out_1)
        else:
            out = block.relu(out_1)
        ori_mask_1 = out.data.bool().int()

        out = block.conv3(out)
        out = block.bn3(out)

        downsample = block.downsample if 'NIPS2017' in args.source_model_path[0] else block.shortcut
        if downsample is not None:
            identity = downsample(identity)
        identity_out = identity + 0
        x_out = out + 0

        out = identity_out + x_out
        out = block.relu(out)
        ori_mask_2 = out.data.bool().int()
        return out, (ori_mask_0, ori_mask_1, ori_mask_2), (identity_out, x_out), (out_0, out_1), (0, conv_in) # return out relu_mask conv_out(identity conved_x) relu_in ori_x

    def linbp_forw_vgg19(model, img):
        """LinBP forward propagation for vgg19_bn"""
        out = model[0](img)
        for ind, mm in enumerate(model[1].features):
            if isinstance(mm, nn.ReLU) and ind >= linbp_layer:
                out = linbp_relu(out)
            else:
                out = mm(out)
        out = out.view(out.size(0), -1)
        out = model[1].classifier(out)
        return out

    def linbp_forw_resnet50(model, img):
        """LinBP forward propagation for resnet50"""
        jj = int(linbp_layer.split('_')[0])
        kk = int(linbp_layer.split('_')[1])
        x = model[0](img)
        x = model[1].conv1(x)
        x = model[1].bn1(x)
        x = model[1].relu(x)
        x = model[1].maxpool(x) if 'NIPS2017' in args.source_model_path[0] else x
        ori_mask_ls = []
        conv_out_ls = []
        relu_out_ls = []
        conv_input_ls = []

        def layer_forw(jj, kk, jj_now, kk_now, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls):
            if jj < jj_now:
                x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
                ori_mask_ls.append(ori_mask)
                conv_out_ls.append(conv_out)
                relu_out_ls.append(relu_out)
                conv_input_ls.append(conv_in)
            elif jj == jj_now:
                if kk_now >= kk:
                    x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
                    ori_mask_ls.append(ori_mask)
                    conv_out_ls.append(conv_out)
                    relu_out_ls.append(relu_out)
                    conv_input_ls.append(conv_in)
                else:
                    x, _, _, _, _ = block_func(mm, x, linbp=False)
            else:
                x, _, _, _, _ = block_func(mm, x, linbp=False)
            return x, ori_mask_ls

        for ind, mm in enumerate(model[1].layer1):
            x, ori_mask_ls = layer_forw(jj, kk, 1, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls)
        for ind, mm in enumerate(model[1].layer2):
            x, ori_mask_ls = layer_forw(jj, kk, 2, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls)
        for ind, mm in enumerate(model[1].layer3):
            x, ori_mask_ls = layer_forw(jj, kk, 3, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls)
        for ind, mm in enumerate(model[1].layer4):
            x, ori_mask_ls = layer_forw(jj, kk, 4, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls)

        if 'NIPS2017' in args.source_model_path[0]:
            x = model[1].avgpool(x)
            x = torch.flatten(x, 1)
            x = model[1].fc(x)
        elif 'CIFAR10' in args.source_model_path[0]:
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            x = model[1].linear(x)

        return x, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls

    def _LinBP(args, img, true_label, target_label, ensemble_models):
        assert len(ensemble_models) == 1, "LinBP doesn't support ensemble attack"
        model = ensemble_models[0].module  # remove DataParallel mode only in LinBP

        if 'vgg19_bn' in args.source_model_path[0]:
            out = linbp_forw_vgg19(model, img)
            loss_args = []
        elif 'resnet50' in args.source_model_path[0]:
            out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(model, img)
            loss_args = [conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls]
        else:
            raise ValueError('LinBP only support VGG19_BN(CIFAR10) or ResNet50(NIPS2017) as surrogate model.')

        loss = torch.nn.CrossEntropyLoss()(out, true_label) \
            if not args.targeted else -torch.nn.CrossEntropyLoss()(out, target_label)
        return loss, loss_args

    return _LinBP


mid_output = None
@Registry.register("loss_function.ila_loss")
def ILA(ila_layer):
    """
    This function is the core of ILA, modified based on the following source:
    link:
        https://github.com/CUAI/Intermediate-Level-Attack
    citation:
        @article{Huang2019EnhancingAE,
          title={Enhancing Adversarial Example Transferability with an Intermediate Level Attack},
          author={Qian Huang and Isay Katsman and Horace He and Zeqi Gu and Serge J. Belongie and Ser-Nam Lim},
          journal={ArXiv},
          year={2019},
          volume={abs/1907.10823}
        }
    """
    args = Registry._GLOBAL_REGISTRY['args']
    source_models = Registry._GLOBAL_REGISTRY['source_models']
    bsl_img = []
    for bsl_file_ind in range(len(os.listdir(args.bsl_adv_img_path))):
        bsl_img.append(torch.from_numpy(np.load(args.bsl_adv_img_path + '/batch_{}.npy'.format(bsl_file_ind))).float())
    bsl_img = torch.cat(bsl_img)
    bsl_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(bsl_img), batch_size=200, shuffle=False, pin_memory=False)
    ori_dl = Registry._GLOBAL_REGISTRY['data_loader']

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o

    assert len(args.source_model_path) == 1, "ILA doesn't support ensemble attack."
    source_layers = get_source_layers(args.source_model_path, next(source_models)[0].module[1])
    # model_layers = next(source_models)[0].module[1]
    # try:
    #     source_layers = get_source_layers(args.source_model_path, model_layers[1])
    # except TypeError:
    #     source_layers = get_source_layers(args.source_model_path, model_layers)
    if len(ila_layer.split('_')) == 2:
        feature_layer = source_layers[int(ila_layer.split('_')[0])][1][1][int(ila_layer.split('_')[1])]
    else:
        feature_layer = source_layers[int(ila_layer)][1][1]
    h = feature_layer.register_forward_hook(get_mid_output)

    ori_feature, bsl_feature = [], []
    with torch.no_grad():
        for ori_batch, _, _ in ori_dl:
            ori_batch = ori_batch.to(DEVICE)
            out = next(source_models)[0].module[0](ori_batch)
            out = next(source_models)[0].module[1](out)
            ori_fea_batch = torch.zeros(mid_output.size()).to(DEVICE)
            ori_fea_batch.copy_(mid_output)
            ori_feature.append(ori_fea_batch.to(device='cuda:1'))
            print('1 batch')
        for [bsl_batch] in bsl_dl:
            bsl_batch = bsl_batch.to(DEVICE)
            out = next(source_models)[0].module[0](bsl_batch)
            out = next(source_models)[0].module[1](out)
            bsl_fea_batch = torch.zeros(mid_output.size()).to(DEVICE)
            bsl_fea_batch.copy_(mid_output)
            bsl_feature.append(bsl_fea_batch.to(device='cuda:1'))

    ori_feature = torch.cat(ori_feature)
    bsl_feature = torch.cat(bsl_feature)

    class Proj_Loss(torch.nn.Module):
        def __init__(self):
            super(Proj_Loss, self).__init__()

        def forward(self, old_attack_mid, new_mid, original_mid, coeff):
            x = (old_attack_mid - original_mid).reshape(1, -1)
            y = (new_mid - original_mid).reshape(1, -1)
            x_norm = x / x.norm()

            proj_loss = torch.mm(y, x_norm.transpose(0, 1)) / x.norm()
            return proj_loss

    counter = InsideCounter(args)
    def _ILA(args, img, true_label, target_label, ensemble_models, ori_f=ori_feature, bsl_f=bsl_feature):
        ori_fea_i = ori_f[counter.batch_size_cur:counter.batch_size_cur + img.shape[0]].to(device='cuda:0')
        bsl_fea_i = bsl_f[counter.batch_size_cur:counter.batch_size_cur + img.shape[0]].to(device='cuda:0')
        counter.step(img.shape[0])

        output_perturbed = ensemble_models[0].module[0](img)
        output_perturbed = ensemble_models[0].module[1](output_perturbed)
        loss = Proj_Loss()(bsl_fea_i.detach(), mid_output, ori_fea_i.detach(), 1.0)

        return loss

    return _ILA


@Registry.register("loss_function.fia_loss")
def FIA(fia_layer, N=30, drop_rate=0.3):
    """
    This function is the core of FIA, modified based on a third-party repo:
    link:
        https://github.com/ZhengyuZhao/TransferAttackEval
    citation:
        @inproceedings{wang2021feature,
          title={Feature importance-aware transferable adversarial attacks},
          author={Wang, Zhibo and Guo, Hengchang and Zhang, Zhifei and Liu, Wenxin and Qin, Zhan and Ren, Kui},
          booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
          pages={7639--7648},
          year={2021}
        }
    """
    args = Registry._GLOBAL_REGISTRY['args']
    assert len(args.source_model_path) == 1, "FIA doesn't support ensemble attack."
    source_models = Registry._GLOBAL_REGISTRY['source_models']
    # model = next(source_models)[0].module
    # model_layers = model[1]
    # try:
    #     source_layers = get_source_layers(args.source_model_path, model_layers[1])
    # except TypeError:
    #     source_layers = get_source_layers(args.source_model_path, model_layers)
    model = next(source_models)[0].module
    source_layers = get_source_layers(args.source_model_path, model[1])
    if len(fia_layer.split('_')) == 2:
        feature_layer = source_layers[int(fia_layer.split('_')[0])][1][1][int(fia_layer.split('_')[1])]
    else:
        feature_layer = source_layers[int(fia_layer)][1][1]

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o

    def get_mid_grad(m, i, o):
        global mid_grad
        mid_grad = o

    h1 = feature_layer.register_forward_hook(get_mid_output)
    h2 = feature_layer.register_full_backward_hook(get_mid_grad)

    ori_dl = Registry._GLOBAL_REGISTRY['data_loader']
    agg_grad = []
    for ori_batch, true_label_batch, target_label_batch in ori_dl:
        label_batch = target_label_batch if args.targeted else true_label_batch
        agg_grad_batch = 0
        X_random = ori_batch.clone().detach().cuda().requires_grad_(True)
        for _ in range(N):
            X_random_norm = model[0](X_random)
            Mask = torch.bernoulli(torch.ones_like(X_random_norm) * (1 - drop_rate)).cuda()
            X_random_M = X_random_norm * Mask
            output_random = model[1](X_random_M)
            loss = 0
            for batch_i in range(ori_batch.shape[0]):
                loss += output_random[batch_i][label_batch[batch_i]]
            loss.backward()
            agg_grad_batch += mid_grad[0].detach()
            X_random.grad.zero_()
        agg_grad.append(agg_grad_batch.to(device='cuda:1'))
        print('1 batch')
    agg_grad = torch.cat(agg_grad)
    for batch_i in range(agg_grad.shape[0]):
        agg_grad[batch_i] /= agg_grad[batch_i].norm(2)
    h2.remove()

    counter = InsideCounter(args)
    def _FIA(args, img, true_label, target_label, ensemble_models):
        agg_grad_i = agg_grad[counter.batch_size_cur:counter.batch_size_cur + img.shape[0]].to(device='cuda:0')
        counter.step(img.shape[0])

        output_perturbed = ensemble_models[0].module(img)
        loss = -(mid_output * agg_grad_i).sum()
        return -loss if args.targeted else loss

    return _FIA


@Registry.register('loss_function.naa_loss')
def NAA(naa_layer, N=30):
    """
    This function is the core of NAA, modified based on a third-party repo:
    link:
        https://github.com/ZhengyuZhao/TransferAttackEval
    citation:
        @inproceedings{zhang2022improving,
          title={Improving adversarial transferability via neuron attribution-based attacks},
          author={Zhang, Jianping and Wu, Weibin and Huang, Jen-tse and Huang, Yizhan and Wang, Wenxuan and Su, Yuxin and Lyu, Michael R},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          pages={14993--15002},
          year={2022}
        }
    """
    args = Registry._GLOBAL_REGISTRY['args']
    assert len(args.source_model_path) == 1, "NAA doesn't support ensemble attack."
    source_models = Registry._GLOBAL_REGISTRY['source_models']
    # model = next(source_models)[0].module
    # model_layers = model[1]
    # try:
    #     source_layers = get_source_layers(args.source_model_path, model_layers[1])
    # except TypeError:
    #     source_layers = get_source_layers(args.source_model_path, model_layers)
    model = next(source_models)[0].module
    source_layers = get_source_layers(args.source_model_path, model[1])
    if len(naa_layer.split('_')) == 2:
        feature_layer = source_layers[int(naa_layer.split('_')[0])][1][1][int(naa_layer.split('_')[1])]
    else:
        feature_layer = source_layers[int(naa_layer)][1][1]

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o

    def get_mid_grad(m, i, o):
        global mid_grad
        mid_grad = o


    h1 = feature_layer.register_forward_hook(get_mid_output)
    h2 = feature_layer.register_full_backward_hook(get_mid_grad)

    ori_dl = Registry._GLOBAL_REGISTRY['data_loader']
    agg_grad, fea_prime = [], []
    for ori_batch, label_batch, _ in ori_dl:
        # integrated attention
        agg_grad_batch = 0
        X = ori_batch.clone().detach().cuda().requires_grad_(True)
        for iter_n in range(N):
            X_norm = model[0](X)
            X_Step = torch.zeros(X_norm.size()).cuda()
            X_Step = X_Step + X_norm * iter_n / N
            output_random = model[1](X_Step)
            output_random = torch.softmax(output_random, 1)

            loss = 0
            for batch_i in range(ori_batch.shape[0]):
                loss += output_random[batch_i][label_batch[batch_i]]
            loss.backward()
            agg_grad_batch += mid_grad[0].detach()
            X.grad.zero_()
        agg_grad_batch /= N
        agg_grad.append(agg_grad_batch.to(device='cuda:1'))

        # feature of baseline images
        X_prime_batch = torch.zeros(X_norm.size()).cuda()
        model[1](X_prime_batch)
        fea_prime_batch = mid_output.detach().clone()
        fea_prime.append(fea_prime_batch.to(device='cuda:1'))
        print('1 batch')

    agg_grad = torch.cat(agg_grad)
    fea_prime = torch.cat(fea_prime)
    h2.remove()

    counter = InsideCounter(args)
    def _NAA(args, img, true_label, target_label, ensemble_models):
        agg_grad_i = agg_grad[counter.batch_size_cur:counter.batch_size_cur+img.shape[0]].to(device='cuda:0')
        fea_prime_i = fea_prime[counter.batch_size_cur:counter.batch_size_cur+img.shape[0]].to(device='cuda:0')
        counter.step(img.shape[0])

        output_perturbed = ensemble_models[0].module(img)
        loss = -((mid_output - fea_prime_i) * agg_grad_i).sum()
        return loss

    return _NAA