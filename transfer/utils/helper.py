import os
import torchvision.transforms as T
import numpy as np
import torch
import shutil
import re
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_and_clip(args, adv_img, ori_img, update_dir):
    if args.norm_type == "inf":
        adv_img = adv_img.data + args.norm_step * torch.sign(update_dir)
        adv_img = torch.where(adv_img > ori_img + args.epsilon, ori_img + args.epsilon, adv_img)
        adv_img = torch.where(adv_img < ori_img - args.epsilon, ori_img - args.epsilon, adv_img)
        adv_img = torch.clamp(adv_img, min=0, max=1)
    elif args.norm_type == "2":
        update_dir = update_dir / (update_dir.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-12)
        adv_img = adv_img.data + args.norm_step * update_dir
        l2_perturb = adv_img - ori_img
        l2_perturb = l2_perturb.renorm(p=2, dim=0, maxnorm=args.epsilon)
        adv_img = ori_img + l2_perturb
        adv_img = torch.clamp(adv_img, min=0, max=1)
    return adv_img


def eval_attack(args, target_models):
    true_labels = torch.from_numpy(np.load(args.save_dir + '/true_labels.npy')).long()
    target_labels = torch.from_numpy(np.load(args.save_dir + '/target_labels.npy')).long()
    target = true_labels if not args.targeted else target_labels

    if 'NIPS2017' in args.source_model_path[0]:
        # trans = T.Compose([])
        trans = T.Compose([]) if args.targeted else T.Compose([T.Resize((256, 256)), T.CenterCrop((224, 224))])
    elif 'CIFAR10' in args.source_model_path[0]:
        trans = T.Compose([])
    advfile_ls = os.listdir(args.save_dir)
    att_suc = np.zeros((len(target_models),))  # initialize the attack success rate matrix
    img_num = 0
    for advfile_ind in range(len(advfile_ls)-2):    # minus 2 labels files
        adv_batch = torch.from_numpy(np.load(args.save_dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255

        img_num += adv_batch.shape[0]
        labels = target[advfile_ind * args.batch_size: advfile_ind * args.batch_size + adv_batch.shape[0]]
        inputs, labels = adv_batch.clone().to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            for j, target_model in enumerate(target_models):
                model_device = next(target_model.parameters()).device
                target_model.to(DEVICE)
                att_suc[j] += sum(torch.argmax(target_model(trans(inputs)), dim=1) != labels).cpu().numpy()
                target_model.to(model_device)
                torch.cuda.empty_cache()
    att_suc = 1 - att_suc/img_num if args.targeted else att_suc/img_num
    print(f"surrogate model {args.source_model_path}\n"
          f"victim model {args.target_model_path}\n"
          f"attack success rate: {att_suc}")
    return att_suc


def iter_eval_attack(args, target_models):
    true_labels = torch.from_numpy(np.load(args.save_dir + '/true_labels.npy')).long()
    target_labels = torch.from_numpy(np.load(args.save_dir + '/target_labels.npy')).long()
    target = true_labels if not args.targeted else target_labels

    if 'NIPS2017' in args.source_model_path[0]:
        # trans = T.Compose([])
        trans = T.Compose([]) if args.targeted else T.Compose([T.Resize((256, 256)), T.CenterCrop((224, 224))])
    elif 'CIFAR10' in args.source_model_path[0]:
        trans = T.Compose([])
    advfile_ls = os.listdir(args.save_iter_dir)
    for iter_idx in range(args.n_iter // args.save_fre):
        att_suc = np.zeros((len(target_models),))  # initialize the attack success rate matrix
        img_num = 0
        for advfile_ind in range(len(advfile_ls)):
            adv_batch = torch.from_numpy(np.load(args.save_iter_dir + '/batch_{}_iters.npy'.format(advfile_ind)))[iter_idx].squeeze().float() / 255
            img_num += adv_batch.shape[0]
            labels = target[advfile_ind * args.batch_size: advfile_ind * args.batch_size + adv_batch.shape[0]]
            inputs, labels = adv_batch.clone().to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                for j, target_model in enumerate(target_models):
                    model_device = next(target_model.parameters()).device
                    target_model.to(DEVICE)
                    att_suc[j] += sum(torch.argmax(target_model(trans(inputs)), dim=1) != labels).cpu().numpy()
                    target_model.to(model_device)
                    torch.cuda.empty_cache()
        att_suc = 1 - att_suc / img_num if args.targeted else att_suc / img_num
        print(f"attack success rate at {(iter_idx+1)*args.save_fre}th iter: {att_suc}")


def calculate_accuracy(data_loader, target_models):
    # trans = T.Compose([T.Resize((256, 256)), T.CenterCrop((224, 224))])
    ori_att_suc = np.zeros((len(target_models),))
    img_num = 0
    for ind, (ori_img, true_label, target_label) in tqdm(enumerate(data_loader)):
        ori_img, true_label, target_label = ori_img.to(DEVICE), true_label.to(DEVICE), target_label.to(DEVICE)
        img_num += ori_img.shape[0]
        with torch.no_grad():
            for j, target_model in enumerate(target_models):
                ori_att_suc[j] += sum(torch.argmax(target_model(ori_img), dim=1) != true_label).cpu().numpy()
    print(f"original attact success rate:{ori_att_suc/img_num}")
    return ori_att_suc/img_num


def makedir(path, exist_ok=False):
    if path != None:
        if os.path.exists(path) and not exist_ok:
            print('Save path already exist!')
            shutil.rmtree(path)
            print('Save path cleared!')
        os.makedirs(path, exist_ok=exist_ok)


def get_source_layers(model_path, model):
    model_name = re.match(r"^.+/pretrained/(\w+)$", model_path[0]).groups()[0]

    if model_name == 'resnet18':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)),
                                  ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'])))

    elif model_name == 'resnet50':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)),
                                  ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'])))

    elif model_name == 'densenet121':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get('features')._modules.get(name)),
                              ['conv0', 'denseblock1', 'transition1', 'denseblock2', 'transition2', 'denseblock3',
                               'transition3', 'denseblock4', 'norm5']))
        layer_list.append(('classifier', model._modules.get('classifier')))
        return list(enumerate(layer_list))

    elif model_name == 'densenet':
        # densenet model for cifar10
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get(name)),
                              ['conv1', 'dense1', 'trans1', 'dense2', 'trans2', 'dense3']))
        return list(enumerate(layer_list))

    elif model_name == 'vgg19_bn':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: ('layer ' + name, [model._modules.get('features')._modules.get(name)]),
                              ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49']))
        layer_list.append(('classifier', model._modules.get('classifier')))
        return list(enumerate(layer_list))

    elif model_name == 'inception_v3':
        # exclude relu, maxpool
        try:
            layer_list = list(map(lambda name: (name, model[1]._modules.get(name)),
                                  ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3',
                                   'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
                                   'Mixed_7a', 'Mixed_7b', 'Mixed_7c']))
        except TypeError:
            layer_list = list(map(lambda name: (name, model._modules.get(name)),
                                  ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3',
                                   'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
                                   'Mixed_7a', 'Mixed_7b', 'Mixed_7c']))
        return list(enumerate(layer_list))

    elif model_name == 'vit_b_16':
        layer_list = list(map(lambda name: (name, model._modules.get(name)),
                              ['conv_proj']))
        layer_list.extend(
            list(map(lambda name: (name, model._modules.get('encoder')._modules.get('layers')._modules.get(name)),
                     [f'encoder_layer_{layer_idx}' for layer_idx in range(11)])))
        layer_list.extend(map(lambda name: (name, model._modules.get(name)),
                              ['heads']))
        return list(enumerate(layer_list))

    else:
        # model is not supported
        assert False


class InsideCounter:
    def __init__(self, args):
        self.args = args
        self.niter = 0
        self.batch_size_cur = 0

    def step(self, step_size):
        self.niter += 1
        if self.niter == self.args.n_iter:
            self.niter = 0  # clear counter
            self.batch_size_cur += step_size  # next batch


def flatten(X):
    return X.reshape((X.shape[0], -1))


def compute_norm(X_adv, X, norm):
    if norm == 'inf':
        norm = np.inf
    else:
        norm = int(norm)
    return np.linalg.norm(flatten(X_adv) - flatten(X), ord=norm, axis=1)
