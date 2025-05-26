import re
import glob
import os
import random
import numpy as np
import torch.nn
from torchvision import models as tvmodels
from torchvision import transforms
from torchvision import models as tmodels
import timm
from collections import OrderedDict
from utils.registry import Registry
from utils.helper import makedir

base_dir = './surrogate_model/'
base_dir = './surrogate_model/'

IMAGENET_MODEL_NAMES = ['googlenet', 'alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet152',
                        'resnext101', 'wrn50', 'wrn101', 'inception_v3', 'densenet121', 'densenet201',
                        'vgg11_bn', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'shufflenet_v2_x1_0',
                        'mobilenet_v2', 'mobilenet_v3', 'mobilenet_v3_large', 'squeezenet1_1',
                        'senet154', 'mnasnet', 'efficientnet_b7', 'regnet_y_16gf',
                        'convnext_b', 'convnext_l', 'convnext_t', 'convnext_t_tv', 'convnext_l_tv',
                        'vit_b_16', 'vit_s_16', 'vit_b_16_google', 'swin_b', 'swin_t', 'swin_l', 'swin_s', 'xcit_s',
                        'pnasnet5_l', 'deit_s', 'poolformer_s', 'pvt_b', 'tnt_s', 'cait_s',
                        'adv_resnet50_gelu', 'adv_xcit_s',
                        'adv_rawrn_101_2_Peng2023Robust', 'adv_wrn_50_2_Salman2020Do_50_2',
                        'adv_resnet50_Salman2020Do_R50', 'adv_resnet50_Engstrom2019Robustness',
                        'adv_resnet50_Wong2020Fast', 'adv_resnet18_Salman2020Do_R18',
                        'adv_convnext_l_Liu2023Comprehensive_ConvNeXt_L',
                        'adv_convnext_b_Liu2023Comprehensive_ConvNeXt_B',
                        'adv_convnext_l_convstem_Singh2023Revisiting_ConvNeXt_L_ConvStem',
                        'adv_convnext_b_convstem_Singh2023Revisiting_ConvNeXt_B_ConvStem',
                        'adv_convnext_s_convstem_Singh2023Revisiting_ConvNeXt_S_ConvStem',
                        'adv_convnext_t_convstem_Singh2023Revisiting_ConvNeXt_T_ConvStem',
                        'adv_swin_b_Liu2023Comprehensive_Swin_B', 'adv_swin_l_Liu2023Comprehensive_Swin_L',
                        'adv_xcit_m_Debenedetti2022Light_XCiT_M12', 'adv_xcit_l_Debenedetti2022Light_XCiT_L12',
                        'adv_vit_b_convstem_Singh2023Revisiting_ViT_B_ConvStem',
                        'adv_vit_s_convstem_Singh2023Revisiting_ViT_S_ConvStem',
                        'tf2torch_adv_inception_v3', 'tf2torch_ens3_adv_inc_v3',
                        'tf2torch_ens4_adv_inc_v3', 'tf2torch_ens_adv_inc_res_v2',]
CIFAR10_MODEL_NAMES = ['densenet', 'pyramidnet272', 'resnext', 'vgg19_bn', 'wrn', 'gdas', 'adv_wrn_28_10', 'resnet50',
                       'inception_v3']
ALL_MODEL_NAMES = IMAGENET_MODEL_NAMES + CIFAR10_MODEL_NAMES
Registry.register('ALL_MODEL_NAMES')(ALL_MODEL_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_normalize_layer(model, mean, std):
    return torch.nn.Sequential(
        transforms.Normalize(mean=mean, std=std),
        model
    )


def add_resize_layer(model, size, **kwargs):
    return torch.nn.Sequential(
        transforms.Resize(size=size, **kwargs),
        model
    )

def add_crop_layer(model, size, **kwargs):
    return torch.nn.Sequential(
        transforms.CenterCrop(size=size),
        model
    )


def guess_arch_from_path(path, MODEL_NAMES=ALL_MODEL_NAMES):
    """
    Return the name of the model
    """
    candidates = [x for x in MODEL_NAMES if x in path]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # pick the longest one
        return max(candidates, key=len)
    raise ValueError('Not able to guess model name')


def save_checkpoints(exp_dir, rf_model_list, name='checkpoint', epoch=None, sample=False, exist_ok=False):
    makedir(os.path.join(base_dir, exp_dir), exist_ok=exist_ok)
    for i, rf_model in enumerate(rf_model_list):
        state = {
            'epoch': (epoch+1) if epoch is not None else None,
            'sample': (i+1) if sample else None,
        }
        if epoch is not None:
            name_file = f'{name}-{(epoch+1):05}.pt'
        elif sample:
            name_file = f'{name}-{(i+1):05}.pt'
        else:
            name_file = f'{name}.pt'
        state.update({'state_dict': rf_model.state_dict()})
        filepath = os.path.join(base_dir, exp_dir, name_file)
        torch.save(state, filepath)
    return [exp_dir]


def detect_path(models_dir):
    """
    If the path links to a checkpoint instead of a torchvision pretrained model,
    detect the existence of this path.
    """
    if re.match(r"^.+/pretrained/(\w+)$", models_dir) or re.match(r"^.+/IAA/(\w+)$", models_dir):
        return models_dir
    # path to single model
    if re.match('.+\\.pth?(\\.tar)?(\\.pt)?$', models_dir):
        if not os.path.isfile(os.path.join(base_dir, models_dir)):
            raise ValueError('Non-existing path surrogate file passed')
        return models_dir


def list_path(model_path, verbose=True):
    """
    Return the vaild model path wrapped in a list.
    """
    paths_ensembles = [detect_path(x) for x in model_path]
    if paths_ensembles[0] == None:
        # directory of models
        paths_ensembles = glob.glob(f'{os.path.join(base_dir, model_path[0])}/*.ckpt')
        paths_ensembles.extend(glob.glob(f'{os.path.join(base_dir, model_path[0])}/*.pt'))
        paths_ensembles.extend(glob.glob(f'{os.path.join(base_dir, model_path[0])}/*.pth'))
        paths_ensembles.extend(glob.glob(f'{os.path.join(base_dir, model_path[0])}/*.pt.tar'))
        paths_ensembles.extend(glob.glob(f'{os.path.join(base_dir, model_path[0])}/*.pth.tar'))
        paths_ensembles = sorted(paths_ensembles)
        paths_ensembles = [path.replace(base_dir, '') if base_dir in path else path for path in paths_ensembles]

    if verbose:
        print(f'Number of models detected in {model_path}: {len(paths_ensembles)}')
    if len(paths_ensembles) == 0:
        raise ValueError('Empty model ensemble')

    return paths_ensembles


def guess_and_load_model(path_model, norm_layer=True, parallel=True, require_grad=False,
                         load_as_ghost=False, load_as_Styless=False, dict_name=None):
    """
    Guess the model class and load model (with normalization layer) from its path.
    :param path_model: str, path to the model file to load
    :return: pytorch instance of a model
    """

    args = Registry._GLOBAL_REGISTRY['args']

    # load model for NIPS2017 dataset
    if 'ImageNet' in path_model or 'NIPS2017' in path_model:

        if 'pretrained' in path_model:
            # load model checkpoints in './imagenet_ckpt', which are official pre-trained weight from TorchVision.
            a = re.match(r"^.+/pretrained/(\w+)$", path_model)
            if a:
                arch = a.groups()[0]
            else:
                raise ValueError("If a pretrained model installed in tv is desired."
                                 "Make sure the form of path is '/dataset_name/pretrained/model_name'.")
            if arch in IMAGENET_MODEL_NAMES:
                if load_as_ghost:
                    # use the architecture of GhostNet(with skip connection erosion) and load state dict of TV pretrained weight.
                    from surrogate_model.NIPS2017 import GhostNet
                    assert arch in GhostNet.__dict__, f'Unsupported model {arch} for GhostNet.'
                    model = GhostNet.__dict__[arch](pretrained=True)
                elif load_as_Styless:
                    from surrogate_model.NIPS2017 import Styless
                    if arch == 'resnet50':
                        arch = 'styless_resnet50'
                    assert arch in Styless.__dict__, f'Unsupported model {arch} for Styless.'
                    model = Styless.__dict__[arch](pretrained=True)
                else:
                    # use the regular architecture(without skip connection erosion) and load state dict of TV pretrained weight.
                    from surrogate_model import imagenet_models
                    model = imagenet_models.__dict__[arch](pretrained=True)
            else:
                raise ValueError(f'Model {arch} not supported.')
        else:
            # arch_flag = [arch_class in path_model for arch_class in IMAGENET_MODEL_NAMES]
            # assert sum(arch_flag) == 1, "Valid ckpt path must contain one and only one model architecture name"
            # arch = IMAGENET_MODEL_NAMES[int(np.where(np.array(arch_flag)>0)[0])]
            arch = guess_arch_from_path(path_model, MODEL_NAMES=IMAGENET_MODEL_NAMES)
            if 'IAA' in path_model:
                from surrogate_model.NIPS2017.IAA import resnet
                assert arch in resnet.__dict__, 'Unsupported model for IAA.'
                model = resnet.__dict__[arch](pretrained=True)
            else:
                if load_as_ghost:
                    # use the architecture of GhostNet(with skip connection erosion)
                    from surrogate_model.NIPS2017 import GhostNet
                    assert arch in GhostNet.__dict__, f'Unsupported model {arch} for GhostNet.'
                    model = GhostNet.__dict__[arch]()
                elif load_as_Styless:
                    from surrogate_model.NIPS2017 import Styless
                    assert arch in Styless.__dict__, f'Unsupported model {arch} for Styless.'
                    model = Styless.__dict__[arch](pretrained=True)
                else:
                    # use the regular architecture(without skip connection erosion)
                    from surrogate_model import imagenet_models
                    model = imagenet_models.__dict__[arch]()
                # try to load state_dir
                # some models were trained with dataparallel, some not.
                ckpt_dict = torch.load(os.path.join(base_dir, path_model), map_location=DEVICE)
                if dict_name is None:
                    dict_name_flag = [dict_key in ckpt_dict for dict_key in
                                      ['state_dict', 'model_state_dict', 'mean_state_dict', 'sqmean_state_dict', 'model']]
                    dict_name = ['state_dict', 'model_state_dict', 'mean_state_dict', 'sqmean_state_dict', 'model'][int(np.where(np.array(dict_name_flag)>0)[0])]
                try:
                    model.load_state_dict(ckpt_dict[dict_name])
                except RuntimeError:
                    try:
                        new_state_dict = OrderedDict()
                        ckpt_dict[dict_name].pop('mean')
                        ckpt_dict[dict_name].pop('n_models')
                        ckpt_dict[dict_name].pop('sq_mean')
                        ckpt_dict[dict_name].pop('subspace.rank')
                        ckpt_dict[dict_name].pop('subspace.cov_mat_sqrt')
                        for k, v in ckpt_dict[dict_name].items():
                            name = k[11:]  # remove `module.`
                            new_state_dict[name] = v
                        # load params
                        model.load_state_dict(new_state_dict)
                    except KeyError:
                        new_state_dict = OrderedDict()
                        for k, v in ckpt_dict[dict_name].items():
                            name = k[7:]  # remove `module.`
                            if 'last_linear' in name:
                                if 'densenet' not in path_model:
                                    name = 'fc.' + name.split('.')[-1]  # ckpt from DRA name 'fc' layer as 'last_linear'
                                else:
                                    name = 'classifier.' + name.split('.')[-1]
                            new_state_dict[name] = v
                        # load params
                        model.load_state_dict(new_state_dict)

        if norm_layer:
            # add normalization layer
            if 'inc' in path_model:
                model = add_normalize_layer(model=model, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                model = add_crop_layer(model=model, size=(299, 299))
                model = add_resize_layer(model=model, size=(299, 299))
            # elif 'pnasnet' in path_model:
            #     model = add_normalize_layer(model=model, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            #     model = add_resize_layer(model=model, size=(331, 331))
            elif 'adv' in path_model and 'tf' not in path_model:
                if 'adv_resnet50_gelu' in path_model:
                    model = add_normalize_layer(model=model, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                model = add_crop_layer(model=model, size=(224, 224))
                model = add_resize_layer(model=model, size=(256, 256))
            elif 'timm' in str(model.__class__) and 'adv' not in path_model:
                data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
                all_transforms = timm.data.create_transform(**data_cfg)
                all_transforms.transforms.pop(-2)
                for trans_op in all_transforms.transforms[::-1]:
                    model = torch.nn.Sequential(trans_op, model)
            else:
                model = add_normalize_layer(model=model, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                model = add_crop_layer(model=model, size=(224, 224))
                model = add_resize_layer(model=model, size=(256, 256))

    # load model for CIFAR10 dataset
    elif 'CIFAR10' in path_model:
        if 'pretrained' in path_model:
            # valid arches: 'densenet'(DenseNet-BC), 'pyramidnet272'(PyramidNet272), 'resnext'(ResNeXt29),
            # 'vgg19_bn'(vgg19_bn), 'wrn'(WRN-28-10)
            a = re.match(r"^.+/pretrained/(\w+)$", path_model)
            if a:
                arch = a.groups()[0]
            else:
                raise ValueError("If a pretrained model installed in tv is desired, "
                                 "make sure the form of path is `/dataset_name/pretrained/model_name`.")
            assert arch in CIFAR10_MODEL_NAMES, f'Model {arch} not supported.'
            if load_as_ghost:
                # use the architecture of GhostNet(with skip connection erosion) and load state dict of TV pretrained weight.
                from surrogate_model.CIFAR10 import GhostNet
                assert arch in GhostNet.__dict__, f'Unsupported model {arch} for GhostNet.'
                model = GhostNet.__dict__[arch](pretrained=True, num_classes=10)
            else:
                # use the regular architecture(without skip connection erosion) and load state dict of TV pretrained weight.
                from surrogate_model import cifar_models
                model = cifar_models.__dict__[arch](pretrained=True, num_classes=10)
        else:
            # arch_flag = [arch_class in path_model for arch_class in CIFAR10_MODEL_NAMES]
            # assert sum(arch_flag) == 1, "Valid ckpt path must contain one and only one model architecture name"
            # arch = CIFAR10_MODEL_NAMES[int(np.where(np.array(arch_flag) > 0)[0])]
            arch = guess_arch_from_path(path_model, MODEL_NAMES=CIFAR10_MODEL_NAMES)
            if load_as_ghost:
                # use the architecture of GhostNet(with skip connection erosion)
                from surrogate_model.CIFAR10 import GhostNet
                assert arch in GhostNet.__dict__, f'Unsupported model {arch} for GhostNet.'
                model = GhostNet.__dict__[arch](num_classes=10)
            else:
                # use the regular architecture(without skip connection erosion)
                from surrogate_model import cifar_models
                model = cifar_models.__dict__[arch](num_classes=10)
            # try to load state_dir
            # some models were trained with dataparallel, some not.
            ckpt_dict = torch.load(os.path.join(base_dir, path_model), map_location=DEVICE)
            if dict_name is None:
                dict_name_flag = [dict_key in ckpt_dict for dict_key in
                                  ['state_dict', 'model_state_dict', 'mean_state_dict', 'sqmean_state_dict']]
                dict_name = ['state_dict', 'model_state_dict'][int(np.where(np.array(dict_name_flag) > 0)[0])]
            try:
                model.load_state_dict(ckpt_dict[dict_name])
            except RuntimeError:
                try:
                    new_state_dict = OrderedDict()
                    ckpt_dict[dict_name].pop('mean')
                    ckpt_dict[dict_name].pop('n_models')
                    ckpt_dict[dict_name].pop('sq_mean')
                    ckpt_dict[dict_name].pop('subspace.rank')
                    ckpt_dict[dict_name].pop('subspace.cov_mat_sqrt')
                    for k, v in ckpt_dict[dict_name].items():
                        name = k[11:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model.load_state_dict(new_state_dict)
                except KeyError:
                    new_state_dict = OrderedDict()
                    for k, v in ckpt_dict[dict_name].items():
                        name = k[7:]  # remove `module.`
                        if 'last_linear' in name:
                            name = 'fc.' + name.split('.')[-1]  # ckpt from DRA name 'fc' layer as 'last_linear'
                        new_state_dict[name] = v
                    # load params
                    model.load_state_dict(new_state_dict)

        if norm_layer:
            # add normalization layer
            model = add_normalize_layer(model=model, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    else:
        raise ValueError('Invalid dataset.')

    if 'LGV' in path_model or 'Bayesian_Attack' in path_model:
        # LGV and Bayesian_attack require loading too many models.
        # To avoid OOM in the main(0) GPU, we allocate models on other GPUs.
        if torch.cuda.memory_allocated(device=1) / torch.cuda.get_device_properties(1).total_memory < 0.1:
            model.to('cuda:1')
        elif torch.cuda.memory_allocated(device=2) / torch.cuda.get_device_properties(2).total_memory < 0.1:
            model.to('cuda:2')
        elif torch.cuda.memory_allocated(device=3) / torch.cuda.get_device_properties(3).total_memory < 0.1:
            model.to('cuda:3')
        elif torch.cuda.memory_allocated(device=4) / torch.cuda.get_device_properties(4).total_memory < 0.1:
            model.to('cuda:4')
        elif torch.cuda.memory_allocated(device=5) / torch.cuda.get_device_properties(5).total_memory < 0.1:
            model.to('cuda:5')
        elif torch.cuda.memory_allocated(device=6) / torch.cuda.get_device_properties(6).total_memory < 0.1:
            model.to('cuda:6')
        elif torch.cuda.memory_allocated(device=7) / torch.cuda.get_device_properties(7).total_memory < 0.1:
            model.to('cuda:7')
        else:
            print('OOM')
    else:
        model.to(DEVICE)
    if parallel:
        model = torch.nn.DataParallel(model)
    model.eval()
    if not require_grad:
        for param in model.parameters():
            param.requires_grad = False

    return model


def build_model(model_path, n_iter, n_ensemble, shuffle=False):

    args = Registry._GLOBAL_REGISTRY['args']
    if args.source_model_refinement is not '':
        # model refinement
        assert len(model_path) == 1, "The current version only support refining one source model, multi-model " \
                                     "refinement will be complemented in the future."
        # rf_model = guess_and_load_model(list_path(model_path)[0], norm_layer=False, parallel=False, require_grad=True, load_as_ghost=False)  # load to-be-refined model without add normalization layer
        from model_refinement import build_model_refinement
        model_refiner = build_model_refinement(args.source_model_refinement)
        print(f"Build model refinement {args.source_model_refinement} complete.")
        final_path = model_refiner(args, model_path)
    else:
        final_path = model_path  # do not refine the model from `source_model_path`

    paths_ensembles = list_path(final_path)
    if shuffle:
        random.shuffle(paths_ensembles)  # shuffle

    max_nb_models_used = n_iter * n_ensemble
    if len(paths_ensembles) > max_nb_models_used:
        paths_ensembles = paths_ensembles[:max_nb_models_used]

    ensemble_list = []
    for i, path_model in enumerate(paths_ensembles):
        assert len(paths_ensembles) >= n_ensemble, "The num of ensemble models couldn't be smaller than args.n_ensemble"
        if len(ensemble_list) == 0:
            # avoid IndexError at first iteration
            ensemble_list.append([path_model, ])
        elif len(ensemble_list[-1]) >= n_ensemble:
            ensemble_list.append([path_model, ])
        else:
            ensemble_list[-1].append(path_model)

    list_ensemble_models = []
    for i, ensemble_path in enumerate(ensemble_list):
        if len(ensemble_path) == 1:
            model = guess_and_load_model(ensemble_path[0], load_as_ghost=args.ghost_attack, load_as_Styless=args.Styless_attack)
            list_ensemble_models.append([model])
        else:
            models_to_ensemble = []
            for j, path_model in enumerate(ensemble_path):
                models_to_ensemble.append(guess_and_load_model(path_model, load_as_ghost=args.ghost_attack,load_as_Styless=args.Styless_attack))
            list_ensemble_models.append(models_to_ensemble)

    return list_ensemble_models