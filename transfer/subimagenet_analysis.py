import argparse
import importlib
import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import pandas as pd
import random
import re
import time
from itertools import cycle
from tqdm import tqdm
import sys
from surrogate_model.utils import build_model, guess_and_load_model
from data.data_loader import build_dataloader
from loss_function import build_loss_function
from input_transformation import build_input_transformation
from gradient_calculation import build_grad_calculator
from update_dir_calculation import build_update_dir_calculator
from utils.helper import update_and_clip, iter_eval_attack, makedir, compute_norm
from utils.registry import Registry
from tools.flatness import flatness_visualization
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import Counter
import palettable
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_attack(args, target_models, method):
    class_correct_sum = {}
    for class_i in range(1000):
        class_correct_sum['{}'.format(class_i)] = 0

    if 'NIPS2017' in args.source_model_path[0]:
        trans = T.Compose([])
        # trans = T.Compose([]) if args.targeted else T.Compose([T.Resize((256, 256)), T.CenterCrop((224, 224))])
    elif 'CIFAR10' in args.source_model_path[0]:
        trans = T.Compose([])

    for dataset_i in range(5):
        args.save_dir = f"./adv_imgs/analysis/subimagenet_NIPS2017_UT_INF_RESNET50/{dataset_i}/resnet50/{method}"

        true_labels = torch.from_numpy(np.load(args.save_dir + '/true_labels.npy')).long()
        target_labels = torch.from_numpy(np.load(args.save_dir + '/target_labels.npy')).long()
        target = true_labels if not args.targeted else target_labels

        advfile_ls = os.listdir(args.save_dir)
        img_num = 0
        for advfile_ind in range(len(advfile_ls)-2):    # minus 2 labels files
            adv_batch = torch.from_numpy(np.load(args.save_dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255

            img_num += adv_batch.shape[0]
            labels = target[advfile_ind * adv_batch.shape[0]: advfile_ind * adv_batch.shape[0] + adv_batch.shape[0]]
            inputs, labels = adv_batch.clone().to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                for j, target_model in tqdm(enumerate(target_models)):
                    labels_now = labels + 1 if 'tf2torch' in target_model else labels
                    target_model = guess_and_load_model(target_model)
                    model_device = next(target_model.parameters()).device
                    target_model.to(DEVICE)
                    pred = torch.argmax(target_model(trans(inputs)), dim=1)
                    for true_i in range(adv_batch.shape[0]):
                        if pred[true_i] != labels_now[true_i]:
                            class_correct_sum['{}'.format(labels_now[true_i])] += 1
                    target_model.to(model_device)
                    torch.cuda.empty_cache()
                    del target_model
    categories = list(class_correct_sum.keys())
    counts = [class_correct_sum[category] for category in categories]
    categories = [int(x) for x in categories]
    sorted_indices = sorted(range(len(categories)), key=lambda k: categories[k])
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_categories)), sorted_counts, tick_label=sorted_categories)
    plt.title('Correctly Recognized Image Counts by Category')
    plt.xlabel('Category')
    plt.ylabel('Correctly Recognized Counts')
    plt.xticks([])  # 隐藏X轴所有标签
    plt.show()

    counts = Counter(class_correct_sum.values())

    print("出现次数统计：")
    for count, num_keys in counts.items():
        print(f"{count}次出现的键的个数：{num_keys}")

def eval_attack_sub(args, target_models, method, subset, start):

    if 'NIPS2017' in args.source_model_path[0]:
        trans = T.Compose([])
        # trans = T.Compose([]) if args.targeted else T.Compose([T.Resize((256, 256)), T.CenterCrop((224, 224))])
    elif 'CIFAR10' in args.source_model_path[0]:
        trans = T.Compose([])

    acc_list = []
    for dataset_i in range(5):
        args.save_dir = f"./adv_imgs/analysis/subimagenet_NIPS2017_UT_INF_RESNET50/{dataset_i}/resnet50/{method}"

        true_labels = torch.from_numpy(np.load(args.save_dir + '/true_labels.npy')).long()
        target_labels = torch.from_numpy(np.load(args.save_dir + '/target_labels.npy')).long()

        advfile_ls = os.listdir(args.save_dir)
        att_suc = np.zeros((len(target_models),))  # initialize the attack success rate matrix
        img_num = 0
        advs = []
        for advfile_ind in range(len(advfile_ls)-2):    # minus 2 labels files
            adv_batch = torch.from_numpy(np.load(args.save_dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255
            advs.append(adv_batch)
        advs = torch.cat(advs)
        labels_batch = true_labels[start : start + subset]
        inputs, labels = advs[start:start+subset].clone().to(DEVICE), labels_batch.to(DEVICE)
        with torch.no_grad():
            for j, target_model in tqdm(enumerate(target_models)):
                labels_now = labels + 1 if 'tf2torch' in target_model else labels
                target_model = guess_and_load_model(target_model)
                model_device = next(target_model.parameters()).device
                target_model.to(DEVICE)
                att_suc[j] += sum(torch.argmax(target_model(trans(inputs)), dim=1) != labels_now).cpu().numpy()
                target_model.to(model_device)
        att_suc = 1 - att_suc / subset if args.targeted else att_suc / subset
        acc_list.append(att_suc.sum()/len(target_models))

    return acc_list

def eval_attack_new_std(args, adv_folder, target_models, method, subset, start):

    if 'NIPS2017' in args.source_model_path[0]:
        trans = T.Compose([])
        # trans = T.Compose([]) if args.targeted else T.Compose([T.Resize((256, 256)), T.CenterCrop((224, 224))])
    elif 'CIFAR10' in args.source_model_path[0]:
        trans = T.Compose([])

    acc_list = []
    for dataset_i in range(5):
        success_one_data = np.zeros((1000,))

        args.save_dir = f"./adv_imgs/analysis/{adv_folder}/{dataset_i}/resnet50/{method}"

        true_labels = torch.from_numpy(np.load(args.save_dir + '/true_labels.npy')).long()
        target_labels = torch.from_numpy(np.load(args.save_dir + '/target_labels.npy')).long()

        advfile_ls = os.listdir(args.save_dir)
        img_num = 0
        for advfile_ind in range(len(advfile_ls)-2):    # minus 2 labels files
            adv_batch = torch.from_numpy(np.load(args.save_dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255
            img_num += adv_batch.shape[0]
            labels = true_labels[advfile_ind * adv_batch.shape[0]: advfile_ind * adv_batch.shape[0] + adv_batch.shape[0]]
            targets = target_labels[advfile_ind * adv_batch.shape[0]: advfile_ind * adv_batch.shape[0] + adv_batch.shape[0]]
            inputs, labels, targets = adv_batch.clone().to(DEVICE), labels.to(DEVICE), targets.to(DEVICE)
            with torch.no_grad():
                for j, target_model in tqdm(enumerate(target_models)):
                    labels_now = labels + 1 if 'tf2torch' in target_model else labels
                    targets_now = targets + 1 if 'tf2torch' in target_model else targets
                    target_model = guess_and_load_model(target_model)
                    model_device = next(target_model.parameters()).device
                    target_model.to(DEVICE)
                    pred = torch.argmax(target_model(trans(inputs)), dim=1)
                    asr = (pred != labels_now).int() if '_UT_' in adv_folder else (pred == targets_now).int()
                    success_one_data[advfile_ind * adv_batch.shape[0]: advfile_ind * adv_batch.shape[0] + adv_batch.shape[0]] += asr.cpu().numpy()
                    target_model.to(model_device)
        acc_list.append(success_one_data.reshape((1,success_one_data.shape[0])))
    return np.concatenate(acc_list)

cudnn.benchmark = True
cudnn.deterministic = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# parse args
parser = argparse.ArgumentParser(description="transfer-based blackbox adversarial benchmark")
parser.add_argument('--csv-export-path', type=str, default=None,
                    help="Path to CSV where to export data about target.")
parser.add_argument('--json-path', type=str, default='./config/NIPS2017/untargeted/l_inf/I-FGSM_subimagenet.json',
                    help="Path for json file providing configurations of current method.")
parser.add_argument('--source-model-path', nargs='+',
                    help="Path to directory containing all the models file of the ensemble model.Also support single path to a model file."
                         "If a pretrained model installed in tv is desired, make sure the form of path is `/dataset_name/pretrained/model_name`."
                         "If loading ckpt is desired, make sure the path contains `dataset_name'&'model_arch_name`."
                         "Please check all supported model architectures in ALL_MODEL_NAMES of `surrogate_model/utils`")
parser.add_argument("--target-model-path", nargs='+',
                    help="Path to the target models."
                         "If a pretrained model installed in tv is desired, make sure the form of path is `/dataset_name/pretrained/model_name `")
parser.add_argument("--rfmodel-dir", type=str,
                    help="Path to the store the refined surrogate models.")
parser.add_argument('--n-iter', type=int, default=50,
                    help="Number of iterations to perform.")
parser.add_argument("--norm-type", choices=['2', 'inf'],
                    help="Type of L-norm to use.")
parser.add_argument("--epsilon", type=float,
                    help="Max L-norm of the perturbation")
parser.add_argument("--norm-step", type=float,
                    help="Max norm at each step.")
parser.add_argument("--decay-factor", type=float,
                    help="Decay factor for momentum.")
parser.add_argument('--n-ensemble', type=int, default=1,
                    help="Number of samples to ensemble for each iteration(Default: 1)."
                         "If multi source-models combine with n-ensemble=1, conduct 'longitudinal ensemble'")
parser.add_argument('--emsemble-type', type=str, default=None,
                    help="The way to ensemble multiple surrogate models: ensemble on logit/loss.")
parser.add_argument('--imagenet-sub', type=int, default=None,
                    help="Whether to use a subset of 'ImageNet2012 Val' as dataset. This value can be set to [0,1,2,3,4],"
                         "which means 5 subsets are officially provided by BlackBoxBench.")
parser.add_argument('--shuffle', action='store_true',
                    help="Random order of models vs sequential order of the MCMC (default)")
parser.add_argument('--targeted', action='store_true',
                    help="Achieve targeted attack.")
parser.add_argument("--seed", type=int, default=None,
                    help="Set random seed")
parser.add_argument("--image-size", type=int, default=None,
                    help="Size of transformed image. If `image_size` is given, study the influence of generated adversarial image size.")
parser.add_argument("--num-workers", type=int, default=0,
                    help="Set num_workers of dataloader")
parser.add_argument("--batch-size", type=int, default=100,
                    help="Batch size. Try a lower value if out of memory.")
parser.add_argument('--save-dir', type=str, default=None,
                    help="Path to save adversarial images.")
parser.add_argument('--save-iter-dir', type=str, default=None,
                    help="Path to save adversarial images at every [save-fre] iterations.")
parser.add_argument('--full-imagenet-dir', type=str, default='../imagenet2012',
                    help="Path to the full ImageNet2012 dataset.")
parser.add_argument('--as-baseline', action='store_true',
                    help="The flag of whether the current method is for generating baseline adversarial samples as the guidance of ILA")
parser.add_argument('--n-var-sample', type=int, default=None,
                    help="The number of images sampled around the current adversarial samples to calculate variance of gradient."
                         "If the value of `n_var_sample` is not None, the calculation of gradient variance will be done!")
parser.add_argument('--calcu-ori-att-suc', action='store_true',
                    help="Calculate original attack success rate.")
parser.add_argument("--random-start-epsilon", type=float, default=0.03,
                    help="The size of a random start of perturbation.")
parser.add_argument("--save-fre", type=int, default=50,
                    help="Save generated adversarial example every [save-fre] iterations")
parser.add_argument("--fig-name", type=str)
parser.add_argument("--start", type=int)
parser.add_argument("--subset", type=int)

# transferability improvements
parser.add_argument('--loss-function', type=str, default='cross_entropy',
                    help="Loss function compatible with each method (default: cross entropy).")
parser.add_argument('--backpropagation', type=str, default='nonlinear',
                    help="choices=['nonlinear', 'linear', 'skip_gradient']")
parser.add_argument('--grad-calculation', type=str, default='general',
                    help="")  # choices=['general'],
parser.add_argument('--update-dir-calculation', type=str, default='sgd',
                    help="")  # choices=['sgd', 'momentum', 'var_tuning']
parser.add_argument('--input-transformation', type=str, default='',
                    help="Input transformation compatible with each method (default: None)"
                         "choices=['admix(strength=0.2, n_samples=3)|SI(n_copies=5)|DI(in_size=224, out_size=256)', 'DI(in_size=32, out_size=40)']")
parser.add_argument("--source-model-refinement", type=str, default='',
                    help="The tricks to refine source model load from the `source_model_path`."
                         "choices=['sample_from_isotropic(std=0.005, n_models=10)', "
                         "'stochastic_weight_collecting(collect=False)', 'stochastic_weight_averaging(collect=False)', "
                         "'stochastic_weight_averaging(collect=False)|sample_from_isotropic()', "
                         "'sample_from_stochastic_weight_averaging_gaussian'],")
parser.add_argument('--ghost-attack', action='store_true',
                    help="Load each model as a Ghost network (default: no model alteration)")
parser.add_argument('--Styless_attack', action='store_true',
                    help="Load each model as a Styless network (default: no model alteration)")
parser.add_argument("--random-start", action='store_true',
                    help="Random start in PGD")
parser.add_argument("--bsl-adv-img-path", type=str, default=None,
                    help="The path of adversarial samples generated by baseline attack method. "
                         "This path is required in `Intermediate Level Attack` method.")


# update args in json file
args = parser.parse_args()
with open(args.json_path, 'r') as f:
    configs = json.load(f)
args.__dict__.update({k:v for k,v in configs.items()})
Registry.register('args')(args)
print(args)


# set random seed
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


start_time = time.perf_counter()
print("____________________PREPARING OUT OF ITERATION____________________")
# load source models

source_models = build_model(args.source_model_path, args.n_iter, args.n_ensemble, shuffle=args.shuffle)
source_size = len(source_models)*args.n_ensemble
source_models = cycle(source_models)    # create a infinite source model iterator
Registry.register('source_models')(source_models)
Registry.register('source_size')(source_size)
print(f"Source models load complete from {args.source_model_path}. Totally {source_size} source models.")

# load target models
# target_models = [guess_and_load_model(target_path) for target_path in args.target_model_path]
target_models = [target_path for target_path in args.target_model_path]
print(f"Target models load complete from {args.target_model_path}. Totally {len(args.target_model_path)} target models.")

csv_list = ['subimagenet_NIPS2017_UT_INF_RESNET50.csv', 'subimagenet_NIPS2017_UT_2_RESNET50.csv',
            'subimagenet_NIPS2017_T_INF_RESNET50.csv', 'subimagenet_NIPS2017_T_2_RESNET50.csv']
adv_img_list = ['subimagenet_NIPS2017_UT_INF_RESNET50', 'subimagenet_NIPS2017_UT_2_RESNET50',
            'subimagenet_NIPS2017_T_INF_RESNET50','subimagenet_NIPS2017_T_2_RESNET50',]
setting = ['untargeted, $\ell_{\infty}$', 'untargeted, $\ell_{2}$', 'targeted, $\ell_{\infty}$', 'targeted, $\ell_{2}$']
colors = palettable.cartocolors.qualitative.Safe_10.mpl_colors[:len(csv_list)]
markers = ['o', '*', 'v', 's', 'd', '+', '2', 'x']
set2marker = dict(zip(setting, markers))
set2color = dict(zip(setting, colors))
# fig, ax = plt.subplots(figsize=(9, 4.3))

# for adv_folder in adv_img_list:
#     if '_UT_' in adv_folder:
#         methods = {'I-FGSM': 'I-FGSM', 'PGD': 'random_start', 'TI-FGSM': 'TI', 'DI2-FGSM': 'DI2-FGSM',
#                    'SI-FGSM': 'SI', 'Admix': 'admix', 'SIA': 'SIA', 'MI-FGSM': 'MI-FGSM',
#                    'NI-FGSM': 'NI', 'LinBP': 'LinBP', 'SGM': 'SGM', 'PI-FGSM': 'PI', 'VT': 'VT',
#                    'RAP': 'RAP', 'PGN': 'PGN', 'ILA': 'ILA', 'FIA': 'FIA', 'NAA': 'NAA',
#                    'Ens-logit': 'ens_logit_I-FGSM', 'Ens-loss': 'ens_loss_I-FGSM',
#                    'Ens-longi.': 'ens_longitudinal_I-FGSM',
#                    'GhostNet': 'GhostNet', 'RD': 'RD', 'DRA': 'DRA', 'IAA': 'IAA',
#                    'LGV': 'LGV', 'SWA': 'SWA', 'Bayesian': 'Bayesian_attack', 'CWA': 'CWA',
#                    'AdaEA':'AdaEA'}
#     else:
#         methods = {'I-FGSM': 'I-FGSM', 'PGD': 'random_start', 'TI-FGSM': 'TI', 'DI2-FGSM': 'DI2-FGSM',
#                    'SI-FGSM': 'SI', 'Admix': 'admix', 'SIA': 'SIA', 'MI-FGSM': 'MI-FGSM',
#                    'NI-FGSM': 'NI', 'LinBP': 'LinBP', 'SGM': 'SGM', 'PI-FGSM': 'PI', 'VT': 'VT',
#                    'RAP': 'RAP', 'PGN': 'PGN', 'ILA': 'ILA',
#                    'Ens-logit': 'ens_logit_I-FGSM', 'Ens-loss': 'ens_loss_I-FGSM',
#                    'Ens-longi.': 'ens_longitudinal_I-FGSM',
#                    'GhostNet': 'GhostNet', 'RD': 'RD', 'DRA': 'DRA', 'IAA': 'IAA',
#                    'LGV': 'LGV', 'SWA': 'SWA', 'Bayesian': 'Bayesian_attack', 'CWA': 'CWA',
#                    'AdaEA':'AdaEA'}
#     att_acc_all_methods = {}
#     for methods_ind in range(len(methods)):
#         method = list(methods.keys())[methods_ind]
#         att_suc = eval_attack_new_std(args, adv_folder, target_models, methods[method], subset=args.subset, start=args.start)
#         att_acc_all_methods[method] = att_suc
#     with open(f"{adv_folder}.pkl", "wb") as tf:
#         pickle.dump(att_acc_all_methods,tf)

# 画图
methods_sorted = {'I-FGSM': 'I-FGSM', 'PGD': 'random_start', 'TI-FGSM': 'TI', 'DI2-FGSM': 'DI2-FGSM',
           'SI-FGSM': 'SI', 'Admix': 'admix', 'SIA': 'SIA', 'MI-FGSM': 'MI-FGSM',
           'NI-FGSM': 'NI', 'LinBP': 'LinBP', 'SGM': 'SGM','PI-FGSM': 'PI', 'VT': 'VT',
           'RAP': 'RAP', 'PGN': 'PGN', 'ILA': 'ILA', 'FIA': 'FIA', 'NAA': 'NAA',
           'GhostNet': 'GhostNet', 'RD': 'RD', 'DRA': 'DRA', 'IAA':'IAA',
           'LGV': 'LGV', 'SWA': 'SWA', 'Bayesian': 'Bayesian_attack',
           'Ens-logit': 'ens_logit_I-FGSM','Ens-loss': 'ens_loss_I-FGSM', 'Ens-longi.': 'ens_longitudinal_I-FGSM',
           'AdaEA':'AdaEA', 'CWA':'CWA'}
for idx, adv_folder in enumerate(adv_img_list):
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    setting_i = setting[idx]
    colors = set2color[setting_i]
    with open(f"{adv_folder}_all.pkl", "rb") as tf:
        att_acc_all_methods = pickle.load(tf)

    # means = np.array([np.mean(results) for results in att_acc_all_methods.values()])
    # # conf_interval = np.array([stats.sem(results) * stats.t.ppf((1 + 0.95) / 2., len(data_sorted)-1)
    # #                           for results in data_sorted.values()])
    # conf_interval = np.array([np.std(results) for results in att_acc_all_methods.values()])
    means = []
    conf_interval = []
    for plot_i in range(len(methods_sorted)):
        try:
            results = att_acc_all_methods[list(methods_sorted.keys())[plot_i]]/17

            # 均值的均值 方差的均值
            mean_method = np.zeros((1000,))
            std_method = np.zeros((1000,))
            for i in range(1000):
                mean_method[i] = np.mean(results[:, i])
                std_method[i] = np.std(results[:, i])
            means.append(np.mean(mean_method))
            conf_interval.append(np.mean(std_method))

            # 所有的均值 所有的方差
            # means.append(np.mean(results.reshape(5000,)))
            # conf_interval.append(np.std(results.reshape(5000,)))

            # 所有的均值 均值的方差
            # mean_method = np.zeros((1000,))
            # std_method = np.zeros((1000,))
            # for i in range(1000):
            #     mean_method[i] = np.mean(results[:, i])
            # means.append(np.mean(mean_method))
            # conf_interval.append(np.std(mean_method))

            # 所有的均值 方差的方差
            # mean_method = np.zeros((1000,))
            # std_method = np.zeros((1000,))
            # for i in range(1000):
            #     mean_method[i] = np.mean(results[:, i])
            #     std_method[i] = np.std(results[:, i])
            # means.append(np.mean(mean_method))
            # conf_interval.append(np.std(std_method))
        except KeyError:
            means.append(np.nan)
            conf_interval.append(np.nan)
    means = np.array(means)
    conf_interval = np.array(conf_interval)

    x = np.arange(len(methods_sorted))
    labels = list(methods_sorted.keys())

    # plt.errorbar(x, means, yerr=conf_interval, color=colors, fmt=set2marker[setting_i],
    #              capsize=5, markersize=6 if set2marker[setting_i] != '*' else 7, label=setting_i)
    plt.plot(x, means, '-',color=colors, alpha=0.7, marker=set2marker[setting_i],
             markersize=6 if set2marker[setting_i]!='*' else 7, label=setting_i)  # '-o'表示带圆圈的折线图
    plt.fill_between(x, means - conf_interval, means + conf_interval, color=colors, alpha=0.2)
    plt.xticks(ticks=x, labels=labels, rotation=45, ha='right')

    # ax.set_xlabel('Method')
    # ax.set_ylabel('Average ASR on various target models', fontsize=12)
    plt.tick_params(axis='y', labelsize=8)
    ax.set_ylim(0,1)
    # plt.grid(alpha=0.5, linewidth=0.4)
    ax.legend(ncol=1, loc='upper center', fontsize=13)
    # plt.title('Averaged per-class mean & std of ASR for transfer-based attacks')
    plt.tight_layout()
    # plt.savefig(args.fig_name, format='png', transparent=True, dpi=600)
    plt.show()