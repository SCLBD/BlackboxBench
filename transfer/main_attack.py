import argparse
import importlib
import json
import os
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
from utils.helper import update_and_clip, eval_attack, iter_eval_attack, makedir, compute_norm, calculate_accuracy
from utils.registry import Registry
from tools.flatness import flatness_visualization



cudnn.benchmark = True
cudnn.deterministic = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# parse args
parser = argparse.ArgumentParser(description="transfer-based blackbox adversarial benchmark")
parser.add_argument('--csv-export-path', type=str, default=None,
                    help="Path to CSV where to export data about target.")
parser.add_argument('--json-path', type=str, required=True,
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
parser.add_argument('--shuffle', action='store_true',
                    help="Random order of models vs sequential order of (ensembled) surrogate models")
parser.add_argument('--targeted', action='store_true',
                    help="Achieve targeted attack or not.")
parser.add_argument("--seed", type=int, default=None,
                    help="Set random seed")
parser.add_argument("--image-size", type=int, default=None,
                    help="Size of transformed image. If `image_size` is given, study the influence of generated adversarial image size.")
parser.add_argument("--num-workers", type=int, default=4,
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

# transferability improvements
parser.add_argument('--loss-function', type=str, default='cross_entropy',
                    help="Loss function compatible with each method (default: cross entropy).")
parser.add_argument('--backpropagation', type=str, default='nonlinear',
                    help="choices=['nonlinear', 'linear', 'skip_gradient']")
parser.add_argument('--grad-calculation', type=str, default='general',
                    help="")
parser.add_argument('--update-dir-calculation', type=str, default='sgd',
                    help="")
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
print(f"Source models load complete from {args.source_model_path}. Totally {source_size} source models.")

# load target models
target_models = [guess_and_load_model(target_path) for target_path in args.target_model_path]
print(f"Target models load complete from {args.target_model_path}. Totally {len(args.target_model_path)} target models.")

# load data
data_loader = build_dataloader(args)
Registry.register('data_loader')(data_loader)
print("Data loader complete.")

# build loss function
loss_func = build_loss_function(args.loss_function)
print(f"Build loss function {args.loss_function} complete.")

# build input transformation
input_trans_func = build_input_transformation(args.input_transformation)
print(f"Build input transformation {args.input_transformation} complete.")

# build gradient calculator
grad_calculator = build_grad_calculator(args.grad_calculation)
print(f"Build {args.grad_calculation} gradient calculator complete.")

# build adversarial direction updator
update_dir_calculator = build_update_dir_calculator(args.update_dir_calculation)
print(f"Build {args.update_dir_calculation} update dir calculator complete.")


if args.calcu_ori_att_suc:
    ori_att_suc = calculate_accuracy(data_loader, target_models)

print('____________________GENERATING ADVERSARIAL EXAMPLES____________________')
if args.as_baseline:
    makedir(args.bsl_adv_img_path)
makedir(args.save_dir)
makedir(args.save_iter_dir)
true_label_ls, target_label_ls = [], []
all_ori, all_adv = [], []
for ind, (ori_img, true_label, target_label) in enumerate(data_loader):
    ori_img, true_label, target_label = ori_img.to(DEVICE), true_label.to(DEVICE), target_label.to(DEVICE)
    img = ori_img.clone()
    true_label_ls.append(true_label), target_label_ls.append(target_label)

    grad_accumulate = torch.zeros(ori_img.size()).cuda()
    grad_last = torch.zeros(ori_img.size()).cuda()
    grad_var_last = torch.zeros(ori_img.size()).cuda()
    save_every_iter = []
    save_every_iter.append(torch.round(ori_img.data * 255).cpu().numpy())
    for i in tqdm(range(args.n_iter), postfix={"batch": ind}):

        # random start
        adv_img = img + img.new(img.size()).uniform_(-args.random_start_epsilon, args.random_start_epsilon) if args.random_start else img
        adv_img.requires_grad_(True)

        # calculate current gradient and variance of gradient
        ensemble_models = next(source_models)
        gradient = grad_calculator(args, i, adv_img, true_label, target_label,
                                   grad_accumulate, grad_last,
                                   input_trans_func, ensemble_models, loss_func)
        if args.n_var_sample:
            grad_var = Registry.lookup('get_variance')()(args, adv_img, true_label, target_label,
                                                         grad_accumulate, grad_last, gradient,
                                                         input_trans_func, ensemble_models, loss_func)
        else:
            grad_var = torch.zeros(ori_img.size()).cuda()
        grad_last = gradient

        # calculate update direction
        update_dir, grad_accumulate = update_dir_calculator(args, gradient, grad_accumulate, grad_var_last)
        grad_var_last = grad_var

        # update adversarial images
        img = update_and_clip(args, img, ori_img, update_dir)

        if args.save_iter_dir is not None and (i+1) % args.save_fre == 0:
            save_every_iter.append(torch.round(img.data * 255).cpu().numpy())

    if args.save_iter_dir is not None:
        all_iter = np.stack(save_every_iter)
        np.save(args.save_iter_dir + '/batch_{}_iters.npy'.format(ind), all_iter)

    flatness_visualization(args, ind, ori_img.data, img.data, true_label, target_label, ensemble_models, loss_func)
    np.save(args.save_dir + '/batch_{}.npy'.format(ind), torch.round(img.data * 255).cpu().numpy())    # save images
    if args.as_baseline:
        np.save(args.bsl_adv_img_path + '/batch_{}.npy'.format(ind), img.data.cpu().numpy())
    print('batch_{}.npy saved'.format(ind))
    all_ori.append(ori_img.cpu().numpy())
    all_adv.append(img.cpu().numpy())

all_ori = np.concatenate(all_ori)
all_adv = np.concatenate(all_adv)
lpnorm = compute_norm(X_adv=all_adv, X=all_ori, norm=args.norm_type)    # calculate L_p norm
true_label_ls = torch.cat(true_label_ls)
target_label_ls = torch.cat(target_label_ls)
np.save(args.save_dir + '/true_labels.npy', true_label_ls.cpu().numpy())
np.save(args.save_dir + '/target_labels.npy', target_label_ls.cpu().numpy())   # save labels
print('all batches saved')
if args.save_iter_dir is not None:
    iter_eval_attack(args, target_models)
att_suc = eval_attack(args, target_models)    # evaluation after saving

end_time = time.perf_counter()

if args.csv_export_path:
    dict_metrics = dict()
    dict_metrics.update({'json_path': args.json_path})  # name of algorithm
    for k, target_model_p in enumerate(args.target_model_path):
        dict_metrics.update({f"ASR_{target_model_p}": att_suc[k]})   # ASR for every target model
    dict_metrics.update({
        'surrogate_model': args.source_model_path,
        'surrogate_size': source_size,
        'targeted': args.targeted,
        'norm_type': args.norm_type,
        'norm_max': args.epsilon,
        'norm_step': args.norm_step,
        'adv_norm_mean': lpnorm.mean(),
        'adv_norm_min': lpnorm.min(),
        'adv_norm_max': lpnorm.max(),
        'n_iter': args.n_iter,
        'n_ensemble': args.n_ensemble,
        'shuffle': args.shuffle,
        'image_size': args.image_size,
        'dataset': 'NIPS2017' if 'NIPS2017' in args.source_model_path[0] else 'CIFAR10',
        'nb_adv': all_adv.shape[0],
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'full_imagenet_dir': args.full_imagenet_dir,
        'seed': args.seed,
        'random_start': args.random_start,
        'input_trans': args.input_transformation,
        'source_model_refine': args.source_model_refinement,
        'loss_function': args.loss_function,
        'grad_calculation': args.grad_calculation,
        'backpropagation': args.backpropagation,
        'update_dir_calcu': args.update_dir_calculation,
        'as_baseline': args.as_baseline,
        'bsl_adv_img_path': args.bsl_adv_img_path,
        'decay_factor': args.decay_factor,
        'n_var_sample': args.n_var_sample,
        'ghost': args.ghost_attack,
        'time': end_time - start_time,
        'adv_save_dir': args.save_dir,
        'command': ' '.join(sys.argv[1:]),
        'args': args.__dict__
    })
    df_metrics = pd.DataFrame([dict_metrics, ])
    os.makedirs(os.path.dirname(args.csv_export_path), exist_ok=True)
    df_metrics.to_csv(args.csv_export_path, mode='a', header=not os.path.exists(args.csv_export_path), index=False)
