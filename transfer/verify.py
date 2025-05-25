import os
import torchvision.transforms as T
import numpy as np
import torch
from tqdm import tqdm
from surrogate_model.utils import guess_and_load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_attack(args, target_models):
    true_labels = torch.from_numpy(np.load(args.save_dir + '/true_labels.npy')).long()
    target_labels = torch.from_numpy(np.load(args.save_dir + '/target_labels.npy')).long()
    target = true_labels if not args.targeted else target_labels

    if 'NIPS2017' in args.source_model_path[0]:
        trans = T.Compose([])
        # trans = T.Compose([]) if args.targeted else T.Compose([T.Resize((256, 256)), T.CenterCrop((224, 224))])
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
            for j, target_model in tqdm(enumerate(target_models)):
                labels_now = labels + 1 if 'tf2torch' in target_model else labels
                target_model = guess_and_load_model(target_model)
                model_device = next(target_model.parameters()).device
                target_model.to(DEVICE)
                att_suc[j] += sum(torch.argmax(target_model(trans(inputs)), dim=1) != labels_now).cpu().numpy()
                target_model.to(model_device)
                torch.cuda.empty_cache()
                del target_model
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
        trans = T.Compose([])
        # trans = T.Compose([]) if args.targeted else T.Compose([T.Resize((256, 256)), T.CenterCrop((224, 224))])
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
                for j, target_model in tqdm(enumerate(target_models)):
                    labels_now = labels + 1 if 'tf2torch' in target_model else labels
                    target_model = guess_and_load_model(target_model)
                    model_device = next(target_model.parameters()).device
                    target_model.to(DEVICE)
                    att_suc[j] += sum(torch.argmax(target_model(trans(inputs)), dim=1) != labels_now).cpu().numpy()
                    target_model.to(model_device)
                    torch.cuda.empty_cache()
                    del target_model
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
                target_model = guess_and_load_model(target_model)
                model_device = next(target_model.parameters()).device
                target_model.to(DEVICE)
                ori_att_suc[j] += sum(torch.argmax(target_model(ori_img), dim=1) == true_label).cpu().numpy()
                target_model.to(model_device)
                torch.cuda.empty_cache()
                del target_model
    print(f"original attact success rate:{ori_att_suc/img_num}")
    return ori_att_suc/img_num

