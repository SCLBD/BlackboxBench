import csv
import math
import os
from collections import OrderedDict
import pickle
import PIL.Image as Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

class NIPS2017(Dataset):
    def __init__(self, images_dir, selected_images_csv, transform=None):
        super(NIPS2017, self).__init__()
        self.images_dir = images_dir
        self.selected_images_csv = selected_images_csv
        self.transform = transform
        self._load_csv()
    def _load_csv(self):
        self.image_id_list = []
        self.label_ori_list = []
        self.label_tar_list = []

        with open(self.selected_images_csv) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                self.image_id_list.append(row['ImageId'])
                self.label_ori_list.append(int(row['TrueLabel']) - 1)
                self.label_tar_list.append(int(row['TargetClass']) - 1)
    def __getitem__(self, item):
        image = Image.open(os.path.join(self.images_dir, self.image_id_list[item] + '.png'))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.label_ori_list[item]), int(self.label_tar_list[item])
    def __len__(self):
        return len(self.image_id_list)


class sub_imagenet(Dataset):
    def __init__(self, images_dir, file2label_csv, transform=None):
        super(sub_imagenet, self).__init__()
        self.images_dir = images_dir
        self.selected_images_csv = file2label_csv
        self.transform = transform
        self._load_csv()
    def _load_csv(self):
        self.image_id_list = []
        self.label_ori_list = []
        self.label_tar_list = []

        with open(self.selected_images_csv) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                self.image_id_list.append(row['FileName'])
                self.label_ori_list.append(int(row['TrueLabel']) - 1)
                self.label_tar_list.append(int(row['TargetLabel']) - 1)
    def __getitem__(self, item):
        image = Image.open(os.path.join(self.images_dir, self.image_id_list[item]))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.label_ori_list[item]), int(self.label_tar_list[item])
    def __len__(self):
        return len(self.image_id_list)


class Cifar10(Dataset):
    def __init__(self, cifar10_dir, transform=None):
        super(Cifar10, self).__init__()
        self.cifar10_dir = cifar10_dir
        self.data = []
        self.targets = []
        file_path = os.path.join(cifar10_dir, 'test_batch')
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.transform = transform
        self.label_switch = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).long()
    def __getitem__(self, item):
        img, true_label = self.data[item], self.targets[item]
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        target_label = self.label_switch[true_label]
        return img, int(true_label), int(target_label)
    def __len__(self):
        return len(self.targets)




def build_dataloader(args):
    if 'CIFAR10' in args.source_model_path[0]:
        img_transform = T.ToTensor()
        data_path = 'data/dataset/CIFAR10/cifar-10-batches-py'
        dataset = Cifar10(cifar10_dir=data_path,
                          transform=img_transform)
    elif 'NIPS2017' in args.source_model_path[0]:
        if args.imagenet_sub is None:
            if args.image_size:
                img_transform = T.Compose([T.Resize(args.image_size),
                                           T.ToTensor()])
            else:
                img_transform = T.Compose([T.ToTensor()])
            data_csv = 'data/dataset/NIPS2017/images.csv'
            data_path = 'data/dataset/NIPS2017/images'
            dataset = NIPS2017(images_dir=data_path,
                               selected_images_csv=data_csv,
                               transform=img_transform)
        else:
            img_transform = T.Compose([T.Resize((299, 299)),
                                       T.ToTensor()])
            data_csv = f'data/dataset/sub_imagenet_{args.imagenet_sub}/f2l.csv'
            data_path = f'data/dataset/sub_imagenet_{args.imagenet_sub}/images'
            dataset = sub_imagenet(images_dir=data_path,
                                   file2label_csv=data_csv,
                                   transform=img_transform)
            print('imagenet!')
    else:
        raise NotImplementedError('Dataset not supported')

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                              pin_memory=True, num_workers=args.num_workers)  # shuffle has to be False!
    return data_loader