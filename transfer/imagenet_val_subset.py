import os
import random
import csv
import shutil

# # imagenet2012验证集的路径
# val_path = "/workspace/zhengmeixi2/imagenet2012/val"
# # 输出CSV文件的路径
# output_csv = "/workspace/zhengmeixi2/transfer_based_blackbox_bench/data/dataset/sub_imagenet_4/f2l.csv"
# # 存放采样图片的目录
# sampled_images_dir = "/workspace/zhengmeixi2/transfer_based_blackbox_bench/data/dataset/sub_imagenet_4/images"
#
# # 确保存放采样图片的目录存在
# if not os.path.exists(sampled_images_dir):
#     os.makedirs(sampled_images_dir)
#
# # 获取所有类别的文件夹并按照文件名升序排序
# categories = sorted([d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))])
#
# # 确保选出的类别数量是1000
# assert len(categories) == 1000, "类别数量不是1000，请检查路径。"
#
# # 准备数据列表，用于之后写入CSV
# data_list = []
#
# # 遍历每个类别
# for idx, category in enumerate(categories):
#     category_path = os.path.join(val_path, category)
#     # 获取当前类别下的所有图片
#     images = os.listdir(category_path)
#     # 随机选择一张图片
#     selected_image = random.choice(images)
#     # 完整的图片路径
#     image_path = os.path.join(category_path, selected_image)
#     # 复制图片到指定目录
#     shutil.copy(image_path, sampled_images_dir)
#     # 随机选择一个错误标签，该标签与真实标签不同
#     wrong_label = idx
#     while wrong_label == idx:
#         wrong_label = random.randint(0, 999)
#     # 记录图片名称和对应的数值标签（使用类别的索引作为标签）
#     data_list.append([selected_image, idx+1, wrong_label+1])
#
# # 打乱data_list中的数据顺序
# random.shuffle(data_list)
#
# # 写入CSV文件
# with open(output_csv, 'w', newline='') as file:
#     writer = csv.writer(file)
#     # 写入表头
#     writer.writerow(["FileName", "TrueLabel", "TargetLabel"])
#     # 写入数据
#     writer.writerows(data_list)
#
# print("CSV文件已成功生成，图片已经保存到指定目录。")


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

class sub_imagenet(Dataset):
    def __init__(self, images_dir, selected_images_csv, transform=None):
        super(sub_imagenet, self).__init__()
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