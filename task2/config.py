import torch
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import random
import re


train_set_dir = './data/train'
run_set_dir = './data/test1'
model_dir = './model/model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32  # 过大过小都会降低准确率
learning_rate = 0.001  # 太小的话似乎会因为太慢&epoch次数不够而训练不充分
num_epoch = 8
num_workers = 0
img_size = 128


train_data_size = 12500
# 保证每次划分的训练/开发集都一样
random.seed('Rihgaruti')
# 即每个class各随机2500张(20%)作开发集，剩下的全部作为训练集
random_dev_list = random.sample(range(train_data_size), int(train_data_size*0.2))


def transform(size=img_size):
    return transforms.Compose([
        transforms.Resize(size),  # 保持比例，将短边放缩为IMG_SIZE
        transforms.CenterCrop(size),  # 裁去长边多余部分，保证图片为不变形的长方形
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def valid_dev(path):
    """
    利用正则表达式筛选出随机的开发集图片，用于ImageFolder
    :param path: 图片集路径
    :return: 该图片是否合格
    """

    re_pattern = r'\w{3}\.(\d+)\.\w+$'
    seq = re.search(re_pattern, path)
    seq = int(seq.group(1))
    return seq in random_dev_list


def valid_train(path):
    """
    开发集以外的图片都为训练集，用于ImageFolder
    :param path: 图片集路径
    :return: 该图片是否合格
    """

    return not valid_dev(path)


def train_loader():
    train_set = ImageFolder(root=train_set_dir, transform=transform(), is_valid_file=valid_train)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def train_res_loader():
    train_set = ImageFolder(root=train_set_dir, transform=transform(224), is_valid_file=valid_train)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def dev_loader():
    dev_set = ImageFolder(root=train_set_dir, transform=transform(), is_valid_file=valid_dev)
    return DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def custom_res_model():
    model = models.resnet18(pretrained=False)
    # 更改最后分类用的fc使之适合二分类，512是按ResNet18源码确定的
    model.fc = torch.nn.Linear(512, 2)
    return model
