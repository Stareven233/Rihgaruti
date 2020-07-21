import torch
from torchvision import transforms

import random
import re


class Config(object):
    train_set_dir = './data/train'
    run_set_dir = './data/test1'

    batch_size = 32
    learning_rate = 0.00015
    num_workers = 0
    num_epoch = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = './model.pkl'

    img_size = 128
    transform = transforms.Compose([
        transforms.Resize(img_size),  # 保持比例，将短边放缩为IMG_SIZE
        transforms.CenterCrop(img_size),  # 裁去长边多余部分，保证图片为不变形的长方形
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_data_size = 12500
    random_dev_list = random.sample(range(train_data_size), int(train_data_size*0.2))
    # 即每个class各随机2500张(20%)作开发集，剩下的全部作为训练集

    @staticmethod
    def valid_dev(path):
        """
        利用正则表达式筛选出随机的开发集图片，用于ImageFolder
        :param path: 图片集路径
        :return: 该图片是否合格
        """

        re_pattern = r'\w{3}\.(\d+)\.\w+$'
        seq = re.search(re_pattern, path)
        seq = int(seq.group(1))
        return seq in Config.random_dev_list

    @staticmethod
    def valid_train(path):
        """
        开发集以外的图片都为训练集，用于ImageFolder
        :param path: 图片集路径
        :return: 该图片是否合格
        """

        return not Config.valid_dev(path)
