from torchvision import transforms

import random
import re


root = './data/train'
img_size = 128
batch_size = 16
num_workers = 2

transform = transforms.Compose([
    transforms.Resize(img_size),  # 保持比例，将短边放缩为IMG_SIZE
    transforms.CenterCrop(img_size),  # 裁去长边多余部分，保证图片为不变形的长方形
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

data_size = 12500
random_dev_list = random.sample(range(data_size), int(data_size*0.2))
# 即每个class各随机2500张(20%)作开发集，剩下的全部作为训练集


def valid_dev(path):
    re_pattern = r'\w{3}\.(\d+)\.\w+$'
    seq = re.search(re_pattern, path)
    seq = int(seq.group(1))
    return seq in random_dev_list


def valid_train(path):
    return not valid_dev(path)
