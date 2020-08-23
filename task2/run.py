import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .model import Net
from . import config
from .config import device
import os
import random
from PIL import Image

sample_num = 3
# input_file 文件路径列表(list)，跟get_samples返回值同样
# input_file = ['./data/neko1.jpg']
input_file = None


def get_samples(data_dir, num):
    """
    从给定目录中随机选取文件
    :param data_dir: 文件目录，不可再有子目录
    :param num: 选取的文件数量
    :return: 文件路径列表
    """

    # 重新设定随机种子，覆盖掉config中设置的固定顺序
    random.seed()
    samples = random.sample(os.listdir(data_dir), num)
    sample_list = [os.path.join(data_dir, s) for s in samples]
    return sample_list


def file_handle(file_list):
    """
    读取并处理路径列表中的图片
    :param file_list: 图片路径组成的列表
    :return: (原始图像数据, 图像tensor) 二元组
    """

    img = [Image.open(f) for f in file_list]
    # config.transform是个函数，返回transform
    img_tensor = [config.transform()(i) for i in img]
    img_tensor = torch.stack(img_tensor, dim=0)
    return img, img_tensor


def run(model):
    model.load_state_dict(torch.load(config.model_dir))
    model.to(device)
    model.eval()
    print("模型加载完成")

    files = input_file or get_samples(config.run_set_dir, sample_num)
    raw_img, inputs = file_handle(files)
    inputs = inputs.to(device)
    print("数据加载完成")

    outputs = model(inputs)
    outputs = F.softmax(outputs, dim=1)

    for i, out in enumerate(outputs):
        # plt.figure、plt.show是为了使所有图片都能一起显示
        plt.figure()
        label = ('cat', 'dog', )[out[0] < out[1]]

        print(f"Sample{i}: This is a {label}", end=', ')
        print("cat: {:.2%}, dog: {:.2%}".format(*out))
        plt.imshow(raw_img[i])
    plt.show()


if __name__ == '__main__':
    print("running...")
    run(Net())  # 自己写的简陋网络
    # run(config.custom_res_model())  # 以ResNet18为基础的网络
