import torch
from model import Net
from config import run_set_dir, transform
import os
import random
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

model_dir = './model.pkl'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    img_tensor = [transform(i) for i in img]
    img_tensor = torch.stack(img_tensor, dim=0)
    return img, img_tensor


def run(model):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    print("模型加载完成")

    files = input_file or get_samples(run_set_dir, sample_num)
    raw_img, inputs = file_handle(files)
    inputs = inputs.to(device)
    print("数据加载完成")

    outputs = model(inputs)
    outputs = F.softmax(outputs, dim=1)

    for i, out in enumerate(outputs):
        plt.figure()
        label = ('cat', 'dog', )[out[0] < out[1]]
        print(f"Sample{i}: This is a {label}", end=', ')
        print("cat: {:.2%}, dog: {:.2%}".format(*out))
        plt.imshow(raw_img[i])
    plt.show()


if __name__ == '__main__':
    net = Net().to(device)
    print("running...")
    run(net)
