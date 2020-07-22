"""
似乎是windows系统的原因，num_workers>0，即启用多进程的时候会导致dataloader报错：[WinError 5] 拒绝访问，应该是由于多进程锁获取的冲突
即便在num_workers=0时，debug下也非常容易卡死，造成系统蓝屏重启...
"""

import torch
import torch.nn.functional as F
from torchvision import models

from model import Net
import config
from config import device


def train(model=None, data_loader=None):
    if model is None:
        model = Net()
        print("模型加载完成")
    model.to(device)
    model.train()
    # model = torch.nn.DataParallel(model)  # 只有一个GPU(cuda)，没必要

    if data_loader is None:
        data_loader = config.train_loader()
        print("数据集加载完成")

    criterion = torch.nn.CrossEntropyLoss()
    # Adam: epoch 10次略低于1次的...而SGD 10次明显好于1次
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

    print("进入epoch循环中...")
    for e in range(config.num_epoch):
        # 每次循环完整训练一次数据集
        for i, (inputs, labels) in enumerate(data_loader):
            # 每次循环训练一个batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # batch_size=16时，outputs.shape=[16, 2]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f'Epoch: {e}, Loop: {i}, Loss: {loss}')

    torch.save(model.state_dict(), config.model_dir)
    print("模型已保存：", config.model_dir)


def train_resnet18():
    print("ResNet18模型加载开始")
    # 此处修改了 torchvison.models.resnet里_resnet的代码，指定了load_state_dict_from_url的参数model_dir
    model = models.resnet18(pretrained=True)  # 直接用, num_classes=2 会与已训练模型的w, b参数size冲突
    model.fc = torch.nn.Linear(512, 2)
    print("ResNet18模型加载完成")

    data_loader = config.train_res_loader()
    print("ResNet18数据集加载完成")

    train(model, data_loader)


def validate(model=None, data_loader=None):
    if model is None:
        model = Net()
        model.load_state_dict(torch.load(config.model_dir))
        print("模型加载完成")
    model.to(device)
    model.eval()

    if data_loader is None:
        data_loader = config.dev_loader()
        print("数据集加载完成")

    # len(dev_loader)是batch的个数，len(dev_set)是完整数据数，但由于drop_last的存在还是得自己统计
    total = 0
    correct = 0

    print("进入predicate循环中...")
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)

        # torch.max返回值中 [0]为对应维度的最大值，[1]为各个最大值所在维度的索引
        # 相当于predicated = torch.max(outputs.detach(), 1)[1]，detach用于忽略本次计算历史?
        pred = outputs.max(1, keepdim=True)[1]

        total += len(labels)
        # 在最后predicated为列向量，而labels是行向量，直接==会触发广播机制(batch>1有影响)
        correct += (pred == labels.view_as(pred)).sum().item()

        if i % 50 == 0:
            print(f'Loop: {i}, Total: {total}, Correct: {correct}')

    print(f"开发集上的准确度：{100 * (correct/total)}%")
    # 第一次的：开发集上的准确度：75.53999999999999%


def validate_resnet18():
    model = config.custom_res_model()
    model.load_state_dict(torch.load(config.model_dir))
    print("ResNet18模型加载完成")

    validate(model)
    # 开发集上的准确度(ResNet18)：91.96714743589743%


if __name__ == '__main__':

    # print("training...")
    # train()
    # train_resnet18()

    print("validating...")
    # validate()
    validate_resnet18()
