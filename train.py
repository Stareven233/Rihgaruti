"""
似乎是windows系统的原因，num_workers>0，即启用多进程的时候会导致dataloader报错：[WinError 5] 拒绝访问，应该是由于多进程锁获取的冲突
即便在num_workers=0时，debug下也非常容易卡死，造成系统蓝屏重启...
"""

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model import Net
from config import Config


device = Config.device
num_workers = Config.num_workers
batch_size = Config.batch_size


def train(model):
    train_set = ImageFolder(root=Config.train_set_dir, transform=Config.transform, is_valid_file=Config.valid_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    print("数据集加载完成")

    model.train()
    # model = torch.nn.DataParallel(model)  # 只有一个GPU(cuda)，没必要
    criterion = torch.nn.CrossEntropyLoss()

    # Adam: epoch 10次略低于1次的...而SGD 10次明显好于1次
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=Config.learning_rate, momentum=0.9)

    print("进入epoch循环中...")
    for e in range(Config.num_epoch):
        # 每次循环完整训练一次数据集
        for i, (inputs, labels) in enumerate(train_loader):
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

    torch.save(model.state_dict(), Config.model_dir)
    print("模型已保存：", Config.model_dir)


def validate(model):
    dev_set = ImageFolder(root=Config.train_set_dir, transform=Config.transform, is_valid_file=Config.valid_dev)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    print("数据集加载完成")

    model.load_state_dict(torch.load(Config.model_dir))
    model.eval()
    print("模型加载完成")

    # len(dev_loader)是batch的个数，len(dev_set)是完整数据数，但由于drop_last的存在还是得自己统计
    total = 0
    correct = 0

    print("进入predicate循环中...")
    for i, (inputs, labels) in enumerate(dev_loader):
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


if __name__ == '__main__':
    net = Net().to(device)

    # print("training...")
    # train(net)

    print("validating...")
    validate(net)
