import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(__class__, self).__init__()

        # 两个卷积层，通过padding使得卷积(仅池化时/2)后图片尺寸不变
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # 两次池化后图片size由128变为32，而通道为16
        self.fc1 = nn.Linear(32*32*16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # todo 开启mini-batch时x是什么结构？

        # 将x变为全连接层所需的 (16, 32*32*16) 向量，但或许该写成view(1, -1)？
        x = x.view(x.shape[0], -1)
        # x = x.view(1, -1)  # 导致下一行fc1报错

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        # 原本还要套上softmax，但train.py中的CrossEntropyLoss自带了log_softmax

        return y
