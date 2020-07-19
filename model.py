import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(__class__, self).__init__()

        # 两个卷积层，通过padding使得卷积(仅池化时/2)后图片尺寸不变
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
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

        # 将x变为全连接层所需的 (32*32*16, 1) 向量，但或许该写成view(1, -1)
        x = x.view(x.shape[0], -1)
        print(f'model.py：x.shape[0] = {x.shape[0]}')

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)

        return y
