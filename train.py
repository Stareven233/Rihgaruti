import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model import Net
from config import root, batch_size, transform
from config import valid_train, valid_dev

lr = 1e-4
num_epoch = 1
num_workers = 2


def train(model):
    train_set = ImageFolder(root=root, transform=transform, is_valid_file=valid_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    model.train()
    # model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(num_epoch):
        # 每次循环完整训练一次数据集
        for i, (inputs, labels) in enumerate(train_loader):
            # 每次循环训练一个batch
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch: {e}, Loss: {loss/batch_size}')

    torch.save(model.state_dict(), 'model.pkl')


def validate(model):
    dev_set = ImageFolder(root=root, transform=transform, is_valid_file=valid_dev)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=True, num_workers=num_workers, drop_last=True)

    model.eval()
    correct = 0

    for inputs, labels in dev_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        # todo 看labels，outputs类型
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        # torch.max返回值中 [0]为对应维度的最大值，[1]为各个最大值所在维度的索引
        predicated = torch.max(outputs.data, 1)[1]

        correct += (predicated == labels).sum()

    print(f"开发集上的准确度：{100 * (correct/len(dev_loader))}")


if __name__ == '__main__':
    net = Net()
    net.cuda()

    train(net)
    validate(net)
