import torch
import numpy as np

from model import Net
from dataset import device, model_dir, VOCAB_SIZE, BATCH_SIZE
from dataset import train_loader, val_loader

EMBEDDING_DIM = 400
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
LR = 0.001  # learning rate
N_EPOCHS = 4


def train(model=None, data_loader=None):
    if not model or not data_loader:
        return
    model.train()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("进入epoch循环中...")
    for e in range(N_EPOCHS):
        h = model.init_hidden(BATCH_SIZE)

        for i, (inputs, labels) in enumerate(data_loader):
            h = tuple([each.detach() for each in h])
            # 从图中分离出独立的隐藏层参数，共用数据但不求导
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, h = model(inputs, h)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch: {e}, Loop: {i}, Loss: {loss}')

    torch.save(model.state_dict(), model_dir)
    print("模型已保存：", model_dir)


def validate(model=None, data_loader=None):
    if not model or not data_loader:
        return
    # model.load_state_dict(torch.load(model_dir))
    model.eval()

    criterion = torch.nn.BCELoss()
    h = model.init_hidden(BATCH_SIZE)
    losses = []
    total = 0
    correct = 0

    for i, (inputs, labels) in enumerate(data_loader):
        h = tuple([each.detach() for each in h])
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, h = model(inputs, h)
        loss = criterion(outputs, labels.float())
        losses.append(loss.item())

        total += len(labels)
        correct += ((outputs > 0.5) == labels).sum().item()

        if i % 50 == 0:
            print(f'Loop: {i}, Total: {total}, Correct: {correct}')

    print(f"val 准确度：{100 * (correct/total)}%")
    print(f"val Loss：{np.mean(losses)}%")


if __name__ == '__main__':
    net = Net(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL).to(device)
    print("training...")
    train(net, train_loader)

    print("validating...")
    validate(net, val_loader)
