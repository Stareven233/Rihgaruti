import torch
import numpy as np

from model import Net
from dataset import device, model_dir, VOCAB_SIZE, BATCH_SIZE
from dataset import train_loader, val_loader, test_loader

EMBEDDING_DIM = 400
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
LR = 0.001  # learning rate
N_EPOCHS = 4


def train(model, data_loader):
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


def val_test(model, data_loader, load_model=False):
    if load_model:
        model.load_state_dict(torch.load(model_dir))
        print('model loaded')
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
        # correct += ((outputs > 0.5) == labels).sum().item()
        correct += labels.eq(outputs.round()).sum().item()

        if i % 100 == 0:
            print(f'Loop: {i}, Total: {total}, Correct: {correct}')

    print(f"准确度：{100 * (correct/total)}%")
    print(f"Loss：{np.mean(losses)}%")
    # first val: 准确度：92.02874331550802% and Loss：0.169401148444958%
    # first test: 准确度：92.1457219251337% and Loss：0.17519030125966087%


if __name__ == '__main__':
    net = Net(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL).to(device)
    # print("training...")
    # train(net, train_loader)
    #
    # print("validating...")
    # val_test(net, val_loader)

    print("testing...")
    val_test(net, test_loader, True)
