import re
import jieba
import numpy as np
import torch

from model import Net
from dataset import pattern, LEN_OF_LINE, vocab_to_int
from dataset import device, model_dir, VOCAB_SIZE, BATCH_SIZE
from train import EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL


def tokenize(review):
    """
    将输入的一句评论转换为相应的整数编码序列并填充0至特定长度
    :param review: 文本序列(str)
    :return: Tensor格式的编码序列
    """

    r = re.sub(pattern, '', review)
    r = r.lower()
    r = list(jieba.cut(r))[:LEN_OF_LINE]
    # print(r)

    review_int = np.zeros(shape=(LEN_OF_LINE,), dtype=int)
    r = [vocab_to_int[w] for w in r]
    review_int[-len(r):] = r

    return torch.from_numpy(review_int)


def predict(model, review):
    """
    接收单句评论，给出正面或负面情感的判断
    :param model: 网络模型
    :param review: 文本序列(str)
    :return: None
    """

    model.load_state_dict(torch.load(model_dir))
    model.eval()
    print("model loaded")
    review_tensor = tokenize(review).to(device)
    h = model.init_hidden(1)

    output, h = model(review_tensor.unsqueeze(0), h)
    # 作为输入的review_tensor.shape应为[1, n]
    pred = int(output.round().item())
    print(f"{('Negative', 'Positive')[pred]} emotion #{output.item()}")


if __name__ == '__main__':
    net = Net(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL).to(device)
    print("predicting...")
    # text = input('输入一句评论：')是
    # 若用input接收输入会在加载model时蓝屏
    text = '求整个工程和数据集，感激不尽'
    predict(net, text)
