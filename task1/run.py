import re
import jieba
import numpy as np
import torch
import pickle

from model import Net
import config as cf


with open(cf.VOCAB_TO_INT_PTH, 'rb') as f:
    vocab_to_int = pickle.load(f)
with open(cf.VOCAB_SIZE_PTH, 'rb') as f:
    VOCAB_SIZE = pickle.load(f)


def tokenize(review):
    """
    将输入的一句评论转换为相应的整数编码序列并填充0至特定长度
    :param review: 文本序列(str)
    :return: Tensor格式的编码序列
    """

    r = re.sub(cf.pattern, '', review)
    r = r.lower()
    r = list(jieba.cut(r))[:cf.LEN_OF_LINE]
    # print(r)

    review_int = np.zeros(shape=(cf.LEN_OF_LINE,), dtype=int)
    r = [vocab_to_int.get(w, 0) for w in r]
    # 若出现生词则以0代替
    review_int[-len(r):] = r

    return torch.from_numpy(review_int)


def predict(model, review):
    """
    接收单句评论，给出正面或负面情感的判断
    :param model: 网络模型
    :param review: 文本序列(str)
    :return: None
    """

    model.load_state_dict(torch.load(cf.MODEL_PTH))
    model.eval()
    print("model loaded")
    review_tensor = tokenize(review).to(cf.device)
    h = model.init_hidden(1)

    output, h = model(review_tensor.unsqueeze(0), h)
    # 作为输入的review_tensor.shape应为[1, n]
    pred = int(output.round().item())
    print(f"Result: {('negative', 'positive')[pred]} emotion")
    print(f"Sigmoid Out: {output.item()}")


if __name__ == '__main__':
    net = Net(VOCAB_SIZE).to(cf.device)
    print("predicting...")
    # text = '感动感动。给我加鸡腿吧！看到自己动捕的作品能得到大家的认可，真心感动！期待正式上线！'
    text = '我昨天做梦  琴肥梦 变成三百斤的虚拟主播    直接把我吓醒了'  # BV1Na4y1E7BV
    predict(net, text)
