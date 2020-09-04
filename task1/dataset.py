import pandas as pd
import jieba
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import re
from collections import Counter
import pickle
import config as cf

pd_all = pd.read_csv(cf.DATA_PTH)
words_cnt = dict()
reviews_cut = []

for review in pd_all['review']:
    r = re.sub(cf.pattern, '', review)
    r = r.encode().decode('utf-8-sig').lower()  # 去掉字节顺序标记BOM \ufeff
    r = list(jieba.cut(r))[:cf.LEN_OF_LINE]
    reviews_cut.append(r)

    cnt = Counter(r)
    for c in cnt:
        if c in words_cnt:
            words_cnt[c] += cnt[c]
        else:
            words_cnt[c] = cnt[c]
# 计算每个词的频数

labels_int = np.array(pd_all['label'])
# 对应的标签
print("data loaded")

vocab = sorted(words_cnt, key=words_cnt.get, reverse=True)
vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}
# 按频数编号，频数大的编码小
VOCAB_SIZE = len(vocab_to_int) + 1  # weibo_senti_100k: 160163
# +1是考虑到评论前填充的0

with open(cf.VOCAB_TO_INT_PTH, 'wb') as f:
    pickle.dump(vocab_to_int, f)
with open(cf.VOCAB_SIZE_PTH, 'wb') as f:
    pickle.dump(VOCAB_SIZE, f)
# 保存train、run所需的变量，加速模型运行

reviews_int = np.zeros(shape=(len(reviews_cut), cf.LEN_OF_LINE), dtype=int)
for i, review in enumerate(reviews_cut):
    r = [vocab_to_int[w] for w in review]
    # r = np.pad(r, (LEN_OF_LINE-len(r), 0))
    reviews_int[i, -len(r):] = r
# 把所有评论的每个词都用对应整数替代，并填充至固定长度

# np.random.shuffle(reviews_int)
split_idx = int(reviews_int.shape[0]*cf.SPLIT_FRAC)
train_x, remain_x = reviews_int[:split_idx], reviews_int[split_idx:]
train_y, remain_y = labels_int[:split_idx], labels_int[split_idx:]

split_idx = int(len(remain_x)*0.5)
val_x, test_x = remain_x[:split_idx], remain_x[split_idx:]
val_y, test_y = remain_y[:split_idx], remain_y[split_idx:]
# 训练集、验证集、测试集的划分

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
val_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_loader = DataLoader(train_data, shuffle=True, batch_size=cf.BATCH_SIZE, num_workers=cf.NUM_WORKER, drop_last=True)
val_loader = DataLoader(val_data, shuffle=True, batch_size=cf.BATCH_SIZE, num_workers=cf.NUM_WORKER, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=cf.BATCH_SIZE, num_workers=cf.NUM_WORKER, drop_last=True)
# 准备DataLoader
print("DataLoader ready")

if __name__ == '__main__':
    print('单词总数: ', VOCAB_SIZE-1)  # 160162
    print('编码后第一句评论: \n', reviews_int[0])
    review_len = sorted([len(x) for x in reviews_int])
    print('最短5个', review_len[:5])  # [1, 1, 1, 1, 1]
    print('最长5个', review_len[-5:])  # [88, 89, 91, 93, 130]

del pd_all
del words_cnt, reviews_cut
del vocab
del reviews_int, labels_int
del train_x, remain_x, train_y, remain_y
del val_x, test_x, val_y, test_y
del train_data, val_data, test_data
# 似乎没有差别...
