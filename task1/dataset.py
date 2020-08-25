import pandas as pd
import re
import jieba
from collections import Counter

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch


data_path = 'data/weibo_senti_100k.csv'
model_dir = 'model/model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEN_OF_LINE = 100
SPLIT_FRAC = 0.8
BATCH_SIZE = 64
NUM_WORKER = 0

ch_punc = r"！？｡。＂＃＄％＆＇（）＊＋，－―／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
pattern = r"(回复)?@.*?[\s:：]|[\s"  # 去掉@与空白部分
pattern += ch_punc+r"""!"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~]"""  # 去标点符号等特殊字符
# 但实际上标点符号能够反映情绪，或许不去会好一些

pd_all = pd.read_csv(data_path)
words_cnt = dict()
reviews_cut = []

for review in pd_all['review']:
    r = re.sub(pattern, '', review)
    r = r.encode().decode('utf-8-sig').lower()  # 去掉字节顺序标记BOM \ufeff
    r = list(jieba.cut(r))[:LEN_OF_LINE]
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
VOCAB_SIZE = len(vocab_to_int) + 1
# +1是考虑到评论前填充的0

reviews_int = np.zeros(shape=(len(reviews_cut), LEN_OF_LINE), dtype=int)
for i, review in enumerate(reviews_cut):
    r = [vocab_to_int[w] for w in review]
    # r = np.pad(r, (LEN_OF_LINE-len(r), 0))
    reviews_int[i, -len(r):] = r
# 把所有评论的每个词都用对应整数替代，并填充至固定长度

# np.random.shuffle(reviews_int)
split_idx = int(reviews_int.shape[0]*SPLIT_FRAC)
train_x, remain_x = reviews_int[:split_idx], reviews_int[split_idx:]
train_y, remain_y = labels_int[:split_idx], labels_int[split_idx:]

split_idx = int(len(remain_x)*0.5)
val_x, test_x = remain_x[:split_idx], remain_x[split_idx:]
val_y, test_y = remain_y[:split_idx], remain_y[split_idx:]
# 训练集、验证集、测试集的划分

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
val_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, drop_last=True)
val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, drop_last=True)
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
