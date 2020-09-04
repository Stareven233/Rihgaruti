import torch.nn as nn
from config import device


class Net(nn.Module):
    """
    使用LSTM的双向RNN模型
    整数表示的单词x -> embedding -> LSTM -> Sigmoid -> 预测的情感y
    """

    def __init__(self, vocab_size, bidirectional=True):
        super(__class__, self).__init__()

        self.embedding_dim = 400
        self.hidden_dim = 256
        self.n_layers = 2
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        # 只要设置好embedding层的size，网络会自己学习
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers,
                            dropout=0.5, batch_first=True, bidirectional=bidirectional)

        self.dropout = nn.Dropout(0.3)
        num = 1 + bidirectional
        # 用num简化了if语句，若双向，使输入size变为两倍
        self.fc = nn.Linear(self.hidden_dim*num, 1)
        # 由于是正负二元的分类，输出只需1位
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        # hidden是隐藏层的参数矩阵，与init_hidden返回值同类型
        batch_size = x.size(0)
        x = x.long()  # embedding参数要求long
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        num = 1 + self.bidirectional
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*num)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)

        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        # 每个单词都会有一个输出，此处取最后一个作为情绪正负与否的预测(它综合了整个句子)
        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).detach()
        num = 1 + self.bidirectional

        return (
            weight.new_zeros((self.n_layers*num, batch_size, self.hidden_dim)).to(device),
            weight.new_zeros((self.n_layers*num, batch_size, self.hidden_dim)).to(device),
            # weight.new(self.n_layers*num, batch_size, self.hidden_dim).zero_().to(device),
        )


if __name__ == '__main__':
    net = Net(233)
    print(net)
    # print(net.init_hidden(4))
