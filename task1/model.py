import torch.nn as nn
from dataset import device, vocab_to_int


class Net(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional=True, drop_prob=0.5):
        super(__class__, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(0.3)
        rate = 1 + bidirectional
        self.fc = nn.Linear(hidden_dim*rate, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        rate = 1 + self.bidirectional
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*rate)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self(out)

        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        # 取最后一个输出(即情绪正负与否的预测)
        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 1 + self.bidirectional

        return (
            weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().to(device),
            weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().to(device)
            # weight.new_zeros
        )


if __name__ == '__main__':
    vocab_size = len(vocab_to_int) + 1
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2
    bidirectional = True

    net = Net(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional)
    print(net)
