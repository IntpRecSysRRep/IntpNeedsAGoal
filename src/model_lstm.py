import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class LSTMBC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size, embd_pretrained, vocab_size, label_size,
                 DEVICE, dropout=0.5):
        super(LSTMBC, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.DEVICE = DEVICE
        self.dropout = dropout
        # embeddings will be updated (otherwise performance is bad)
        self.word_embds = nn.Embedding.from_pretrained(torch.from_numpy(embd_pretrained), freeze=False)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
        # print(type(self.word_embds.weight))       # <class 'torch.nn.parameter.Parameter'>
        # print(type(self.word_embds.weight.data))  #<class 'torch.Tensor'>

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim).to(self.DEVICE)),
                Variable(torch.zeros(1, self.batch_size, self.hidden_dim).to(self.DEVICE)))

    def forward(self, sentence):
        x = self.word_embds(sentence).view(len(sentence), self.batch_size, -1).float()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # y = self.hidden2label(torch.mean(lstm_out, dim=0, keepdim=False))
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return y, lstm_out, x, log_probs
