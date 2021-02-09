import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttBC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size, embd_pretrained, vocab_size, label_size,
                 DEVICE, dropout=0.5):
        super(LSTMAttBC, self).__init__()
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
        self.softmax = nn.Softmax(dim=0)
        # print(type(self.word_embds.weight))       # <class 'torch.nn.parameter.Parameter'>
        # print(type(self.word_embds.weight.data))  #<class 'torch.Tensor'>

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim).to(self.DEVICE)),
                Variable(torch.zeros(1, self.batch_size, self.hidden_dim).to(self.DEVICE)))

    def forward(self, sentence):
        mask_words = (sentence > 1).float().unsqueeze(-1).to(self.DEVICE)
        x = self.word_embds(sentence).view(len(sentence), self.batch_size, -1).float()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # aggregate representations with attention
        atts = torch.sum(torch.mul(lstm_out, lstm_out[-1].unsqueeze(0)), dim=-1, keepdim=True)
        atts = self.softmax(atts * mask_words)
        repre_weighted = torch.sum(torch.mul(lstm_out, atts), dim=0)
        # linear predictor
        y = self.hidden2label(repre_weighted)
        log_probs = F.log_softmax(y)
        return y, lstm_out, x, atts, log_probs
