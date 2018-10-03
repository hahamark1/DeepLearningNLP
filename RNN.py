import torch.nn as nn
import torch
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, word_vocab_size, hidden_dim, layer_dim, output_dim, dim_embedding=512):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # .type(torch.LongTensor)
        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.embeddings = nn.Embedding(word_vocab_size, dim_embedding)
        self.rnn = nn.RNN(dim_embedding, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # x = self.embedding(x)
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        x = self.embeddings(x)
        out, hn = self.rnn(x, h0)

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 2
        return out
