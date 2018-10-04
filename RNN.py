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
        self.embeddings = nn.Embedding(word_vocab_size, dim_embedding, padding_idx=0)
        self.rnn = nn.RNN(dim_embedding, hidden_dim, layer_dim, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)



    def forward(self, x, lengths):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # One time step
        input = self.embeddings(x)
        out, hn = self.rnn(input, h0)
        # Index hidden state of last time step
        # out.size() --> batch_size, seq_length, dimension
        # get final state of end of sequence (ignore padding)
        final_out = out.gather(1, lengths.view(-1, 1, 1).expand(out.size(0), 1, out.size(2)))
        out = self.fc(final_out).squeeze(1)
        # out.size() --> batch_size, output_dim
        return out
