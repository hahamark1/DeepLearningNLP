import torch.nn as nn
import torch

class RNNModel(nn.Module):
    def __init__(self, word_vocab_size, hidden_dim, layer_dim, output_dim, dim_embedding=512):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embeddings = nn.Embedding(word_vocab_size, dim_embedding, padding_idx=0)
        self.rnn = nn.RNN(dim_embedding, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        # Initialize hidden state with zeros
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # Embed the inputs
        input = self.embeddings(x)
        # Run the whole sequence through the rnn cell
        out, hn = self.rnn(input, h0)
        # Reshape the output so its ready for the fully connected layer
        final_out = out.gather(1, lengths.view(-1, 1, 1).expand(out.size(0), 1, out.size(2)))
        # Run the output through the fully connected layer to get the final prediction
        out = self.fc(final_out).squeeze(1)
        return out

if __name__ == '__main__':
    print('Please run the bash scripts or train_*.py files!')