import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, word_vocab_size, chr_vocab_size, dim_embedding=128):
        super(ConvNet, self).__init__()
        self.dim_embedding = dim_embedding
        print("Maximum character size:, ", chr_vocab_size)
        self.word_embedding = nn.Embedding(word_vocab_size, dim_embedding, padding_idx = 0)
        self.chr_embedding = nn.Embedding(chr_vocab_size, dim_embedding, padding_idx = 0)
        self.conv_chr = nn.Conv1d(dim_embedding, dim_embedding, 3, padding=1)
        self.conv_word = nn.Conv1d(dim_embedding, dim_embedding, 3, padding=1)
        self.conv_sentence = nn.Conv1d(2*dim_embedding, 2*dim_embedding, 3, padding=1)
        self.linear_1 = nn.Linear(2*dim_embedding, 4*dim_embedding)
        self.linear_2 = nn.Linear(4*dim_embedding, 2)
        
    def forward(self, words, words_in_char, sentence_vector=0):
        # Create word embeddings
        vector_wrds = self.word_embedding(words)

        # Create character word embeddings
        vector_wchs = []
        for word_char in words_in_char:
            vector_chr = self.chr_embedding(word_char).t()
            vector_chr = vector_chr.reshape(1, self.dim_embedding, -1)
            word_convolution = self.conv_chr(vector_chr)
            vector_wch = torch.max(word_convolution, 2)[0].squeeze()
            vector_wchs.append(vector_wch)
        vector_wchs = torch.stack(vector_wchs)

        # Combine word and character word embeddings
        u = torch.cat((vector_wrds, vector_wchs), 1).t().reshape(1, 2*self.dim_embedding, -1)
        #
        # r_sents = []
        # for u_i in u:
        #     r_sent = self.conv_sentence(u_i.reshape(1, self.dim_embedding * 2, -1)).squeeze()
        #     r_sents.append(r_sent)
        r_sents = self.conv_sentence(u)
        r_sents = torch.max(r_sents, 2)[0]

        out = torch.tanh(self.linear_1(r_sents))
        out = self.linear_2(out)

        return out
