import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, word_vocab_size, chr_vocab_size, max_sentence_length, dim_embedding=128):
        super(ConvNet, self).__init__()
        self.dim_embedding = dim_embedding
        print("Maximum character size: {}".format(chr_vocab_size))
        print(chr_vocab_size)
        self.word_embedding = nn.Embedding(word_vocab_size, dim_embedding, padding_idx = 0)
        self.chr_embedding = nn.Embedding(chr_vocab_size, dim_embedding, padding_idx = 0)
        self.conv_chr = nn.Conv2d(max_sentence_length, max_sentence_length, 3, padding=1)
        self.conv_word = nn.Conv1d(dim_embedding, dim_embedding, 3, padding=1)
        self.conv_sentence = nn.Conv1d(2*dim_embedding, 2*dim_embedding, 3, padding=1)
        self.linear_1 = nn.Linear(2*dim_embedding, 4*dim_embedding)
        self.linear_2 = nn.Linear(4*dim_embedding, 2)

    def forward(self, words, words_in_char, sentence_vector=0):
        # Create word embeddings
        vector_wrds = self.word_embedding(words)

        # Create character word embeddings
        print(torch.max(words_in_char))
        vector_wchs = self.chr_embedding(words_in_char)
        word_wchs = self.conv_chr(vector_wchs)
        word_wchs, _ = torch.max(word_wchs, 2)

        # word_wchs = []
        # for i in range(vector_wchs.shape[1]):
        #     word_matrix = vector_wchs[:, i, :, :]
        #     word_matrix = word_matrix.permute(0, 2, 1)
        #     word_matrix_conv = self.conv_chr(word_matrix)
        #     word_wch, _ = torch.max(word_matrix_conv, 2)
        #     word_wchs.append(word_wch)
        # word_wchs = torch.stack(word_wchs)
        # word_wchs = word_wchs.permute(1, 0, 2)

        # Combine word and character word embeddings
        u = torch.cat((vector_wrds, word_wchs), 2)
        u = u.permute(0, 2, 1)

        r_sents = self.conv_sentence(u)
        r_sents, _ = torch.max(r_sents, 2)

        out = torch.tanh(self.linear_1(r_sents))
        out = self.linear_2(out)

        return out
