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

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, words, words_in_char, sentence_vector=0):
        # Create word embeddings
        vector_wrds = []
        for word in words:
            word = torch.tensor(word, dtype=torch.int64)
            vector_wrd = self.word_embedding(word)
            vector_wrds.append(vector_wrd)

        # Create character word embeddings
        vector_wchs = []
        for word_char in words_in_char:
            vector_chr = []
            if len(word_char) == 0:
                print('Continue please')
                continue

            for char in word_char:
                chr_emb = self.chr_embedding(torch.tensor(char, dtype=torch.int64))
                vector_chr.append(chr_emb)
            vector_chr = torch.stack(vector_chr)
            vector_chr = vector_chr.reshape(1, self.dim_embedding, -1)
            word_convolution = self.conv_chr(vector_chr)
            word_convolution = word_convolution.squeeze().reshape(self.dim_embedding, -1)
            vector_wch = torch.max(word_convolution, 1)[0]
            vector_wchs.append(vector_wch)

        # Combine word and character word embeddings
        u = []
        for i in range(len(vector_wrds)):
            u_i = torch.cat((vector_wrds[i], vector_wchs[i]), 0)
            u.append(u_i)

        r_sents = []
        for u_i in u:
            r_sent = self.conv_sentence(u_i.reshape(1, self.dim_embedding * 2, -1)).squeeze()
            r_sents.append(r_sent)
        r_sents = torch.stack(r_sents)
        r_sents = torch.max(r_sents, 0)[0]

        out = F.tanh(self.linear_1(r_sents))
        out = self.linear_2(out)

        return out
