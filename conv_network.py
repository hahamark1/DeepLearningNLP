import torch 
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, word_vocab_size, chr_vocab_size, dim_embedding=128):
        super(ConvNet, self).__init__()
        self.dim_embedding = dim_embedding
        print("Maximum character size:, ", chr_vocab_size)
        self.word_embedding = nn.Embedding(word_vocab_size, dim_embedding, padding_idx = 0)
        self.chr_embedding = nn.Embedding(chr_vocab_size, dim_embedding, padding_idx = 0)
        self.conv_chr = nn.Conv1d(dim_embedding, dim_embedding, 3, padding=1)
        self.conv_word = nn.Conv1d(dim_embedding, dim_embedding, 3, padding=1)

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
        
    def forward(self, word_vector, words_in_char, sentence_vector=0):
        # word_emb = self.word_embedding(word_vector)
        for word_char in words_in_char:
            char_input = []
            for char in word_char:
                chr_emb = self.chr_embedding(torch.tensor(char, dtype=torch.int64))
                char_input.append(chr_emb)
            char_input = torch.stack(char_input).reshape(1, self.dim_embedding, -1)
            word_convolution = self.conv_chr(char_input)
            word_max_conv = torch.argmax(word_convolution.squeeze(), 1)
            print(word_max_conv.shape)
        out = self.conv_word(word_emb.reshape(1, self.dim_embedding, -1))
        # print(chr_emb.shape)
        # chr_emb = chr_emb.view(1, 1, self.dim_embedding, chr_emb.shape[0])
        # out = torch.max(self.conv_chr(chr_emb).squeeze(), 0)
        print(out.shape)
        return out
