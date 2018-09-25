import torch
import torch.nn as nn
from conv_network import ConvNet
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer
from vocab import Vocab, UnkVocab
import numpy as np
import string
from random import shuffle
import numpy as np
import sys
from DataLoader_CharSCNN import DataLoader

use_LSTM = False

'''
STEP 1: LOADING DATASET
'''

dataloader = DataLoader(limit=50)
dataloader.load_train_comments()
dataloader.load_test_comments()

sentence_list = np.concatenate((x_train, x_test))
word_list = []
for sentence in sentence_list:
    word_list.extend(sentence.split())
word_list = list(set(word_list))

word_vocabulary = Vocab()
word_vocabulary.word2index(word_list, train=True)

char_list = []
for sentence in sentence_list:
    char_list.extend([char for char in sentence])
char_list = list(set(char_list))
# char_list.extend(string.digits)
# char_list.extend(string.ascii_letters)
# char_list.extend(string.punctuation)
char_vocabulary = Vocab()
char_vocabulary.word2index(char_list, train=True)

x_train_indices = []
x_test_indices = []
x_train_char_indices = []
x_test_char_indices = []

# Padding inndex is zero, so we shift everything with plus 1

# Word indices
for sentence in x_train:
    words = sentence.split()
    indices = [word_vocabulary.word2index(word) + 1 for word in words]
    x_train_indices.append(indices)
for sentence in x_test:
    words = sentence.split()
    indices = [word_vocabulary.word2index(word.lower()) + 1 for word in words]
    x_test_indices.append(indices)

# Char indices
for sentence in x_train:
    word_char = []
    words = sentence.split()
    for word in words:
        indices = [char_vocabulary.word2index(char) + 1 for char in word]
        indices = torch.tensor(indices, dtype=torch.int64)
        word_char.append(indices)
    x_train_char_indices.append(word_char)
for sentence in x_test:
    word_char = []
    words = sentence.split()
    for word in words:
        indices = [char_vocabulary.word2index(char) + 1 for char in word]
        indices = torch.tensor(indices, dtype=torch.int64)
        word_char.append(indices)
    x_test_char_indices.append(word_char)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

word_vocab_size = len(word_list)
chr_vocab_size = len(char_list)

# Hyperparameters
learning_rate = 0.01
eval_freq = 100

print("Model is running on: ", device)
model = ConvNet(word_vocab_size + 1, chr_vocab_size + 1).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = 0
for i in range(1000):
    x_wrd, x_chr, y, y_score = dataloader.train_data.next_batch(5)
    optimizer.zero_grad()
    sentence_word_vectors = torch.LongTensor(torch.LongTensor(x_train_indices[i]))
    sentence_chr_vectors = x_train_char_indices[i]
    output = model.forward(sentence_word_vectors, sentence_chr_vectors)
    output = output.reshape(1, -1)
    y = torch.tensor([y_train[i]], dtype=torch.int64)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    loss += loss.item()
    if i % 50 == 0:
        print("Current iteration: ", i + 1)
        print("Average loss: {}".format(loss / (i + 1)))
    if i % eval_freq == 0:
        correct_predictions = 0
        positive_predictions = 0
        negative_predictions = 0
        for step in range(len(x_test)):
            sentence_word_vectors = torch.LongTensor(torch.LongTensor(x_test_indices[step]))
            sentence_chr_vectors = x_test_char_indices[step]
            output = model.forward(sentence_word_vectors, sentence_chr_vectors)
            if torch.argmax(output).data.numpy() == y_test[step]:
                correct_predictions += 1
            if torch.argmax(output).data.numpy() == 1:
                positive_predictions += 1
            else:
                negative_predictions += 1
        print("Accuracy is: ", correct_predictions / len(x_test))

        print("Number of positive predictions is: ", positive_predictions)
        print("Number of negative predictions is: ", negative_predictions)




# chr_vector = torch.LongTensor(x_train_char_indices[0])


#
