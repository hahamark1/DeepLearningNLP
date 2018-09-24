import torch 
import torch.nn as nn
from conv_network import ConvNet
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer
from vocab import Vocab, UnkVocab
import numpy as np
from DataLoader import DataLoader
import string
from random import shuffle
import numpy as np
import sys

use_LSTM = False

'''
STEP 1: LOADING DATASET
'''

dataloader = DataLoader(data_limit=5)
x_train, y_train, x_test, y_test = dataloader.get_comments()

sentence_list = np.concatenate((x_train, x_test))
word_list = []
for sentence in sentence_list:
	word_list.extend(sentence.split(' '))
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
	words = sentence.split(' ')
	indices = [word_vocabulary.word2index(word) +  1 for word in words]
	x_train_indices.append(indices)
for sentence in x_test:
	words = sentence.split(' ')
	indices = [word_vocabulary.word2index(word.lower()) + 1 for word in words]
	x_test_indices.append(indices)

# Char indices
for sentence in x_train:
	word_char = []
	words = sentence.split(' ')
	for word in words:

		indices = [char_vocabulary.word2index(char) + 1 for char in word]
		word_char.append(indices)
	x_train_char_indices.append(word_char)
for sentence in x_test:
	word_char = []
	words = sentence.split(' ')
	for word in words:
		indices = [char_vocabulary.word2index(char) + 1 for char in word]
		word_char.append(indices)
	x_test_char_indices.append(word_char)

if torch.cuda.is_available():
	device = "cuda:0"
else:
	device = "cpu"

word_vocab_size = len(word_list)
chr_vocab_size = len(char_list)

learning_rate = 0.01
print("Model is running on: ", device)
model = ConvNet(word_vocab_size, chr_vocab_size + 1).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Testing
# print(x_train[0])
# print(x_train_indices[0])
# print("-" * 20)
print(x_train_char_indices[0])
# sys.exit()
for i in range(0, 10):
	word_vector = torch.LongTensor(torch.LongTensor(x_train_indices[i]))
	chr_vector = x_train_char_indices[i]
	model.forward(word_vector, chr_vector)
# chr_vector = torch.LongTensor(x_train_char_indices[0])


#



