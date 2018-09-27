import torch
import os
import time
from datetime import datetime
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer
from vocab import Vocab, UnkVocab
import numpy as np
# from src.DataLoader import DataLoader
import string
from random import shuffle
import pickle
from src.DataLoader_CharSCNN import DataLoader
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize


USE_PADDING = False
use_LSTM = False


'''
STEP 1: LOADING DATASET
'''
limit = 500
data_loader_filename = 'data/dataloader_{}.p'.format(limit)

if os.path.isfile(data_loader_filename):
    with open(data_loader_filename, 'rb') as rf:
        dl = pickle.load(rf)
else:
    dl = DataLoader(limit=limit)
    dl.load_train_comments()
    dl.load_test_comments()
    with open(data_loader_filename, 'wb') as wf:
        pickle.dump(dl, wf)

print(dl.train_data.vocab_size_words)

# comments = dl.train_data.comments
#
# comments = [[word_tokenize(sent) for sent in sent_tokenize(comment.replace('<br />',''))] for comment in comments]
# comments = [[word for sentence in comment for word in sentence] for comment in comments]
# comment_lengths = [len(comment) for comment in comments]
# print(max(comment_lengths))
# a = np.array(comment_lengths)
# p = np.percentile(a, 95) # return 50th percentile, e.g median.
# print(p)
# print(dl.train_data.max_sent_len)
# sentence_list = np.concatenate((x_train, x_test))
# word_list = []
# for sentence in sentence_list:
#     word_list.extend(sentence.split(' '))
# word_list = list(set(word_list))
#
# word_vocabulary = Vocab()
# word_vocabulary.word2index(word_list, train=True)
#
# char_list = []
# char_list.extend(string.digits)
# char_list.extend(string.ascii_letters)
# char_list.extend(string.punctuation)
# char_vocabulary = Vocab()
# char_vocabulary.word2index(char_list, train=True)

# x_train_indices = []
# x_test_indices = []


# for sentence in x_train:
#     words = sentence.split(' ')
#     indices = [word_vocabulary.word2index(word) + 1 for word in words]
#     x_train_indices.append(indices)
#
#
#
# for sentence in x_test:
#     words = sentence.split(' ')
#     indices = [word_vocabulary.word2index(word) + 1 for word in words]
#     x_test_indices.append(indices)


'''
STEP 2: MAKING DATASET ITERABLE
'''


batch_size = 128
n_iters = 3000
num_epochs = n_iters / (dl.train_data.num_examples / batch_size)
num_epochs = int(num_epochs)

'''
STEP 3: CREATE MODEL CLASS
'''

class RNNModel(nn.Module):
    def __init__(self, word_vocab_size, hidden_dim, layer_dim, output_dim, dim_embedding=512):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Linear(1, dim_embedding)
        # .type(torch.LongTensor)
        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
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
        x = x.type(torch.FloatTensor)
        out, hn = self.rnn(x, h0)

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 2
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

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

        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 2
        return out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 1
hidden_dim = 20
layer_dim = 3  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 2


word_seq_size = dl.train_data.seq_size_words
chr_seq_size = dl.train_data.seq_size_chars

if use_LSTM:
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
else:
    model = RNNModel(word_seq_size, hidden_dim, layer_dim, output_dim)
print(model)

#######################
#  USE GPU FOR MODEL  #
#######################

if torch.cuda.is_available():
    model.cuda()

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''

n_epochs = 25
max_steps = 10
total_loss = 0
for index in range(n_iters):
    t1 = time.time()
    print('Starting on iteration {}'.format(index+1))
    # Clear gradients w.r.t. parameters
    batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores = dl.train_data.next_batch(batch_size)
    # batch_targets = torch.stack(batch_targets).reshape(-1)
    # one_hot = torch.zeros(word_seq_size, batch_size, dl.train_data.vocab_size_words)
    # Fill the empty array with one hot vectors
    # batch_inputs_words = one_hot.scatter_(2, batch_inputs, 1)
    batch_inputs_words = batch_inputs_words.t().reshape(batch_size, word_seq_size, 1)

    optimizer.zero_grad()
    # Forward pass to get output/logits

    # outputs.size() --> 100, 10
    # batch = np.array(batch_inputs_words)
    # batch = batch.reshape((1, len(item), 1))
    # batch = torch.tensor(batch, dtype=torch.float)

    outputs = model(batch_inputs_words)

    # Calculate Loss: softmax --> cross entropy loss
    label = batch_targets_label.type('torch.LongTensor').reshape(-1)
    loss = criterion(outputs, label)
    total_loss += loss.item()

    # Getting gradients w.r.t. parameters
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)

    # Updating parameters
    optimizer.step()
    t2 = time.time()
    examples_per_second = batch_size/float(t2-t1)

    if index % 1 == 0:
        print('[{}]\t Step {}\t Loss {} \t Examples/Sec = {:.2f},'.format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                        index, total_loss, examples_per_second))
        total_loss = 0
