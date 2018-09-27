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
from src.DataLoader_CharSCNN import DataLoader

use_LSTM = False

'''
STEP 1: LOADING DATASET
'''

dl = DataLoader(limit=50)
dl.load_train_comments()
dl.load_test_comments()

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

word_seq_size = dl.train_data.seq_size_words
chr_seq_size = dl.train_data.seq_size_chars

# Hyperparameters
learning_rate = 0.01
eval_freq = 100

print("Model is running on: {}".format(device))
model = ConvNet(word_seq_size, chr_seq_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = 0
for i in range(1000):
    # # TODO: The line below will not work because padding is false is not yet implemented
    batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores = dl.train_data.next_batch(5, padding=False)
    optimizer.zero_grad()
    output = model.forward(batch_inputs_words, batch_inputs_chars)
    output = output.reshape(1, -1)
    y = batch_targets_label.type('torch.LongTensor').reshape(-1)
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
