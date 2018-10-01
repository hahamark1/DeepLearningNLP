import torch
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from sklearn.feature_extraction.text import CountVectorizer
from vocab import Vocab, UnkVocab
import numpy as np
# from src.DataLoader import DataLoader
from RNN import RNNModel
from LSTM import LSTMModel
import string
from random import shuffle
import pickle
from src.DataLoader_CharSCNN import DataLoader
import nltk
nltk.download('punkt')


from nltk.tokenize import word_tokenize, sent_tokenize

'''
HYPERPARAMETERS
'''
USE_PADDING = False
use_LSTM = False
input_dim = 1
hidden_dim = 20
layer_dim = 3  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 2
batch_size = 16
n_iters = 3000

n_epochs = 25
max_steps = 10

learning_rate = 0.1

def calc_accuracy(output, batch_targets):
    predictions = torch.argmax(output,dim=1)
    correct_predictions = torch.eq(predictions, batch_targets).sum()
    return float(correct_predictions.item() / len(batch_targets)) * 100


def train(dl, num_epochs):
    total_loss = 0
    word_seq_size = dl.train_data.seq_size_words
    chr_seq_size = dl.train_data.seq_size_chars

    if use_LSTM:
        model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    else:
        model = RNNModel(word_seq_size, hidden_dim, layer_dim, output_dim)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for index in range(n_iters):
        t1 = time.time()
        print('Starting on iteration {}'.format(index+1))
        
        # load new batch
        batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores = dl.train_data.next_batch(batch_size)
        if torch.cuda.is_available():
            batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores = batch_inputs_words.cuda(), batch_inputs_chars.cuda(), batch_targets_label.cuda(), batch_targets_scores.cuda()
        batch_inputs_words = batch_inputs_words.t().reshape(batch_size, word_seq_size, 1)

        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(batch_inputs_words)

        # Calculate Loss: softmax --> cross entropy loss
        label = batch_targets_label.type('torch.LongTensor').reshape(-1)
        loss = criterion(outputs, label)
        total_loss += loss.item()

        acc = calc_accuracy(outputs, label)

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

        if index % 10 == 0:
            test(dl, index, model)


def test(dl, step, model):
    word_seq_size = dl.test_data.seq_size_words
    chr_seq_size = dl.train_data.seq_size_chars

    criterion = nn.CrossEntropyLoss()
    t1 = time.time()
    # load new batch
    batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores = dl.test_data.next_batch(dl.test_data.num_examples)

    if torch.cuda.is_available():
        batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores = batch_inputs_words.cuda(), batch_inputs_chars.cuda(), batch_targets_label.cuda(), batch_targets_scores.cuda()
    batch_inputs_words = batch_inputs_words.t().reshape(-1, word_seq_size, 1)
    # Forward pass to get output/logits
    outputs = model(batch_inputs_words)

    # Calculate Loss: softmax --> cross entropy loss
    label = batch_targets_label.type('torch.LongTensor').reshape(-1)
    loss = criterion(outputs, label)
    acc = calc_accuracy(outputs, label)

    t2 = time.time()
    examples_per_second = batch_size/float(t2-t1)

    print('Here the test results after {} steps.\n[{}]\t Loss {} \t Acc {} \t Examples/Sec = {:.2f},'.format(step, datetime.now().strftime("%Y-%m-%d %H:%M"),
                    loss.item(), acc, examples_per_second))


def main():
    limit = 16
    data_loader_filename = 'data/dataloader_{}.p'.format(limit)

    if os.path.isfile(data_loader_filename):
        with open(data_loader_filename, 'rb') as rf:
            dl = pickle.load(rf)
    else:
        dl = DataLoader(limit=limit)
        dl.load_train_comments()
        dl.load_test_comments(dl.train_data)
        with open(data_loader_filename, 'wb') as wf:
            pickle.dump(dl, wf)

    num_epochs = n_iters / (dl.train_data.num_examples / batch_size)
    num_epochs = int(num_epochs)

    train(dl, num_epochs)

if __name__ == '__main__':
    main()
