import torch
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import argparse

from sklearn.feature_extraction.text import CountVectorizer
# from vocab import Vocab, UnkVocab
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
from tensorboardX import SummaryWriter

def save_mistakes(output, labels, inputs, dl):
    predictions = torch.argmax(output,dim=1)
    correct_predictions = torch.eq(predictions, batch_targets)

    mistakes_indices = torch.where(correct_predictions == 0)
    comments = [(dl.train_data.word2idx[idx], batch_targets[index]) for index in mistakes_indices for idx in inputs[index] if idx != 0]

    with open('mistaken_comments.txt', 'w') as wf:
        for comment in comments:
            truth = 'positive' if comment[0] == 1 else 'negative'
            pred = 'positive' if comment[0] == 0 else 'negative'
            wf.write('The following comment was predicted as {} but truely was {}. \n\n{}'.format(pred, truth, comment[1]))
    return


def calc_accuracy(output, batch_targets):
    predictions = torch.argmax(output,dim=1)
    correct_predictions = torch.eq(predictions, batch_targets).sum()
    return float(correct_predictions.item() / len(batch_targets)) * 100


def train(dl, config):

    writer = SummaryWriter(config.summary_path)

    total_loss = 0
    word_seq_size = dl.train_data.seq_size_words
    chr_seq_size = dl.train_data.seq_size_chars

    if config.use_LSTM:
        model = LSTMModel(input_dim, config.hidden_dim, config.num_layers, config.output_dim)
    else:
        model = RNNModel(word_seq_size, config.hidden_dim, config.num_layers, config.output_dim)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

        # Load checkpoint
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print("Checkpoint loaded")


    for index in range(config.n_iters):
        t1 = time.time()
        print('Starting on iteration {}'.format(index+1))

        # load new batch
        batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores = dl.train_data.next_batch(config.batch_size)
        # print(batch_inputs_words.shape)
        if torch.cuda.is_available():
            batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores = batch_inputs_words.cuda(), batch_inputs_chars.cuda(), batch_targets_label.cuda(), batch_targets_scores.cuda()
        batch_inputs_words = batch_inputs_words.t().reshape(config.batch_size, word_seq_size, 1)

        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(batch_inputs_words)

        # Calculate Loss: softmax --> cross entropy loss
        label = batch_targets_label.type('torch.LongTensor').reshape(-1)
        loss = criterion(outputs, label)
        total_loss += loss.item()

        acc = calc_accuracy(outputs, label)

        # niter = epoch*len(data_loader)+step
        writer.add_scalar('accuracy', acc, index)
        writer.add_scalar('loss', loss.item(), index)


        # Getting gradients w.r.t. parameters
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)

        # Updating parameters
        optimizer.step()
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if index % config.print_every == 0:
            print('[{}]\t Step {}\t Loss {} \t Examples/Sec = {:.2f},'.format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                            index, total_loss, examples_per_second))
            total_loss = 0

        if index % config.save_every == 0:
            save_checkpoint(model, optimizer)

        if index % config.test_every == 0:
            test(dl, index, model, config.test_size)


def test(dl, step, model, test_size=1000):
    word_seq_size = dl.test_data.seq_size_words
    chr_seq_size = dl.train_data.seq_size_chars

    criterion = nn.CrossEntropyLoss()
    t1 = time.time()
    # load new batch
    batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores = dl.test_data.next_batch(test_size)

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
    examples_per_second = config.batch_size/float(t2-t1)

    print('Here the test results after {} steps.\n[{}]\t Loss {} \t Acc {} \t Examples/Sec = {:.2f},'.format(step, datetime.now().strftime("%Y-%m-%d %H:%M"),
                    loss.item(), acc, examples_per_second))

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  print('\n')
  for key, value in vars(config).items():
    print(key + ' : ' + str(value))
  print('\n')

def save_checkpoint(model, optimizer, filename='checkpoints_dl4nlt/checkpoint.pth.tar'):
    if not os.path.exists('./checkpoints_dl4nlt'):
        os.mkdir('./checkpoints_dl4nlt')
    filename = "checkpoints_dl4nlt/checkpoint_{}.pth.tar".format(datetime.now().strftime("%d_%m_%H_%M"),)
    checkpoint = {
        # 'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        }
    torch.save(checkpoint, filename)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    return checkpoint

def evaluate(dl, config):
    word_seq_size = dl.train_data.seq_size_words
    chr_seq_size = dl.train_data.seq_size_chars

    if config.use_LSTM:
        model = LSTMModel(input_dim, config.hidden_dim, config.num_layers, config.output_dim)
    else:
        model = RNNModel(word_seq_size, config.hidden_dim, config.num_layers, config.output_dim)

    if torch.cuda.is_available():
        model.cuda()

        # Load checkpoint
    if config.checkpoint:
        checkpoint = load_checkpoint(config.model_path)
        model.load_state_dict(checkpoint['model'])
        print("Checkpoint loaded")

    batch_inputs_words,_, batch_targets_label,_ = dl.test_data.next_batch()

    batch_inputs_words = batch_inputs_words.t().reshape(-1, word_seq_size, 1)
    # Forward pass to get output/logits
    outputs = model(batch_inputs_words)

    # Calculate Loss: softmax --> cross entropy loss
    label = batch_targets_label.type('torch.LongTensor').reshape(-1)
    acc = calc_accuracy(outputs, label)

    save_mistakes(outputs, label, batch_inputs_words)

    t2 = time.time()
    examples_per_second = config.batch_size/float(t2-t1)

    print('Here the test results after {} steps.\n[{}] \t Acc {} \t Examples/Sec = {:.2f},'.format(step, datetime.now().strftime("%Y-%m-%d %H:%M"),
                     acc, examples_per_second))



def main(config):
    limit = 20
    data_loader_filename = 'data/dataloader_{}.p'.format(limit)


    if os.path.isfile(data_loader_filename):
        with open(data_loader_filename, 'rb') as rf:
            dl = pickle.load(rf)
    else:
        dl = DataLoader(limit=limit)
        dl.load_train_comments()
        dl.load_test_comments(dl.train_data)
        dl.load_twitter_comments()
        with open(data_loader_filename, 'wb') as wf:
            pickle.dump(dl, wf)

    num_epochs = config.n_iters / (dl.train_data.num_examples / config.batch_size)
    num_epochs = int(num_epochs)

    if not config.testing:
        train(dl, config)
    else:
        evaluate(dl, config)

if __name__ == "__main__":


    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--use_LSTM', type=bool, default=False, help='To use an LSTM instead of RNN')
    parser.add_argument('--hidden_dim', type=int, default=20, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of stacked RNN/LSTM layers in the model')
    parser.add_argument('--output_dim', type=int, default=2, help='Output dimension of the model')
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension of the model')

    # Training params
    parser.add_argument('--use_padding', type=bool, default=False, help='To use padding on input sentences.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--n_iters', type=int, default=3000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries_dl4nlt/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=100, help='How often to print training progress')
    parser.add_argument('--test_every', type=int, default=100, help='How often to test the model')
    parser.add_argument('--save_every', type=int, default=100, help='How often to save checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of samples in the test')

    # Test Args
    parser.add_argument('--testing', type=bool, default=False, help='Will the network train or only perform a test')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to test')

    config = parser.parse_args()

    print_flags()

    # Train the model
    main(config)
