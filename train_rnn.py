import os
import time
import pickle
import argparse
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn

from RNN import RNNModel
from LSTM import LSTMModel
from src.DataLoader_CharSCNN import DataLoader


def save_mistakes(output, labels, input_words, dl):
    """ Given the predictions performed on the test set, save examples of
    mispredictions to do qualitative evaluation.
    """
    predictions = torch.argmax(output, dim=1)
    correct_predictions = torch.eq(predictions, labels)

    mistakes_indices = correct_predictions == 0
    wrong_comments = []
    correct_comments = []
    dl.train_data.idx2word[len(dl.train_data.idx2word) + 1] = "UNKNOWN_WORD"

    for i, mistake in enumerate(mistakes_indices):
        if mistake == True:
            comment = ' '.join([dl.train_data.idx2word[int(idx)] for idx in input_words[i] if idx != 0])
            wrong_comments.append((comment, labels[i]))
        else:
            comment = ' '.join([dl.train_data.idx2word[int(idx)] for idx in input_words[i] if idx != 0])
            correct_comments.append((comment, labels[i]))
    with open('mistaken_comments.txt', 'w') as wf:
        for comment in wrong_comments:
            truth = 'positive' if comment[1] == 1 else 'negative'
            pred = 'positive' if comment[1] == 0 else 'negative'
            wf.write('The following comment was predicted as {} but truely was {}. \n{}'.format(pred, truth,
                                                                                                comment[0] + "\n\n"))

    with open('correctly_comments.txt', 'w') as wf:
        for comment in correct_comments:
            pred = 'positive' if comment[1] == 1 else 'negative'
            wf.write('The following comment was correctly predicted as {}  \n{}'.format(pred, comment[0] + "\n\n"))

    TP, TN, FP, FN = 0, 0, 0, 0

    for index in range(len(predictions)):
        if predictions[index] == 1 and labels[index] == 1:
            TP += 1
        elif predictions[index] == 0 and labels[index] == 0:
            TN += 1
        elif predictions[index] == 0 and labels[index] == 1:
            FN += 1
        else:
            FP += 1

    print("Number of true positives: ", TP)
    print("Number of true negatives: ", TN)
    print("Number of false negatives: ", FN)
    print("Number of false positives: ", FP)
    print("Precision for Positive is: ", TP / (TP + FP))
    print("Recall for Positive is: ", TP / (TP + FN))
    print("Precision for negative is: ", TN / (TN + FN))
    print("Recall for negative is: ", TN / (TN + FP))


def calc_accuracy(output, batch_targets):
    """ Calculate the accuracy of a prediction given labels
    """
    predictions = torch.argmax(output, dim=1)
    correct_predictions = torch.eq(predictions, batch_targets).sum()
    return float(correct_predictions.item() / len(batch_targets)) * 100


def train(dl, config):
    """ Train the model given the parameters in the config object
    """
    writer = SummaryWriter(config.summary_path)

    total_loss = 0
    word_vocab_size = dl.train_data.vocab_size_words + 1

    if config.use_LSTM:
        model = LSTMModel(word_vocab_size, config.hidden_dim, config.num_layers, config.output_dim)
    else:
        model = RNNModel(word_vocab_size, config.hidden_dim, config.num_layers, config.output_dim)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # Load checkpoint
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # epoch = checkpoint['epoch']
        print("Checkpoint loaded")

    best_acc = [0]
    for index in range(config.n_iters):
        t1 = time.time()

        # load new batch
        batch_inputs_words, _, batch_targets_label, _, batch_lengths \
            = dl.train_data.next_batch(config.batch_size, padding=True, type='long')

        if torch.cuda.is_available():
            batch_inputs_words, batch_targets_label, batch_lengths \
                = batch_inputs_words.cuda(), batch_targets_label.cuda(), batch_lengths.cuda()

        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(batch_inputs_words, batch_lengths)

        # Calculate Loss: softmax --> cross entropy loss
        label = batch_targets_label.squeeze()
        loss = criterion(outputs, label)
        total_loss += loss.item()

        acc = calc_accuracy(outputs, label)

        # Write the outcomes to the tensorboard
        writer.add_scalar('accuracy', acc, index)
        writer.add_scalar('loss', loss.item(), index)

        # Getting gradients w.r.t. parameters
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)

        # Updating parameters
        optimizer.step()
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if index % config.print_every == 0:
            print('[{}]\t Step {}\t Loss {} \t Examples/Sec = {:.2f},'.format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                              index, total_loss / config.print_every,
                                                                              examples_per_second))
            total_loss = 0

        if index % config.save_every == 0:
            save_checkpoint(model, optimizer, config.checkpoint_path)

        if index % config.test_every == 0:
            test_acc = test(dl, index, model, test_size=config.test_size)
            writer.add_scalar('test_accuracy', test_acc, index)
            if acc > max(best_acc):
                torch.save(model, 'best_model.pt')
            best_acc.append(acc)


def test(dl, step, model, test_size=1000, validation=False):
    """ Test the model on part of the test set
    """
    criterion = nn.CrossEntropyLoss()
    t1 = time.time()

    if validation == True:
        model = torch.load('best_model.pt')

    # load new batch
    if validation == False:
        batch_inputs_words, _, batch_targets_label, _, batch_lengths = dl.test_data.next_batch(test_size, padding=True, type='long')
    else:
        batch_inputs_words, _, batch_targets_label, _, batch_lengths = dl.val_data.next_batch(test_size, padding=True,
                                                                                               type='long')


    if torch.cuda.is_available():
        batch_inputs_words, batch_targets_label, batch_lengths \
            = batch_inputs_words.cuda(), batch_targets_label.cuda(), batch_lengths.cuda()

    # Forward pass to get output/logits
    outputs = model(batch_inputs_words, batch_lengths)

    # Calculate Loss: softmax --> cross entropy loss
    if torch.cuda.is_available():
        label = batch_targets_label.type('torch.cuda.LongTensor').squeeze()
    else:
        label = batch_targets_label.type('torch.LongTensor').squeeze()

    if validation == True:
        save_mistakes(outputs, label, batch_inputs_words, dl)

    loss = criterion(outputs, label)
    acc = calc_accuracy(outputs, label)

    t2 = time.time()
    examples_per_second = config.batch_size / float(t2 - t1)

    print(
        'Here the test results after {} steps.\n[{}]\t Loss {} \t Acc {} \t Examples/Sec = {:.2f}, pos_labels: {}, neg_labels: {}'.format(
            step, datetime.now().strftime("%Y-%m-%d %H:%M"),
            loss.item(), acc, examples_per_second, torch.sum(label), len(label) - torch.sum(label)))
    return acc


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    print('\n')
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))
    print('\n')


def save_checkpoint(model, optimizer, path):
    """ Save the trained model checkpoint
    """
    if not os.path.exists(path):
        os.mkdir(path)
    filename = "{}/checkpoint_{}.pth.tar".format(path, datetime.now().strftime("%d_%m_%H_%M"),)
    checkpoint = {
        # 'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        }
    torch.save(checkpoint, filename)


def load_checkpoint(filepath):
    """ Load a previously trained checkpoint
    """
    checkpoint = torch.load(filepath)
    return checkpoint


def evaluate(dl, config):
    """ Given a trained network, evaluate on the whole test set and receive outcomes for qualitative analysis
    """
    t1 = time.time()
    word_seq_size = dl.train_data.seq_size_words

    if config.use_LSTM:
        model = LSTMModel(word_seq_size, config.hidden_dim, config.num_layers, config.output_dim)
    else:
        model = RNNModel(word_seq_size, config.hidden_dim, config.num_layers, config.output_dim)

    if torch.cuda.is_available():
        model.cuda()

    # Load checkpoint
    if config.checkpoint:
        checkpoint = load_checkpoint(config.model_path)
        model.load_state_dict(checkpoint['model'])
        print("Checkpoint loaded")

    batch_inputs_words, _, batch_targets_label, _, batch_lengths = dl.test_data.next_batch()
    batch_inputs_words = batch_inputs_words.t().reshape(-1, word_seq_size, 1)

    # Forward pass to get output/logits
    outputs = model(batch_inputs_words, batch_lengths)

    # Calculate Loss: softmax --> cross entropy loss
    label = batch_targets_label.type('torch.LongTensor').reshape(-1)
    acc = calc_accuracy(outputs, label)

    save_mistakes(outputs, label, batch_inputs_words, dl)

    t2 = time.time()
    examples_per_second = config.batch_size / float(t2 - t1)

    print('Here the test results.\n[{}] \t Acc {} \t Examples/Sec = {:.2f},'.format(
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        acc, examples_per_second))

def test_eval(dl, step, model, test_size=1000, validation=False, config=None, optimizer=None):
    word_seq_size = dl.test_data.seq_size_words
    chr_seq_size = dl.train_data.seq_size_chars
    max_sen_len = max(dl.train_data.seq_size_words, dl.test_data.seq_size_words, dl.val_data.seq_size_words)
    dl.val_data.seq_size_words = max_sen_len
    if validation == True:
        # # Load checkpoint
        if config.checkpoint:
            checkpoint = torch.load(config.checkpoint)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # epoch = checkpoint['epoch']
            print("Checkpoint loaded")
        else:
            print("No checkpoint supplied!")

    criterion = nn.CrossEntropyLoss()
    t1 = time.time()
    # load new batch
    if validation == False:
        batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores, batch_lengths = dl.test_data.next_batch(
            test_size, padding=True, type='long')
    else:
        batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores, batch_lengths = dl.val_data.next_batch(
            test_size, padding=True, type='long')

    if torch.cuda.is_available():
        batch_inputs_words, batch_inputs_chars, batch_targets_label, batch_targets_scores, batch_lengths = batch_inputs_words.cuda(), batch_inputs_chars.cuda(), batch_targets_label.cuda(), batch_targets_scores.cuda()

    # Forward pass to get output/logits
    outputs = model(batch_inputs_words, batch_lengths)

    # Calculate Loss: softmax --> cross entropy loss
    if torch.cuda.is_available():
        label = batch_targets_label.type('torch.cuda.LongTensor').squeeze()
    else:
        label = batch_targets_label.type('torch.LongTensor').squeeze()
    if validation == True:
        save_mistakes(outputs, label, batch_inputs_words, dl)

    loss = criterion(outputs, label)
    acc = calc_accuracy(outputs, label)

    t2 = time.time()
    examples_per_second = config.batch_size / float(t2 - t1)

    print(
        'Here the test results after {} steps.\n[{}]\t Loss {} \t Acc {} \t Examples/Sec = {:.2f}, pos_labels: {}, neg_labels: {}'.format(
            step, datetime.now().strftime("%Y-%m-%d %H:%M"),
            loss.item(), acc, examples_per_second, torch.sum(label), len(label) - torch.sum(label)))

    return loss, acc



def main(config):
    # Limit defines the size of the dataset, used for testing the code
    limit = 0
    data_loader_filename = '{}dataloader_twitter_{}.p'.format(config.data_path, limit)
    if os.path.isfile(data_loader_filename):
        with open(data_loader_filename, 'rb') as rf:
            dl = pickle.load(rf)
    else:
        dl = DataLoader(limit=limit, data_path=config.data_path)
        dl.load_twitter_comments()
        with open(data_loader_filename, 'wb') as wf:
            pickle.dump(dl, wf)

    dl.train_data.seq_size_words = max(dl.train_data.seq_size_words, dl.test_data.seq_size_words)

    if not config.testing:
        train(dl, config)
        loss, acc = test(dl, 0, 0, test_size=500, validation=True)
        print("The accuracy on the validation set is: ", acc)
    else:
        word_vocab_size = dl.train_data.vocab_size_words + 1
        if config.use_LSTM:
            model = LSTMModel(word_vocab_size, config.hidden_dim, config.num_layers, config.output_dim)
        else:
            model = RNNModel(word_vocab_size, config.hidden_dim, config.num_layers, config.output_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        loss, acc = test_eval(dl, 0, model, test_size=500, validation=True, config=config, optimizer=optimizer)
        print("The accuracy on the validation set is: ", acc)
        # evaluate(dl, config)


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--use_LSTM', type=int, default=0, help='To use an LSTM instead of RNN')
    parser.add_argument('--hidden_dim', type=int, default=20, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of stacked RNN/LSTM layers in the model')
    parser.add_argument('--output_dim', type=int, default=2, help='Output dimension of the model')
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension of the model')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_dl4nlt')

    # Training params
    parser.add_argument('--use_padding', type=int, default=0, help='To use padding on input sentences.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--n_iters', type=int, default=3000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries_dl4nlt/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=10, help='How often to print training progress')
    parser.add_argument('--test_every', type=int, default=100, help='How often to test the model')
    parser.add_argument('--save_every', type=int, default=100, help='How often to save checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of samples in the test')

    # Test Args
    parser.add_argument('--testing', type=int, default=0, help='Will the network train or only perform a test')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to test')

    config = parser.parse_args()

    print_flags()

    # Train the model
    main(config)
