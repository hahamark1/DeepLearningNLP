import os
import csv
import pickle

from src.Preprocessor import PreProcessor
from random import shuffle

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

DATA_PATH = '../data'
TRAIN_DATA_PATH = '{}/train'.format(DATA_PATH)
TEST_DATA_PATH = '{}/test'.format(DATA_PATH)
VOCAB_PATH = '{}/imdb.vocab'.format(DATA_PATH)
WORD_SENTIMENT_PATH = '{}/imdbEr.txt'.format(DATA_PATH)
TWITTER_PATH = '{}/twitter_sentiment.cxv'.format(DATA_PATH)
DATALOADER_PATH = '{}/dataloaders.p'.format(DATA_PATH)

COMMENTS = ['neg', 'pos']
BATCH_SIZE = 32

class DataLoader(object):
    """ Object that loads all formats from the datafolder to Python Object
    """
    def __init__(self, data_limit=0):
        super(DataLoader, self).__init__()

        # The dicts below follow the format:
        # {Comment_ID: [comment, pos/neg, sentiment_score]}

        self.data_limit = data_limit

        self.train_comments = {}
        self.test_comments = {}
        self.vocabulaire = []
        self.word_sentiment = {}
        self.twitter_comments = {}

        self.test_loader = 0

        self.load_train_comments()
        self.load_test_comments()
        self.train_loader, self.val_loader, self.test_loader = self.initialize_dataloaders()


    def initialize_dataloaders(self):
        """ Generates a torchtext Dataloader from the SST dataset
        """
        print('Initializing the dataloaders')

        tokenize = lambda x: x.split()
        TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
        LABEL = data.Field(sequential=False)
        # make splits for data
        train, val, test = datasets.SST.splits(
            TEXT, LABEL, fine_grained=True, train_subtrees=True,
            filter_pred=lambda ex: ex.label != 'neutral')

        # build the vocabulary
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        TEXT.build_vocab(train, vectors=Vectors('wiki.simple.vec', url=url))
        LABEL.build_vocab(train)

        # make iterator for splits
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=32)

        return train_iter, val_iter, test_iter

    def get_comments(self):
        keys = list(dict.keys())
        x_train, y_train = []
        y_train = []
        x_test = []
        y_test = []

        for i, dict in enumerate([self.train_comments, self.test_comments]):
            keys = list(dict.keys())
            shuffle(keys)
            for item_key in keys:
                x = dict[item_key][0]
                y = dict[item_key][1]
                if i == 0:
                    x_train.append(x)
                    y_train.append(y)
                else:
                    x_test.append(x)
                    y_test.append(y)

        return x_train, y_train, x_test, y_test


    def load_train_comments(self):
        """ Load the different train comments to the object
        """
        print('Load train comments')

        for sentiment in COMMENTS:
            if sentiment == 'pos':
                sentiment_label = 1
            else:
                sentiment_label = 0
            folder_path = '{}/{}'.format(TRAIN_DATA_PATH, sentiment)
            for i, file in enumerate(os.listdir(folder_path)):
                file_path = '{}/{}'.format(folder_path, file)

                if 0 < self.data_limit <= i:
                    break

                with open(file_path, 'r') as rf:
                    comment = rf.read()
                    id, sent_score = file.strip('.txt').split('_')
                    self.train_comments[id] = [comment, sentiment_label, sent_score]

        # preprocessor = PreProcessor(self.train_comments)
        # self.train_comments = preprocessor.DL

    def load_test_comments(self):
        """ Load the different test comments to the object
        """
        print('Load test comments')

        for sentiment in COMMENTS:
            if sentiment == 'pos':
                sentiment_label = 1
            else:
                sentiment_label = 0

            folder_path = '{}/{}'.format(TEST_DATA_PATH, sentiment)
            for i, file in enumerate(os.listdir(folder_path)):
                file_path = '{}/{}'.format(folder_path, file)

                if 0 < self.data_limit <= i:
                    break

                with open(file_path, 'r') as rf:
                    comment = rf.read()
                    id, sent_score = file.strip('.txt').split('_')
                    self.test_comments[id] = [comment, sentiment_label, sent_score]

    # def load_twitter_comments(self):
    #     with open(TWITTER_PATH, 'r') as rf:
    #         csv_reader = csv.reader(rf, delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         if line_count == 0:
    #             print(f'Column names are {", ".join(row)}')
    #             line_count += 1
    #         else:
    #             self.twitter_comments[row[0]] = [row{2}+row{3}, COMMENTS[int(row{1})],row{1}]
    #             print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
    #             line_count += 1
    #     print(f'Processed {line_count} lines.')

    def load_vocabulaire(self):
        """ Load the vocabulaire
        """
        with open(VOCAB_PATH, 'r') as rf:
            vocab = rf.read().splitlines()
        self.vocabulaire = [word for word in vocab]

    def load_word_sentiment(self):
        with open(WORD_SENTIMENT_PATH, 'r') as rf:
            word_sentiment = rf.read().splitlines()
        for index in range(len(self.vocabulaire)):
            self.word_sentiment[self.vocabulaire[index]] = word_sentiment[index]
