import os

from src.Preprocessor import PreProcessor
from random import shuffle

DATA_PATH = './data'
TRAIN_DATA_PATH = '{}/train'.format(DATA_PATH)
TEST_DATA_PATH = '{}/test'.format(DATA_PATH)

COMMENTS = ['pos', 'neg']

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

        self.load_train_comments()
        self.load_test_comments()

    def get_comments(self):
        x_train = []
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

        preprocessor = PreProcessor(self.train_comments)
        self.train_comments = preprocessor.DL

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

        preprocessor = PreProcessor(self.test_comments)
        self.test_comments = preprocessor.DL

