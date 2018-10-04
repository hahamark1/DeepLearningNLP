import os
import torch
from src.DataSet import DataSet

COMMENTS = ['pos', 'neg']

class DataLoader(object):
    """ Object that loads all formats from the datafolder to Python Object
    """
    def __init__(self, limit=0, use_padding=True, data_path='data'):
        super(DataLoader, self).__init__()

        # The dicts below follow the format:
        # {Comment_ID: [comment, pos/neg, sentiment_score]}
        self.train_data = DataSet()
        self.test_data = DataSet()
        self.val_data = DataSet()
        self.twitter_train_data = DataSet()
        self._limit = limit
        self.vocabulaire = []
        self._padding = use_padding
        self.data_path = data_path

        self.TRAIN_DATA_PATH = '{}/train'.format(self.data_path)
        self.TEST_DATA_PATH = '{}/test'.format(self.data_path)
        self.TWITTER_PATH = '{}/train.txt'.format(self.data_path)

    def load_twitter_comments(self):
        """ Load the data from the Twitter Sentiment Dataset
        """
        comments_train = []
        comments_test = []
        comments_val = []
        line_count = 0
        with open(self.TWITTER_PATH, 'r') as rf:
            file = rf.read().splitlines()
            test_size = 0.6
            val_size = 0.8
            for i, row in enumerate(file):
                row = row.split('\t')
                sentiment, comment = row[0], row[1]
                if sentiment == '0':
                    label = 'neg'
                else:
                    label = 'pos'

                if line_count == 0:
                    line_count += 1
                else:
                    if i < test_size * len(file):
                        self.train_data.add_data(comment, label, sentiment)
                        comments_train.append(comment)
                    elif i < val_size * len(file):
                        self.val_data.add_data(comment, label, sentiment)
                        comments_val.append(comment)
                    else:
                        self.test_data.add_data(comment, label, sentiment)
                        comments_test.append(comment)
                    line_count += 1
        print('Now constructing')
        self.train_data.construct_dataset(comments_train)
        self.test_data.construct_dataset(comments_test, self.train_data)
        self.val_data.construct_dataset(comments_val, self.train_data)
        print('Processed {} lines.'.format(line_count))

    def load_train_comments(self):
        """ Load the different train comments to the object
        """
        print("Load train comments")

        comments = []
        for i, sentiment in enumerate(COMMENTS):
            folder_path = '{}/{}'.format(self.TRAIN_DATA_PATH, sentiment)
            for file in os.listdir(folder_path):
                file_path = '{}/{}'.format(folder_path, file)
                with open(file_path, 'r') as rf:
                    comment = rf.read()
                    if self._limit != 0 and len(comments) == (i + 1) * self._limit:
                        break
                    comments.append(comment)
                    id, sent_score = file.strip('.txt').split('_')
                    self.train_data.add_data(comment, sentiment, sent_score)
        print('Now constructing')
        self.train_data.construct_dataset(comments)

    def load_test_comments(self, train_set):
        """ Load the different test comments to the object
        """
        print("Load test comments")

        comments = []
        i = 1
        for i, sentiment in enumerate(COMMENTS):
            folder_path = '{}/{}'.format(self.TEST_DATA_PATH, sentiment)
            for file in os.listdir(folder_path):
                file_path = '{}/{}'.format(folder_path, file)
                with open(file_path, 'r') as rf:
                    if self._limit != 0 and len(comments) == (i + 1) * self._limit:
                        break
                    comment = rf.read()
                    comments.append(comment)
                    id, sent_score = file.strip('.txt').split('_')
                    self.test_data.add_data(comment, sentiment, sent_score)
        self.test_data.construct_dataset(comments, train_set)


    def load_vocabulaire(self):
        """ Load the vocabulaire
        """
        file_path = '{}/imdb.vocab'.format(self.DATA_PATH)
        with open(file_path, 'r') as rf:
            vocab = rf.read()
        self.vocabulaire = [word for word in vocab]


if __name__ == '__main__':
    print('Please run the bash scripts or train_*.py files!')
