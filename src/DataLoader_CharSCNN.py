import os
import torch
from DataSet import DataSet

DATA_PATH = 'data'
TRAIN_DATA_PATH = '{}/train'.format(DATA_PATH)
TEST_DATA_PATH = '{}/test'.format(DATA_PATH)

COMMENTS = ['pos', 'neg']

class DataLoader(object):
    """ Object that loads all formats from the datafolder to Python Object
    """
    def __init__(self, limit=0):
        super(DataLoader, self).__init__()

        # The dicts below follow the format:
        # {Comment_ID: [comment, pos/neg, sentiment_score]}
        self.train_data = DataSet()
        self.test_data = DataSet()
        self._limit = limit
        self.vocabulaire = []

    def load_train_comments(self):
        """ Load the different train comments to the object
        """
        print("Load train comments")

        comments = []
        for sentiment in COMMENTS:
            folder_path = '{}/{}'.format(TRAIN_DATA_PATH, sentiment)
            for file in os.listdir(folder_path):
                file_path = '{}/{}'.format(folder_path, file)
                with open(file_path, 'r') as rf:
                    comment = rf.read().split()
                    if self._limit != 0 and len(comments) == self._limit:
                        break
                    comments.append(comment)                    
                    id, sent_score = file.strip('.txt').split('_')
                    self.train_data.add_data(comment, sentiment, sent_score)
        self.train_data.construct_dataset(comments)                    

    def load_test_comments(self):
        """ Load the different test comments to the object
        """
        print("Load test comments")

        comments = []
        for sentiment in COMMENTS:
            folder_path = '{}/{}'.format(TEST_DATA_PATH, sentiment)
            for file in os.listdir(folder_path):
                file_path = '{}/{}'.format(folder_path, file)
                with open(file_path, 'r') as rf:
                    if self._limit != 0 and len(comments) == self._limit:
                        break
                    comment = rf.read().split()
                    comments.append(comment)
                    id, sent_score = file.strip('.txt').split('_')
                    self.test_data.add_data(comment, sentiment, sent_score)
        self.test_data.construct_dataset(comments)


    def load_vocabulaire(self):
        """ Load the vocabulaire
        """
        file_path = '{}/imdb.vocab'.format(DATA_PATH)
        with open(file_path, 'r') as rf:
            vocab = rf.read()
        self.vocabulaire = [word for word in vocab]


if __name__ == '__main__':
    dl = DataLoader()
    dl.load_train_comments()
    print(torch.stack(dl_train_data.x_chr))