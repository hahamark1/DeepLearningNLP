import os
import torch
from src.DataSet import DataSet

DATA_PATH = 'data'
TRAIN_DATA_PATH = '{}/train'.format(DATA_PATH)
TEST_DATA_PATH = '{}/test'.format(DATA_PATH)

COMMENTS = ['pos', 'neg']

class DataLoader(object):
    """ Object that loads all formats from the datafolder to Python Object
    """
    def __init__(self, limit=0, use_padding=True):
        super(DataLoader, self).__init__()

        # The dicts below follow the format:
        # {Comment_ID: [comment, pos/neg, sentiment_score]}
        self.train_data = DataSet()
        self.test_data = DataSet()
        self.twitter_train_data = DataSet()
        self._limit = limit
        self.vocabulaire = []
        self._padding = use_padding

    def load_twitter_comments():
        comments = []
        with open(TWITTER_PATH, 'r') as rf:
            csv_reader = csv.reader(rf, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if self._limit != 0 and len(comments) == self._limit:
                        break
                    comments.append(comment)
                    id, sent_score = row[1], row[1]
                    self.twitter_train_data.add_data(comment, sentiment, sent_score)
                    line_count += 1
            print('Now constructing')
            self.twitter_train_data.construct_dataset(comments)
            print('Processed {} lines.'.format(line_count))

    def load_train_comments(self):
        """ Load the different train comments to the object
        """
        print("Load train comments")

        comments = []
        for i, sentiment in enumerate(COMMENTS):
            folder_path = '{}/{}'.format(TRAIN_DATA_PATH, sentiment)
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
            folder_path = '{}/{}'.format(TEST_DATA_PATH, sentiment)
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
        file_path = '{}/imdb.vocab'.format(DATA_PATH)
        with open(file_path, 'r') as rf:
            vocab = rf.read()
        self.vocabulaire = [word for word in vocab]


if __name__ == '__main__':
    dl = DataLoader()
    dl.load_train_comments()
    print(torch.stack(dl_train_data.x_chr))
