import os

DATA_PATH = '../data'
TRAIN_DATA_PATH = '{}/train'.format(DATA_PATH)
TEST_DATA_PATH = '{}/test'.format(DATA_PATH)

COMMENTS = ['pos', 'neg']

class DataLoader(object):
    """ Object that loads all formats from the datafolder to Python Object
    """
    def __init__(self):
        super(DataLoader, self).__init__()

        # The dicts below follow the format:
        # {Comment_ID: [comment, pos/neg, sentiment_score]}

        self.train_comments = {}
        self.test_comments = {}
        self.vocabulaire = []

    def load_train_comments(self):
        """ Load the different train comments to the object
        """
        for sentiment in COMMENTS:
            folder_path = '{}/{}'.format(TRAIN_DATA_PATH, sentiment)
            for file in os.listdir(folder_path):
                file_path = '{}/{}'.format(folder_path, file)
                with open(file_path, 'r') as rf:
                    comment = rf.read()
                    id, sent_score = file.strip('.txt').split('_')
                    self.train_comments[id] = [comment, sentiment, sent_score]

    def load_test_comments(self):
        """ Load the different test comments to the object
        """
        for sentiment in COMMENTS:
            folder_path = '{}/{}'.format(TEST_DATA_PATH, sentiment)
            for file in os.listdir(folder_path):
                file_path = '{}/{}'.format(folder_path, file)
                with open(file_path, 'r') as rf:
                    comment = rf.read()
                    id, sent_score = file.strip('.txt').split('_')
                    self.test_comments[id] = [comment, sentiment, sent_score]


    def load_vocabulaire(self):
        """ Load the vocabulaire
        """
        file_path = '{}/imdb.vocab'.format(DATA_PATH)
        with open(file_path, 'r') as rf:
            vocab = rf.read()
        self.vocabulaire = [word for word in vocab]
