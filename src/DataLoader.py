import os
import csv

DATA_PATH = './data'
TRAIN_DATA_PATH = '{}/train'.format(DATA_PATH)
TEST_DATA_PATH = '{}/test'.format(DATA_PATH)
VOCAB_PATH = '{}/imdb.vocab'.format(DATA_PATH)
WORD_SENTIMENT_PATH = '{}/imdbEr.txt'.format(DATA_PATH)
TWITTER_PATH = '{}/twitter_sentiment.cxv'.format(DATA_PATH)

COMMENTS = ['neg', 'pos']

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
        self.word_sentiment = {}
        self.twitter_comments = {}

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

    def load_twitter_comments(self):
        with open(TWITTER_PATH, 'r') as rf:
            csv_reader = csv.reader(rf, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                self.twitter_comments[row[0]] = [row{2}+row{3}, COMMENTS[int(row{1})],row{1}]
                print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                line_count += 1
        print(f'Processed {line_count} lines.')

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
