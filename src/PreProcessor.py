import re
from nltk.stem import PorterStemmer

STOPWORD_PATH = 'lib/stopwords.txt'
CHARACTERS_TO_REMOVE = '[^\w^\s]'


class PreProcessor(object):
    """docstring for PreProcessor."""
    def __init__(self, DL, stemming=True, stopwords_removal=True, charecter_removal=True):
        super(PreProcessor, self).__init__()
        self.DL = DL
        self.load_stopwords()

        if charecter_removal:
            self.remove_special_characters()
        if stopwords_removal:
            self.remove_stop_words()
        if stemming:
            self.perform_stemming()

    def remove_special_characters(self):
        """ Remove special characters from the comments, given above
        """
        for id in self.DL.train_comments.keys():
            self.DL.train_comments[id][0] = re.sub(CHARACTERS_TO_REMOVE, '', self.DL.train_comments[id][0])
        for id in self.DL.test_comments.keys():
            self.DL.test_comments[id][0] = re.sub(CHARACTERS_TO_REMOVE, '', self.DL.test_comments[id][0])

    def remove_stop_words(self):
        """ Remove the stopwords from the comments given stopwords defined in the lib folder
        """
        for id in self.DL.train_comments.keys():
            self.DL.train_comments[id][0] = ' '.join([word for word in self.DL.train_comments[id][0].split(' ') if word not in self.stopwords])
        for id in self.DL.test_comments.keys():
            self.DL.test_comments[id][0] = ' '.join([word for word in self.DL.test_comments[id][0].split(' ') if word not in self.stopwords])

    def perform_stemming(self):
        ps = PorterStemmer()
        for id in self.DL.train_comments.keys():
            self.DL.train_comments[id][0] = ' '.join([ps.stem(word) for word in self.DL.train_comments[id][0].split(' ')])
        for id in self.DL.test_comments.keys():
            self.DL.test_comments[id][0] = ' '.join([ps.stem(word) for word in self.DL.test_comments[id][0].split(' ')])

    def load_stopwords(self):
        """ Load the stopwords from the file.
        """
        with open(STOPWORD_PATH, 'r') as rf:
            self.stopwords = rf.read().splitlines()
