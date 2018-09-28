import torch
import numpy as np
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize

UNK_TOKEN = 'UNK'

class DataSet(object):
    def __init__(self):
        self.x_chr = []
        self.x_wrd = []
        self.y = []
        self.y_score = []
        self.max_sent_len = 0
        self.max_word_len = 0
        self.word2idx = {}
        self.char2idx = {}
        self.word_idx = 1.
        self.char_idx = 1.
        self.word_w_size = 5
        self.chr_w_size = 3
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.num_examples = 0

    def add_data(self, comment, sentiment, sent_score):
        """ For every comment we receive we see if we need to add words and
        chars to the dicts and if we need to change our max sizes
        """
        self.y.append(torch.tensor([1]) if sentiment == 'pos' else torch.tensor([0]))
        self.y_score.append(torch.tensor([int(sent_score)]))

        comment = [word_tokenize(sent) for sent in sent_tokenize(comment.replace('<br />',''))]
        comment = [word for sentence in comment for word in sentence]
        self.max_sent_len = max(self.max_sent_len, len(comment))

        for token in comment:
            if token not in self.word2idx:
                self.word2idx[token] = self.word_idx
                self.word_idx += 1.
                self.max_word_len = max(self.max_word_len, len(token))
            for i in range(len(token)):
                if token[i] not in self.char2idx:
                    self.char2idx[token[i]] = self.char_idx
                    self.char_idx += 1.
        token = UNK_TOKEN

        self.word2idx[token] = self.word_idx
        self.word_idx += 1.
        self.max_word_len = max(self.max_word_len, len(token))
        self.char2idx[token] = self.char_idx
        self.char_idx += 1.

    def construct_dataset(self, comments, train_set=False):
        """ Set some of the dataset specific values at the end of the initialization
        """
        self.comments = comments
        self.seq_size_words = self.max_sent_len + self.word_w_size - 1
        self.seq_size_chars = self.max_word_len + self.chr_w_size - 1

        self.num_examples = len(comments)
        self.vocab_size_words = len(self.word2idx.keys())
        self.voczb_size_char = len(self.char2idx.keys())
        if train_set:
            self.word2idx = train_set.word2idx
            self.char2idx = train_set.char2idx
            self.vocab_size_words = len(self.word2idx.keys())
            self.vocab_size_char = len(self.char2idx.keys())
            self.max_sent_len = train_set.max_sent_len
            self.max_word_len = train_set.max_word_len
            self.seq_size_words = self.max_sent_len + self.word_w_size - 1
            self.seq_size_chars = self.max_word_len + self.chr_w_size - 1




    def next_batch(self, batch_size=4, padding=True):
        """
        Return the next `batch_size` examples from this data set.
        Args:
          batch_size: Batch size.
          padding: Boolean, if True, pad to the max sizes with zeros, else ??
        """

        self.x_chr = []
        self.x_wrd = []

        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples or self.index_in_epoch == 0:
            self.epochs_completed += 1

            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            print(perm)
            self.comments = [self.comments[i] for i in perm]
            # self.x_chr[:] = [self.x_chr[i] for i in perm]
            self.y = [self.y[i] for i in perm]
            self.y_score = [self.y_score[i] for i in perm]

            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples

        end = self.index_in_epoch
        for comment in self.comments[start:end]:
            if padding:
                comment = [word_tokenize(sent) for sent in sent_tokenize(comment.replace('<br />',''))]
                comment = [word for sentence in comment for word in sentence]
                word_mat = torch.zeros((self.seq_size_words,))
                char_mat = torch.zeros((self.seq_size_words, self.seq_size_chars)).type('torch.LongTensor')

                for i in range(len(comment)):
                    if comment[i] not in self.word2idx:
                        word_mat[int(self.word_w_size / 2) + i] = float(self.word2idx[UNK_TOKEN])
                    else:
                        word_mat[int(self.word_w_size / 2) + i] = float(self.word2idx[comment[i]])
                    for j in range(len(comment[i])):
                        if comment[i][j] not in self.char2idx:
                            char_mat[int((self.word_w_size / 2)) + i][int(self.chr_w_size / 2) + j] = float(self.char2idx[UNK_TOKEN])
                        else:
                            char_mat[int((self.word_w_size / 2)) + i][int(self.chr_w_size / 2) + j] = float(self.char2idx[comment[i][j]])
                self.x_chr.append(char_mat)
                self.x_wrd.append(word_mat)
            else:
                ## TODO: We need to see how we implement batches if we do not have a set size
                raise NotImplementedError

        self.y_out = torch.stack(self.y)
        self.y_score_out = torch.stack(self.y_score)

        self.x_wrd = torch.stack(self.x_wrd).type('torch.FloatTensor')
        self.x_chr = torch.stack(self.x_chr).type('torch.FloatTensor')

        return self.x_wrd, self.x_chr, self.y_out[start:end], self.y_score_out[start:end]
