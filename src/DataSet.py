import torch
import numpy as np
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize


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
        self.word_idx = 1
        self.char_idx = 1
        self.word_w_size = 5
        self.chr_w_size = 3
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.num_examples = 0

    def add_data(self, comment, sentiment, sent_score):
        # print(sentiment, sent_score)
        # if
        self.y.append(torch.tensor([1]) if sentiment == 'pos' else torch.tensor([0]))
        self.y_score.append(torch.tensor([int(sent_score)]))

        comment = [word_tokenize(sent) for sent in sent_tokenize(comment.replace('<br />',''))]
        comment = [word for sentence in comment for word in sentence]
        self.max_sent_len = max(self.max_sent_len, len(comment))

        for token in comment:
            if token not in self.word2idx:
                self.word2idx[token] = self.word_idx
                self.word_idx += 1
                self.max_word_len = max(self.max_word_len, len(token))
            for i in range(len(token)):
                if token[i] not in self.char2idx:
                    self.char2idx[token[i]] = self.char_idx
                    self.char_idx += 1

    def construct_dataset(self, comments):
        self.comments = comments
        self.seq_size_words = self.max_sent_len + self.word_w_size - 1
        self.seq_size_chars = self.max_word_len + self.chr_w_size - 1

        		# sentences = [word_tokenize(sent) for comment in iter(DataLoader.train_comments.values()) \
        		# 				for sent in sent_tokenize(comment[0].replace('<br />',''))]
        # for tokens in comments:
        #     print('Now at comment {}'.format(o))
        #     print(tokens)
        #     # tokens = [word_tokenize(sent) for sent_tokenize(tokens.replace('<br />',''))
        #     # o+=1
        #     word_mat = torch.zeros((self.seq_size_words,))
        #     char_mat = torch.zeros((self.seq_size_words, self.seq_size_chars)).type('torch.LongTensor')
        #
        #     for i in range(len(tokens)):
        #
        #         word_mat[int(self.word_w_size / 2) + i] = self.word2idx[tokens[i]]
        #         for j in range(len(tokens[i])):
        #             char_mat[int((self.word_w_size / 2)) + i][int(self.chr_w_size / 2) + j] = self.char2idx[
        #                 tokens[i][j]]
        #     self.x_chr.append(char_mat)
        #     self.x_wrd.append(word_mat)

        self.y = torch.stack(self.y)
        self.y_score = torch.stack(self.y_score)

        # self.max_word_len += self.chr_w_size - 1
        # self.max_sent_len += self.word_w_size - 1
        self.num_examples = len(comments)

    def next_batch(self, batch_size=4):
        """
        Return the next `batch_size` examples from this data set.
        Args:
          batch_size: Batch size.
        """

        self.x_chr = []
        self.x_wrd = []

        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            self.epochs_completed += 1

            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.x_wrd = self.x_wrd[perm]
            self.x_chr = self.x_chr[perm]
            self.y = self.y[perm]
            self.y_score = self.y_score[perm]

            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples

        end = self.index_in_epoch

        print('In the batchloader')
        print(self.max_sent_len)
        print(self.seq_size_words, self.seq_size_chars)
        for comment in self.comments[start:end]:
            # print(comment)
            comment = [word_tokenize(sent) for sent in sent_tokenize(comment.replace('<br />',''))]
            comment = [word for sentence in comment for word in sentence]
            word_mat = torch.zeros((self.seq_size_words,))
            char_mat = torch.zeros((self.seq_size_words, self.seq_size_chars)).type('torch.LongTensor')

            for i in range(len(comment)):
                word_mat[int(self.word_w_size / 2) + i] = self.word2idx[comment[i]]
                for j in range(len(comment[i])):
                    char_mat[int((self.word_w_size / 2)) + i][int(self.chr_w_size / 2) + j] = self.char2idx[
                        comment[i][j]]
            self.x_chr.append(char_mat)
            self.x_wrd.append(word_mat)

        self.x_wrd = torch.stack(self.x_wrd).type('torch.LongTensor')
        self.x_chr = torch.stack(self.x_chr).type('torch.LongTensor')

        return self.x_wrd, self.x_chr, self.y[start:end], self.y_score[start:end]
