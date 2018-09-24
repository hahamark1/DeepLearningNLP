import torch
import numpy as np

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
        self.y.append(sentiment)
        self.y_score.append(sent_score)
        self.max_sent_len = max(self.max_sent_len, len(comment))
        for token in comment:
            if token not in self.word2idx:
                self.word2idx[token] = self.word_idx
                self.word_idx += 1
                self.max_word_len = max(self.max_word_len,len(token))
            for i in range(len(token)):
                if token[i] not in self.char2idx:
                    self.char2idx[token[i]] = self.char_idx
                    self.char_idx += 1

    def construct_dataset(self, comments):
        for comment in comments:
            tokens = comment
            word_mat = torch.zeros((self.max_sent_len+self.word_w_size-1, ))
            char_mat = torch.zeros((self.max_sent_len+self.word_w_size-1, self.max_word_len+self.chr_w_size-1))

            for i in range(len(tokens)):
                word_mat[int(self.word_w_size/2)+i] = self.word2idx[tokens[i]]
                for j in range(len(tokens[i])):
                    char_mat[int((self.word_w_size/2))+i][int(self.chr_w_size/2)+j] = self.char2idx[tokens[i][j]]
            self.x_chr.append(char_mat)
            self.x_wrd.append(word_mat)
        self.max_word_len += self.chr_w_size-1
        self.max_sent_len += self.word_w_size-1
        self.num_examples = word_mat.shape[0]


      def next_batch(self, batch_size):
        """
        Return the next `batch_size` examples from this data set.
        Args:
          batch_size: Batch size.
        """
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
        return self.wrd[start:end], self.chr[start:end], self.y[start:end], self.y_score[start:end]


