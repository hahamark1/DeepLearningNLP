import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer
from vocab import Vocab, UnkVocab
import numpy as np


'''
STEP 1: LOADING DATASET
'''
x = ['ik ben blij', 'ik ben niet blij', 'ik ben niet niet blij',
     'ik ben happy', 'ik ben niet happy', 'ik ben niet niet happy']
y = [1, 0, 1, 1, 0, 1]

vectorizer = CountVectorizer()
vectorizer.fit_transform(x).todense()
voc_list = list(vectorizer.vocabulary_)
vocabulary = Vocab()
vocabulary.word2index(voc_list, train=True)

x_indices = []
for sentence in x:
    words = sentence.split(' ')
    indices = [vocabulary.word2index(word) for word in words]
    x_indices.append(torch.tensor(indices))

x_indices = x_indices

'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size = 1
n_iters = 3000
num_epochs = n_iters / (len(x) / batch_size)
num_epochs = int(num_epochs)


'''
STEP 3: CREATE MODEL CLASS
'''

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # x = self.embedding(x)
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, hn = self.rnn(x, h0)
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 2
        return out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 1
hidden_dim = 300
layer_dim = 5  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 2

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
print(model)

#######################
#  USE GPU FOR MODEL  #
#######################

if torch.cuda.is_available():
    model.cuda()
    
'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

'''
STEP 7: TRAIN THE MODEL
'''

n_epochs = 25
for e in range(n_epochs):
    total_loss = 0
    for i, item in enumerate(x_indices):
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        batch = np.array(item)
        batch = batch.reshape((1, len(item), 1))
        batch = torch.tensor(batch, dtype=torch.float)
        outputs = model(batch)

        # Calculate Loss: softmax --> cross entropy loss
        label = np.array([y[i]])
        label = torch.tensor(label, dtype=torch.int64)
        loss = criterion(outputs, label)
        total_loss += loss

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

    print('Epoch {}\t Loss {}'.format(e, total_loss))

input_indices = []
sentence = 'ik ben niet blij'
words = sentence.split(' ')
indices = [vocabulary.word2index(word) for word in words]
input = indices
input_batch = np.array([input])
input_batch = input_batch.reshape((1, len(input), 1))
input_batch = torch.tensor(input_batch, dtype=torch.float)
print(model(input_batch))


