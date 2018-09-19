import gensim
from DataLoader import DataLoader
from gensim.models import Word2Vec

class Embedding(object):
	"""
	Embedding object that can be trained given a DataLoader object.	
	"""

	def __init__(self, dim=256, emb_type='cbow'):
		self.dim = dim
		self.type = emb_type
		self.model = None

	def train(self, DataLoader):
		"""
		Trains the embedding
		"""
		sentences = (sent for sent in comment.split('.') for comment in \
						DataLoader.train_comments.values())
		self.model = Word2Vec(sentences, size=self.dim)
