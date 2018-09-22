import gensim
from DataLoader import DataLoader
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize

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
		# word_tokenize(sent) for sent in
		print('Indexing training sentences...') 
		sentences = [word_tokenize(sent) for comment in iter(DataLoader.train_comments.values()) \
						for sent in sent_tokenize(comment[0].replace('<br />',''))]
		sentences = read_sentences(DataLoader)
		print('Training word2vec model...')
		self.model = Word2Vec(sentences, size=self.dim, sg=1 if self.type=='skip_gram' else 0)
		print('Finished training model.')


def main():
	dl = DataLoader()
	dl.load_train_comments()
	emb = Embedding()
	emb.train(dl)

if __name__ == '__main__':
	main()
