import gensim
from DataLoader import DataLoader
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize

class Embedding(object):
	"""
	Embedding object that can be trained given a DataLoader object.	
	"""

	def __init__(self, dim=256, emb_type='cbow'):
		"""
		Args:
			dim:		number embedding dimensions
			emb_type:	word2vec type 'cbow' or 'skip_gram'

		"""
		self.dim = dim
		self.type = emb_type
		self.model = None

	def train(self, DataLoader):
		"""
		Trains the embedding

		Args:
			DataLoader:	DataLoader object containing loaded train comments.

		"""
		print('Indexing training sentences...') 
		# Extract sentences from all training comments in the DataLoader
		sentences = [word_tokenize(sent) for comment in iter(DataLoader.train_comments.values()) \
						for sent in sent_tokenize(comment[0].replace('<br />',''))]
		print('Training word2vec model...')
		self.model = Word2Vec(sentences, size=self.dim, sg=1 if self.type=='skip_gram' else 0)
		print('Finished training model.')


def main():
	train = False
	
	if train:
		# Demonstration training new model
		dl = DataLoader()
		dl.load_train_comments()
		emb = Embedding()
		emb.train(dl)
		emb.model.save('./cbow_model')
	else:

		# Demonstration using pretrained model
		model = Word2Vec.load("./cbow_model")
		vector = model.wv['computer']  # numpy vector of a word
		print('\n\nCalling model.wv[\"computer\"] yields:\n', vector)
		sim_to_book = model.wv.most_similar(positive='book', topn=5)
		print('\n\nCalling model.wv.most_similar(positive=\"book\", topn=5) yields:\n', sim_to_book)

if __name__ == '__main__':
	main()
