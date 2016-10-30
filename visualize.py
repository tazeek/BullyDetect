from sklearn.manifold import TSNE
from gensim.models import Word2Vec

import codecs
import numpy as np 
import matplotlib.pyplot as plt 
import logging 

#For Logging Parameters
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Loading Word Embeddings takes place here
def load_embeddings(file):

	#Open the file
	model = Word2Vec.load_word2vec_format(file, binary=True)


	vectors = []
	words = []

	#Add vectors and words
	for word in model.vocab:
		vectors.append(model[word]) 
		words.append(word)

	return vectors, words


# Visualization takes place here
embeddings_file = "reddit.bin"

wv, vocabulary = load_embeddings(embeddings_file) # Load the model and the words

# Initialize TSNE Model
tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)

# Fit with 1000 words 
Y = tsne.fit_transform(wv[100000:115000])

# Use Scatterplot
plt.scatter(Y[:, 0], Y[:, 1])

# Initialize Points
for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
	plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.axis("off")
plt.show()