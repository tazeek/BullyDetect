import time 
import pickle
import os
import logging

from gensim.models import Word2Vec as w2v 
from sklearn.cluster import KMeans

os.system('cls')

# Load Word2Vec model
print("LOADING WORD2VEC MODEL \n\n")
model = w2v.load_word2vec_format('W2V Models/w2v_reddit_unigram_300d.bin', binary=True)

# Specify the number of words and clusters (250,500,1000,2000)
# NOTE: When utilizing full Word2Vec power, ignore WORDS
#WORDS = 100000
CLUSTERS = 250

# Get the word vectors and the word
print("GETTING WORD VECTORS AND WORDS \n\n")
word_vectors = model.syn0
words = model.index2word

# Initialize K-Means
k_means = KMeans( n_clusters = CLUSTERS )

# Fit the model, get the centroid number and calculate time
print("TRAINING K-MEANS WITH %i CLUSTERS \n\n" % (CLUSTERS))
start = time.time()
idx = k_means.fit_predict(word_vectors)
end = time.time()

print("TIME TAKEN: ", end-start)

# Create a Word / Index dictionary
# Each vocabulary word is matched to a cluster center
# Motivation: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors

word_centroid_map = dict(zip(words,idx))

# Save the dictionary
print("\n\nSAVING MODEL")
FILE = "K-Means Models/dict_250_full.pk"
pickle.dump(word_centroid_map, open(FILE, "wb"))