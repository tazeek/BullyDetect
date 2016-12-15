import time 
import pickle
import os
import logging

from gensim.models import Word2Vec as w2v 
from sklearn.cluster import KMeans

os.system('cls')

# Load Word2Vec model
model = w2v.load_word2vec_format('W2V Models/w2v_reddit_unigram_300d.bin', binary=True)

# Specify the number of words and clusters
# NOTE: When utilizing full Word2Vec power, ignore WORDS
WORDS = 50000
CLUSTERS = 500

# Get the word vectors and the word
word_vectors = model.syn0[:WORDS]
words = model.index2word[:WORDS]

# Initialize K-Means
k_means = KMeans( n_clusters = CLUSTERS )

# Fit the model, get the centroid number and calculate time
start = time.time()
idx = k_means.fit_predict(word_vectors)
end = time.time()

print("TIME TAKEN: ", end-start)

# Create a Word / Index dictionary
# Each vocabulary word is matched to a cluster center
# Motivation: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors

word_centroid_map = dict(zip(words,idx))

# Save the dictionary
FILE = "K-Means Models/dict_500.pk"
pickle.dump(word_centroid_map, open(FILE, "wb"))