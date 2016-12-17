import time 
import pickle
import os
import logging
import datetime

from gensim.models import Word2Vec as w2v 
from sklearn.cluster import KMeans

def doClustering():

	os.system('cls')

	# Load Word2Vec model
	print("LOADING WORD2VEC MODEL \n\n")
	model = w2v.load_word2vec_format('W2V Models/w2v_reddit_unigram_300d.bin', binary=True)

	# Specify the number of words and clusters (250,500,1000,2000)
	# NOTE: When utilizing full Word2Vec power, ignore WORDS
	#WORDS = 600000
	CLUSTERS = 500

	# Get the word vectors and the word
	print("GETTING WORD VECTORS AND WORDS \n\n")
	word_vectors = model.syn0
	words = model.index2word

	# Delete the model to clear up memory
	del model

	# Initialize K-Means
	k_means = KMeans( n_clusters = CLUSTERS, n_jobs=6, precompute_distances=True)

	# Give starting time of initialization
	start = datetime.datetime.now()
	print("STARTING AT: %i/%i/%i %i:%i \n" % (start.month, start.day, start.year, start.hour, start.minute))

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
	FILE = "K-Means Models/full_500C.pk"
	pickle.dump(word_centroid_map, open(FILE, "wb"))

if __name__ == '__main__':

	doClustering()