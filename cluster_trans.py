import pickle
import os

import numpy as np 

from gensim.models import Word2Vec as w2v 

os.system('cls')

def getAverageCluster(word_list, model):

	# Get number of words
	total_words = len(word_list)

	# Pre-initialize an empty numpy array (for speed)
	# 300 is used, as it is the number of vectors in Word2Vec
	avgWordsFeature = np.zeros((300,),dtype="float32")

	# Loop word by word
	for word in word_list:
		
		# Add the word's feature vectors 
		avgWordsFeature = np.add(avgWordsFeature, model[word])

	
	# Divide to get the mean
	avgWordsFeature = np.divide(avgWordsFeature,total_words)

	return avgWordsFeature

# Load the Cluster dictionary
print("LOADING CLUSTER DICTIONARY \n\n")
cluster_num = 500
FILE = "Word Dictionaries/dict_" + str(cluster_num) + "C.pk"
array_dict_cluster = pickle.load(open(FILE, "rb"))

# Load Word2Vec model 
print("LOADING WORD2VEC MODEL \n\n")
FILE = "W2V Models/w2v_reddit_unigram_300d.bin"
model = w2v.load_word2vec_format(FILE, binary=True)

# Loop cluster by cluster
print("STARTING TRANSFORMATIONS \n\n")
for cluster in array_dict_cluster:

	# Get the word list 
	words = cluster['word_list']
	
	# Call the function
	getAverageCluster(words, model)
	break

exit()