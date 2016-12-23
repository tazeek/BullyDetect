import pickle
import os

import numpy as np 

from gensim.models import Word2Vec as w2v 

os.system('cls')

def getAverageCluster(word_list, model):

	for word in word_list:
		print(word)

	return

# Load the Cluster dictionary
print("LOADING CLUSTER DICTIONARY \n\n")
cluster_num = 500
FILE = "Word Dictionaries/dict_" + str(cluster_num) + "C.pk"
array_dict_cluster = pickle.load(open(FILE, "rb"))

for cluster in array_dict_cluster:
	words = cluster['word_list']
	
	getAverageCluster(words, "")
	break
exit()
# Load Word2Vec model 
print("LOADING WORD2VEC MODEL \n\n")
FILE = "W2V Models/w2v_reddit_unigram_300d.bin"
model = w2v.load_word2vec_format(FILE, binary=True)