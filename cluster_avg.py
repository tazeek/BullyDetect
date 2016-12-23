import pandas as pd 
import numpy as np 

import pickle
import os

from gensim.models import Word2Vec as w2v 

from evaluation import evaluate

def characterVec(word, cluster_dict, cluster_map, num_features):
	featureVec = np.zeros((num_features,), dtype="float32")

	n_char = 0.

	for character in word:

		n_char += 1.

		cluster_num = cluster_map[character]
		avg_vector = array_dict_cluster[cluster_num]['average_vector']

		featureVec = np.add(featureVec,avg_vector)

	featureVec = np.divide(featureVec, n_char)

	return featureVec 

def getAverageComment(sentence, cluster_dict, cluster_map, num_features):

	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,),dtype="float32")

	# Count number of words
	nwords = 0.

	# Loop over word by word
	# If in vocabulary, add its feature vector to the total
	for word in sentence.split():

		if word in cluster_map: #and word not in stop_words:

			nwords += 1.

			cluster_num = cluster_map[word]
			avg_vector = cluster_dict[cluster_num]['average_vector']

			featureVec = np.add(featureVec,avg_vector)

	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)

	if nwords == 0:
		featureVec = characterVec(sentence, cluster_dict, cluster_map, num_features)

	return featureVec

def transformData(comments, cluster_dict, cluster_map, num_features):

	# Initialize empty counter
	counter = 0

	# Preallocate a 2D numpy array for speed
	reviewFeatureVecs = np.zeros((len(comments),num_features),dtype="float32")

	for comment in comments:

		# Call function that gets the average vectors
		reviewFeatureVecs[counter] = getAverageComment(comment, cluster_dict, cluster_map, num_features)

		# Increment counter
		counter += 1


	return reviewFeatureVecs

os.system('cls')
# Load the dataset here
print("LOADING DATASET \n\n")
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']

# Specify cluster file to load
cluster_file = 500

# Specify path and load files using pickle
print("LOADING K-MEANS MODELS \n\n")
FILE_DICT = "C:/Users/MyPC/Desktop/Vegito/Word Dictionaries/trans_dict_" + str(cluster_file) + "C.pk"
FILE_CLUS = "C:/Users/MyPC/Desktop/Vegito/K-Means Models/full_" + str(cluster_file) + "C.pk"

array_dict_cluster = pickle.load(open(FILE_DICT, "rb"))
word_centroid_map =  pickle.load(open(FILE_CLUS,"rb"))

# Transform data
print("TRANSFORMING DATA \n\n")
X = transformData(X, array_dict_cluster, word_centroid_map, 300)

# Get the Python's file name. Remove the .py extension
file_name = os.path.basename(__file__)
file_name = file_name.replace(".py","")

# Evaluate models 
evaluate(X,y, file_name)