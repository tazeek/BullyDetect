import pandas as pd 
import numpy as np 
import os
import pickle

from gensim.models import Word2Vec as w2v
from collections import defaultdict

from evaluation import evaluate

# Use each word as a feature
def makeFeatureVec(comment, model, tfidf_model, num_features):

	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,),dtype="float32")

	# Loop word-by-word, as well as index
	for i,word in enumerate(comment.split()):

		# INCOMPLETE SENTENCE DETECTED
		if i == len(featureVec):
			break

		# If word is in model, return average of the word's feature vectors
		# Else, return -1 which indicates no word found
		if word in model:
			word_feature = np.mean(model[word])
			word_feature = np.multiply(word_feature, tfidf_model[word])
		else:
			word_feature = -1.0

		# Overwrite the sentence of the numpy array
		featureVec[i] = word_feature

	return featureVec

# One of the kaggle tests
def commentFeatureVecs(comments, model, tfidf_model, num_features):

	# Initialize empty counter
	counter = 0

	# Preallocate a 2D numpy array for speed
	reviewFeatureVecs = np.zeros((len(comments),num_features),dtype="float32")

	for comment in comments:

		# Call function that gets the average vectors
		reviewFeatureVecs[counter] = makeFeatureVec(comment, model, tfidf_model, num_features)

		# Increment counter
		counter += 1


	return reviewFeatureVecs

# Get TFIDF Model and Transform it
# Motivation: http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
def getTFIDIF(FILE):

	tfidf_model = None

	# Load it
	tfidf_model = pickle.load(open(FILE,"rb"))

	# Get max idf value
	max_idf = max(tfidf_model.idf_)

	# Transform the model into a dictionary 
	tfidf_model = defaultdict(lambda: max_idf, [(w, tfidf_model.idf_[i]) 
		for w, i in tfidf_model.vocabulary_.items()])

	return tfidf_model


os.system('cls')

# Load Word2Vec model here
print("LOADING WORD2VEC MODEL\n\n")
FILE = "W2V Models/w2v_reddit_unigram_300d.bin"
model = w2v.load_word2vec_format(FILE, binary=True)

# Load the dataset here
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']

# Load TF-IDF Model Here
print("LOADING TFIDF DICTIONARY \n\n")
FILE = "TFIDF models/tfidf_stop.pk"
tfidf_model = getTFIDIF(FILE)

# Transform the data
print("TRANSFORMING DATA \n\n")
MAX_WORDS = 300
X = commentFeatureVecs(X, model, tfidf_model, MAX_WORDS)

# Get the Python's file name. Remove the .py extension
file_name = os.path.basename(__file__)
file_name = file_name.replace(".py","")

# Evaluate models 
evaluate(X,y, file_name)