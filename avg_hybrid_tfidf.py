import pandas as pd 
import numpy as np 
import os
import pickle

from collections import defaultdict

from gensim.models import Word2Vec as w2v

from evaluation import evaluate

# One of the kaggle tests 
def makeFeatureVec(words, model, vector_dict, tfidf_model, num_features):

	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,),dtype="float32")

	# Count number of words
	nwords = 0.

	# Loop over word by word
	# If in vocabulary, add its feature vector to the total
	for word in words:

		if word in model: #and word not in stop_words:
			nwords += 1.

			# Add to the vector
			avgWordsFeature = np.multiply(vector_dict[word], tfidf_model[word])
			featureVec = np.add(featureVec, avgWordsFeature)

	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)

	return featureVec

# One of the kaggle tests
def getAvgFeatureVecs(comments, model, vector_dict, tfidf_model, num_features):

	# Initialize empty counter
	counter = 0

	# Preallocate a 2D numpy array for speed
	reviewFeatureVecs = np.zeros((len(comments),num_features),dtype="float32")

	for comment in comments:

		# Call function that gets the average vectors
		reviewFeatureVecs[counter] = makeFeatureVec(comment, model, vector_dict, tfidf_model, num_features)

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

# Load TF-IDF Model Here
print("LOADING TFIDF DICTIONARY \n\n")
FILE = "TFIDF models/tfidf_stop.pk"
tfidf_model = getTFIDIF(FILE)

# Load the dataset here
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']

# Load the dictionary
print("LOADING INNOVATIVE DICTIONARY\n\n")
FILE = "Word Dictionaries/vect_dict_5.p"
vector_dict = pickle.load(open(FILE,"rb"))

# Data Transformation
print("TRANSFORMING DATA \n\n")
X = getAvgFeatureVecs(X, model, vector_dict, tfidf_model, 300)

# Get the Python's file name. Remove the .py extension
file_name = os.path.basename(__file__)
file_name = file_name.replace(".py","")

# Evaluate models 
evaluate(X,y, file_name)