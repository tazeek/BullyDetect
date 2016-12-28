import pandas as pd 
import numpy as np 
import os

from gensim.models import Word2Vec as w2v 

from evaluation import evaluate

def characterVec(words, model, num_features):
	featureVec = np.zeros((num_features,), dtype="float32")

	nwords = 0.

	for word in words:

		nwords += 1.
		featureVec = np.add(featureVec, model[word])

	featureVec = np.divide(featureVec, nwords)

	return featureVec

# One of the kaggle tests
def makeFeatureVec(words, model, num_features):

	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,),dtype="float32")

	# Count number of words
	nwords = 0.

	# Loop over word by word
	# If in vocabulary, add its feature vector to the total
	for word in words.split():

		if word in model: 
			nwords += 1.
			featureVec = np.add(featureVec,model[word])

	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)
	
	# If number of words zero
	if nwords == 0:
		featureVec = characterVec(words, model, num_features)
	
	return featureVec

# One of the kaggle tests
def getAvgFeatureVecs(comments, model, num_features):

	# Initialize empty counter
	counter = 0

	# Preallocate a 2D numpy array for speed
	reviewFeatureVecs = np.zeros((len(comments),num_features),dtype="float32")

	for comment in comments:

		# Call function that gets the average vectors
		reviewFeatureVecs[counter] = makeFeatureVec(comment, model, num_features)

		# Increment counter
		counter += 1


	return reviewFeatureVecs

os.system('cls')

# Load Word2Vec model here
print("LOADING WORD2VEC MODEL \n\n")
FILE = "W2V Models/w2v_reddit_unigram_300d.bin"
model = w2v.load_word2vec_format(FILE, binary=True)

# Load the dataset here
print("LOADING DATASET \n\n")
df = pd.read_csv('balanced_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']

# Transform the data
print("TRANSFORMING DATA \n\n")
X = getAvgFeatureVecs(X, model, 300)

# Get the Python's file name. Remove the .py extension
file_name = os.path.basename(__file__)
file_name = file_name.replace(".py","")

# Evaluate models 
print("EVALUATING \n\n")
evaluate(X,y, file_name)