import pandas as pd 
import numpy as np 
import time
import os
import pickle

from stop_words import get_stop_words

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec as w2v

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score 
from sklearn.model_selection import cross_val_score

# Stop-word list. Remove the last ten words
#stop_words = get_stop_words('english')
#stop_words = stop_words[:(len(stop_words) - 9)]

# The testing function
def testing(model, model_name, X_train, y_train, X_test, y_test):

	print(model_name + " STARTS HERE\n\n")

	# Train the model
	model.fit(X_train , y_train)

	# Predict and calculate run-time
	start = time.time()
	result = model.predict(X_test)
	end = time.time()

	# Evaluation Metrics
	accuracy = accuracy_score(y_true=y_test , y_pred=result)
	precision = precision_score(y_true=y_test, y_pred=result)

	# Display results
	print("TEST ACCURACY: ", round(accuracy*100, 2))
	print("TEST PRECISION: ", round(precision*100, 2))
	print("RUN TIME: ", end - start)

	print("\n\n" + model_name + " STOPS HERE\n\n")

# One of the kaggle tests 
def makeFeatureVec(words, model, vector_dict, num_features):

	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,),dtype="float32")

	# Count number of words
	nwords = 0.

	# Loop over word by word
	# If in vocabulary, add its feature vector to the total
	for word in words:

		if word in model: #and word not in stop_words:
			nwords += 1.

			# Get average of the word
			#avgWordFeature = wordsAverage(word, model)

			# Add to the vector
			featureVec = np.add(featureVec, vector_dict[word])

	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)

	return featureVec

# One of the kaggle tests
def getAvgFeatureVecs(comments, model, vector_dict, num_features):

	# Initialize empty counter
	counter = 0

	# Preallocate a 2D numpy array for speed
	reviewFeatureVecs = np.zeros((len(comments),num_features),dtype="float32")

	for comment in comments:

		# Call function that gets the average vectors
		reviewFeatureVecs[counter] = makeFeatureVec(comment, model, vector_dict, num_features)

		# Increment counter
		counter += 1


	return reviewFeatureVecs

os.system('cls')

# Load Word2Vec model here
print("LOADING WORD2VEC MODEL\n\n")
FILE = "W2V Models/w2v_reddit_unigram_300d.bin"
model = w2v.load_word2vec_format(FILE, binary=True)

# Load the dataset here
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']
split = 3900

# Split the sample or make your own sample
print("SPLITTING DATA\n\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
#X_train, y_train = df['Comment'][:split], df['Insult'][:split]
#X_test, y_test = df['Comment'][split:], df['Insult'][split:]

# Load the dictionary
print("LOADING DICTIONARY\n\n")
FILE = "Word Dictionary/vect_dict_10.p"
vector_dict = pickle.load(open(FILE,"rb"))

# Data Transformation
print("TRANSFORMING TRAINING SET\n\n")
X_train = getAvgFeatureVecs(X_train, model, vector_dict, 300)

print("TRANSFORMING TESTING SET\n\n")
X_test = getAvgFeatureVecs(X_test , model, vector_dict, 300)

#X = getAvgFeatureVecs(X, model, vector_dict, 300)

# Implement Classifier(s) here and store in dictionary
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100)
svm = LinearSVC()

models = { "Naive Bayes": nb, "Support Vector Machines": svm, "Random Forest": rf}

print("TESTING  MODELS\n\n")
for key, value in models.items():
	
	# Test each model
	testing(value, key, X_train, y_train, X_test, y_test)