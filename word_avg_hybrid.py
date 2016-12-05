import pandas as pd 
import numpy as np 
import time
import os
import pickle

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec as w2v

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.model_selection import cross_val_score

# The testing function
def testing(model, model_name, X_train, y_train, X_test, y_test):

	print(model_name + " STARTS HERE\n\n")

	# Train the model
	model.fit(X_train , y_train)

	# Predict and calculate run-time
	start = time.time()
	result = model.predict(X_test)
	end = time.time()

	# Confusion Matrix
	cm = confusion_matrix(y_true=y_test, y_pred=result)

	tp = cm[1][1] # True positives
	fp = cm[0][1] # False positives
	tn = cm[0][0] # True negatives
	fn = cm[1][0] # False negatives

	# Evaluation Metrics
	accuracy = accuracy_score(y_true=y_test , y_pred=result)
	precision = tp/(tp+fp)
	fdr = 1 - precision # False Discovery Rate
	fpr = fp/(fp + tn) # False Positive Rate

	#precision = precision_score(y_true=y_test, y_pred=result)

	# Display results
	print("ACCURACY: ", round(accuracy*100, 2))
	print("PRECISION: ", round(precision*100, 2))
	print("FALSE DISCOVERY RATE: ", round(fdr*100, 2))
	print("FALSE POSITIVE RATE: ", round(fpr*100, 2), "\n")
	print("TRUE POSITIVES: ", tp)
	print("FALSE POSITIVES:",fp,"\n")
	print("TRUE NEGATIVES: ", tn)
	print("FALSE NEGATIVES: ", fn,"\n")
	print("RUN TIME: ", end - start)

	print("\n\n" + model_name + " STOPS HERE\n\n")


# Use each word as a feature
def makeFeatureVec(comment, model, vect_dict, num_features):

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
			word_feature = np.mean(vect_dict[word])
		else:
			word_feature = -1.0

		# Overwrite the sentence of the numpy array
		featureVec[i] = word_feature

	return featureVec

# One of the kaggle tests
def commentFeatureVecs(comments, model, vect_dict, num_features):

	# Initialize empty counter
	counter = 0

	# Preallocate a 2D numpy array for speed
	reviewFeatureVecs = np.zeros((len(comments),num_features),dtype="float32")

	for comment in comments:

		# Call function that gets the average vectors
		reviewFeatureVecs[counter] = makeFeatureVec(comment, model, vect_dict, num_features)

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
FILE = "Word Dictionaries/vect_dict_5.p"
vect_dict = pickle.load(open(FILE,"rb"))

# Data Transformation (4th parameter indicates maximum words allowed)
MAX_WORDS = 500

print("TRANSFORMING TRAINING SET\n\n")
X_train = commentFeatureVecs(X_train, model, vect_dict, MAX_WORDS)

print("TRANSFORMING TESTING SET\n\n")
X_test = commentFeatureVecs(X_test , model, vect_dict, MAX_WORDS)

# Implement Classifier(s) here and store in dictionary
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100)
svm = LinearSVC()

models = { "Naive Bayes": nb, "Support Vector Machines": svm, "Random Forest": rf}

print("TESTING  MODELS\n\n")
for key, value in models.items():
	
	# Test each model
	testing(value, key, X_train, y_train, X_test, y_test)