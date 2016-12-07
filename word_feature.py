import pandas as pd 
import numpy as np 
import time
import os

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec as w2v

from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.model_selection import cross_val_score

# The testing function
def testing(model, model_name, X, y, cv):

	print(model_name + " STARTS HERE\n\n")

	# Array to store results
	accuracy_array = []
	precision_array = []
	fdr_array = []
	fpr_array = []
	execution_time_array = []

	for train_cv, test_cv in cv:

		# Seperate the training and testing fold
		X_train, X_test = X[train_cv], X[test_cv]
		y_train, y_test = y[train_cv], y[test_cv]

		# Train the model
		model.fit(X_train , y_train)

		# Predict and calculate run-time
		start = time.time()
		result = model.predict(X_test)
		end = time.time()

		execution_time = end - start

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

		# Append results
		accuracy_array.append(accuracy)
		precision_array.append(precision)
		fdr_array.append(fdr)
		fpr_array.append(fpr)
		execution_time_array.append(execution_time)

	# Get mean results
	mean_accuracy = np.mean(accuracy_array)
	mean_precision = np.mean(precision_array)
	mean_fdr = np.mean(fdr_array)
	mean_fpr = np.mean(fpr_array)
	mean_execution_time = np.mean(execution_time_array)

	# Display results
	print("MEAN ACCURACY: ", round(mean_accuracy*100, 2))
	print("MEAN PRECISION: ", round(mean_precision*100, 2), "\n")
	print("MEAN FALSE DISCOVERY RATE: ", round(mean_fdr*100, 2))
	print("MEAN FALSE POSITIVE RATE: ", round(mean_fpr*100, 2), "\n")
	#print("TRUE POSITIVES: ", tp)
	#print("FALSE POSITIVES:",fp,"\n")
	#print("TRUE NEGATIVES: ", tn)
	#print("FALSE NEGATIVES: ", fn,"\n")
	print("MEAN RUN TIME: ", mean_execution_time)


# Use each word as a feature
def makeFeatureVec(comment, model, num_features):

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
		else:
			word_feature = -1.0

		# Overwrite the sentence of the numpy array
		featureVec[i] = word_feature

	return featureVec

# One of the kaggle tests
def commentFeatureVecs(comments, model, num_features):

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
print("LOADING WORD2VEC MODEL\n\n")
FILE = "W2V Models/w2v_reddit_unigram_300d.bin"
model = w2v.load_word2vec_format(FILE, binary=True)

# Load the dataset here
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']
split = 3900

# Transform the data
print("TRANSFORMING DATA \n\n")
MAX_WORDS = 300
X = commentFeatureVecs(X, model, MAX_WORDS)

# Split the sample or make your own sample
#print("SPLITTING DATA\n\n")
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)
#X_train, y_train = df['Comment'][:split], df['Insult'][:split]
#X_test, y_test = df['Comment'][split:], df['Insult'][split:]

# Data Transformation (4th parameter indicates maximum words allowed)

#print("TRANSFORMING TRAINING SET\n\n")
#X_train = commentFeatureVecs(X_train, model, MAX_WORDS)

#print("TRANSFORMING TESTING SET\n\n")
#X_test = commentFeatureVecs(X_test , model, MAX_WORDS)

# Implement Classifier(s) here and store in dictionary
print("INITIALIZING CLASSIFIERS \n\n")
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100)
svm = LinearSVC()

models = { "Naive Bayes": nb, "Support Vector Machines": svm, "Random Forest": rf}

# Test with 10 fold cross validation
print("TESTING WITH 10 FOLD CROSS VALIDATION\n\n")
cv = KFold(n=len(X), shuffle=False, n_folds=10)

print("TESTING  MODELS\n\n")
for key, value in models.items():
	
	# Test each model
	testing(value, key, X, y, cv)