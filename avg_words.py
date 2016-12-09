import pandas as pd 
import numpy as np 
import time
import os

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec as w2v 

from sklearn.cross_validation import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

def testing(model, model_name, X, y, cv):

	print(model_name + " STARTS HERE\n\n")

	# Array to store results
	accuracy_array = []
	precision_array = []
	fdr_array = []
	fpr_array = []
	execution_time_array = []

	for train_cv, test_cv in cv.split(X,y):

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

	# Get standard deviation (population)
	accuracy_std = np.std(accuracy_array)
	precision_std = np.std(precision_array)
	fdr_std = np.std(fdr_array)
	fpr_std = np.std(fpr_array)
	run_std = np.std(mean_execution_time)

	# Display results
	print("MEAN ACCURACY: %0.2f (+/- %0.2f) \n" % (mean_accuracy*100, accuracy_std * 100))
	print("MEAN PRECISION: %0.2f (+/- %0.2f) \n" % (mean_precision*100, precision_std * 100))
	print("MEAN FALSE DISCOVERY RATE: %0.2f (+/- %0.2f) \n" % (mean_fdr*100, fdr_std*100))
	print("MEAN FALSE POSITIVE RATE: %0.2f (+/- %0.2f) \n" % (mean_fpr*100, fpr_std*100))
	print("MEAN RUN TIME: %0.2f (+/- %0.2f) \n" % (mean_execution_time, run_std * 100))

	print("\n\n" + model_name + " STOPS HERE\n\n")

# One of the kaggle tests
def makeFeatureVec(words, model, num_features):

	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,),dtype="float32")

	# Count number of words
	nwords = 0.

	# Loop over word by word
	# If in vocabulary, add its feature vector to the total
	for word in words:

		if word in model: #and word not in stop_words:
			nwords += 1.
			featureVec = np.add(featureVec,model[word])

	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)

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
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']
split = 3900

# Transform the data
print("TRANSFORMING DATA \n\n")
X = getAvgFeatureVecs(X, model, 300)

# Split the sample or make your own sample or skip
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
#X_train, y_train = df['Comment'][:split], df['Insult'][:split]
#X_test, y_test = df['Comment'][split:], df['Insult'][split:]

# Data Transformation
#X = getAvgFeatureVecs(X, model, 300)
#X_train = getAvgFeatureVecs(X_train, model, 300)
#X_test = getAvgFeatureVecs(X_test , model, 300)

# Implement Classifier(s) here and store in dictionary
print("INITLIAZING CLASSIFIERS \n\n")
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100)
svm = LinearSVC()

models = { "Naive Bayes": nb, "Support Vector Machines": svm, "Random Forest": rf}


# Test with 10 fold Cross validation/Stratified K Fold
print("TESTING WITH STRATIFIED K FOLD \n\n")
#cv = KFold(n=len(X), shuffle=False, n_folds=10)
skf = StratifiedKFold(n_splits=5)

for key, value in models.items():
	testing(value, key, X, y, skf)