import pandas as pd 
import numpy as np 
import time
import os
import pickle

from stop_words import get_stop_words
from collections import defaultdict

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec as w2v 

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# Stop-word list. Remove the last ten words
#stop_words = get_stop_words('english')
#stop_words = stop_words[:(len(stop_words) - 9)]

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

# One of the kaggle tests
def makeFeatureVec(words, model, tfidf_model, num_features):

	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,),dtype="float32")

	# Count number of words
	nwords = 0.

	# Loop over word by word
	# If in word2vec vocabulary, add its feature vector to the total
	for word in words:

		if word in model: #and word not in stop_words:
			nwords += 1.

			# Transform the word feature
			wordFeature = np.multiply(model[word], tfidf_model[word])
			featureVec = np.add(featureVec,wordFeature)

	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)

	return featureVec

# One of the kaggle tests
def getAvgFeatureVecs(comments, model, tfidf_model, num_features):

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
print("LOADING WORD2VEC MODEL \n\n")
FILE = "W2V Models/w2v_reddit_unigram_300d.bin"
model = w2v.load_word2vec_format(FILE, binary=True)

# Load TF-IDF Model Here
print("LOADING TFIDF DICTIONARY \n\n")
FILE = "TFIDF models/tfidif_stop.pk"
tfidf_model = getTFIDIF(FILE)

# Load the dataset here
print("LOADING DATASET \n\n")
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']
split = 3900

# Split the sample or make your own sample
print("SPLITTING DATASET \n\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
#X_train, y_train = df['Comment'][:split], df['Insult'][:split]
#X_test, y_test = df['Comment'][split:], df['Insult'][split:]

# Data Transformation
print("TRANSFORMING DATA \n\n")
#X = getAvgFeatureVecs(X, model, 300)
X_train = getAvgFeatureVecs(X_train, model, tfidf_model, 300)
X_test  = getAvgFeatureVecs(X_test , model, tfidf_model, 300)

# Implement Classifier(s) here and store in dictionary
print("INITIALIZING MODELS \n\n")
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100)
svm = LinearSVC()

models = { "Naive Bayes": nb, "Support Vector Machines": svm, "Random Forest": rf}

for key, value in models.items():
	testing(value, key, X_train, y_train, X_test, y_test)
	#score = np.mean(cross_val_score(value, X, y, cv=10))
	#print(key," - ", score)