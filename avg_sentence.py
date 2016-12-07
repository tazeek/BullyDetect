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

	print("\n\n" + model_name + " STOPS HERE\n\n")


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
#split = 4200

# Split the sample or make your own sample
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#X_train, y_train = df['Comment'][:split], df['Insult'][:split]
#X_test, y_test = df['Comment'][split:], df['Insult'][split:]

# Convert test labels to numpy arrays
#y_train = np.asarray(y_train)
#y_test = np.asarray(y_test)

# Transform the set
print("TRANSFORMING DATA \n\n")
X = np.array([np.mean([model[word] for word in sentence if word in model]) for sentence in X])
X = X.reshape(len(X),1)
#X_train = np.array([np.mean([model[word] for word in sentence if word in model]) for sentence in X_train])
#X_test = np.array([np.mean([model[word] for word in sentence if word in model]) for sentence in X_test])

# Implement Classifier(s) here and store in dictionary
print("INITIALIZING CLASSIFIERS \n\n")
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100)
svm = LinearSVC()

models = { "Naive Bayes": nb, "Support Vector Machines": svm, "Random Forest": rf}

# Test with 10 fold Cross validation
print("TESTING WITH 10 FOLD CROSS VALIDATION \n\n")
cv = KFold(n=len(X), shuffle=False, n_folds=10)

for key, value in models.items():
	testing(value, key, X, y, cv)