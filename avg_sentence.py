import pandas as pd 
import numpy as np 
import time

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from gensim.models import Word2Vec as w2v 

def testing(model, model_name, X_train, y_train, X_test, y_test):

	print(model_name + " STARTS HERE\n\n")

	# Train the model
	model.fit(X_train.reshape(len(X_train),1) , y_train)

	# Predict and calculate run-time
	start = time.time()
	result = model.predict(X_test.reshape(len(X_test),1))
	end = time.time()

	# Evaluation Metrics
	accuracy = accuracy_score(y_true=y_test , y_pred=result)
	precision = precision_score(y_true=y_test, y_pred=result)

	# Display results
	print("TEST ACCURACY: ", round(accuracy*100, 2))
	print("TEST PRECISION: ", round(precision*100, 2))
	print("RUN TIME: ", round(end - start, 3))

	print("\n\n" + model_name + " STOPS HERE\n\n")


# Load Word2Vec model here
FILE = "W2V Models/w2v_reddit_unigram_300d.bin"
model = w2v.load_word2vec_format(FILE, binary=True)

# Load the dataset here
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']
split = 4200

# Split the sample or make your own sample
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train, y_train = df['Comment'][:split], df['Insult'][:split]
X_test, y_test = df['Comment'][split:], df['Insult'][split:]

# Convert test labels to numpy arrays
#y_train = np.asarray(y_train)
#y_test = np.asarray(y_test)

# Transform the set
X_train = np.array([np.mean([model[word] for word in sentence if word in model]) for sentence in X_train])
X_test = np.array([np.mean([model[word] for word in sentence if word in model]) for sentence in X_test])

positive_train = [ value for value in X_train if value > 0]
positive_test = [ value for value in X_test if value > 0]

#print(len(positive_train))
#print(len(positive_test))
#exit()

# Implement Classifier(s) here and store in dictionary
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100)
svm = LinearSVC()

models = { "Naive Bayes": nb, "Support Vector Machines": svm, "Random Forest": rf}

for key, value in models.items():
	testing(value, key, X_train, y_train, X_test, y_test)