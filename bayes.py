import pandas as pd 
import numpy as np 
import time

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from gensim.models import Word2Vec as w2v 

# Load Word2Vec model here
model = w2v.load_word2vec_format('w2v_reddit_unigram_300d.bin', binary=True)

# Load the dataset here
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']

# Split the sample or make your own sample
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_train, y_train = df['Comment'][:4200], df['Insult'][:4200]
X_test, y_test = df['Comment'][4200:], df['Insult'][4200:]

# Convert test labels to numpy arrays
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# Transform the set
X_train = np.array([np.mean([model[word] for word in sentence if word in model]) for sentence in X_train])
X_test = np.array([np.mean([model[word] for word in sentence if word in model]) for sentence in X_test])

# Implement Classifier here
nb = GaussianNB()

# Train the classifier 
start = time.time()
nb.fit(X_train.reshape(len(X_train),1) , y_train)
end = time.time()

# Test the Classifier 
result = nb.predict(X_test.reshape(len(X_test),1))

accuracy = accuracy_score(y_true=y_test , y_pred=result)
precision = precision_score(y_true=y_test, y_pred=result)

print("TEST ACCURACY: ", round(accuracy*100, 2))
print("TEST PRECISION: ", round(precision*100, 2))
print("RUN TIME: ", round(end - start, 3))