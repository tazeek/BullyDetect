import pandas as pd 
import numpy as np 
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec as w2v 

from sklearn.model_selection import StratifiedKFold

from evaluation import evaluatingModel


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

# Transform the set
# NOTE: You need to reshape the file since it is only ONE feature
print("TRANSFORMING DATA \n\n")
X = np.array([np.mean([model[word] for word in sentence if word in model]) for sentence in X])
X = X.reshape(len(X),1)

# Implement Classifier(s) here and store in dictionary
print("INITLIAZING CLASSIFIERS \n\n")
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100)
svm = LinearSVC()

# Store them in a dicitonary
models = { "NB": nb, "SVM": svm, "RF": rf}


# Test with 10 fold Cross validation/Stratified K Fold
skf = StratifiedKFold(n_splits=10)

for key, value in models.items():
	evaluatingModel(value, key, X, y, skf)