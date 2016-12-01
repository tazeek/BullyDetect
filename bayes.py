import pandas as pd 
import numpy as np 

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cross_validation import train_test_split
from gensim.models import Word2Vec as w2v 

# Load Word2Vec model here
w2v_model = w2v.load_word2vec_format('w2v_reddit_unigram_300d.bin', binary=True)

# Load the dataset here
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']

# Split the sample (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Implement Classifier here
nb = MultinomialNB()