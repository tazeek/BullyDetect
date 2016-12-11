import pandas as pd 
import numpy as np 
import os

from gensim.models import Word2Vec as w2v 

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

# Evaluate models 
evaluate(X,y)