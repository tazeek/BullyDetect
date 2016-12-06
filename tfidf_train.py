import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv('clean_dataset.csv')

# Load the comments
X = df['Comment']

# Initialize the model: One with stop words, One without
tfidf_stop_words = TfidfVectorizer()
tfidf = TfidfVectorizer(stop_words='english')

# Train both
tfidf_stop_words.fit(X)
tfidf.fit(X)

# Save both models
