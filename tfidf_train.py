import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
print("LOADING DATASET\n\n")
df = pd.read_csv('clean_dataset.csv')

# Load the comments
print("LOADING COMMENTS\n\n")
X = df['Comment']

# Initialize the model: One with stop words, One without
print("INITIALIZING TFIDF WITH STOP WORDS\n\n")
tfidf_stop_words = TfidfVectorizer()

print("INITIALIZING TFIDF WITHOUT STOP WORDS\n\n")
tfidf = TfidfVectorizer(stop_words='english')

# Train both
print("TRAINING TFIDF WITH STOP WORDS\n\n")
tfidf_stop_words.fit(X)

print("TRAINING TFIDF WITHOUT STOP WORDS\n\n")
tfidf.fit(X)

# Save both models
print("SAVING TFIDF WITH STOP WORDS\n\n")
FILE = "TFIDF models/tfidif_stop.pk"
pickle.dump(tfidf_stop_words, open(FILE, "wb"))

print("SAVING TFIDIF WITHOUT STOP WORDS\n\n")
FILE = "TFIDF models/tfidf_normal.pk"
pickle.dump(tfidf, open(FILE, "wb"))