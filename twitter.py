import pandas as pd 
import numpy as np 
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

#Function used to clean the "raw" sentence
def cleanSentence(raw_sentence):

	#1. Use BeautifulSoup to clear labels
	try:
		comment_text = BeautifulSoup(raw_sentence,"lxml").get_text()
	except UserWarning:
		comment_text = "HTTPLINK"

	#2. Replace @user with @USERNAME 
	comment_text = ' '.join(re.sub("(@[A-Za-z0-9_\.]+)","@USERNAME",comment_text).split())

	#3. Replace "http//..." with HTTPLINK
	comment_text = re.sub(r"http\S+", "HTTPLINK", comment_text)

	#4. Remove Non-ASCII characters
	comment_text = (''.join([i if ord(i) < 128 else '' for i in comment_text]))

	#5. Remove escape characters
	comment_text.replace("\\","")

	#6. Lowercase all letters (Avoid confusion with Bag-of-words)
	comment_text = comment_text.lower()

	#7. Return the cleaned comment by removing all tabs and newlines via split() 
	return " ".join(comment_text.split())

df = pd.read_csv("Twitter/data.csv",index_col = False, encoding="ISO-8859-1")
df.drop('Unnamed: 0', axis=1, inplace=True)
clean_comments = []

not_null_df = df[df['Text'].notnull()]
not_null_df = pd.concat([not_null_df], ignore_index=True)

total_comments = len(not_null_df)

for i in range(0, total_comments):

	clean_comments.append( cleanSentence( not_null_df['Text'][i] ) )


#Implement Bag-Of-Words
vectorizer = CountVectorizer()
data_features = vectorizer.fit_transform(clean_comments)
data_features = data_features.toarray()

print(data_features.shape)