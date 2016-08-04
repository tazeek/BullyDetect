import pandas as pd 
import numpy as np 
import re
import codecs
import time

from bs4 import BeautifulSoup
from contractions import contractions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def grammarContractions(original_text):
	text = original_text.lower().split()
	reformed = [contractions[word] if word in contractions else word for word in text]

	return " ".join(reformed)

def cleaning(original_text):

	text = BeautifulSoup(original_text,"lxml").get_text()

	#Encodings.....
	text = re.sub(r'\\\\', r'\\', text)
	text = re.sub(r'\\x\w{2,2}',' ', text)
	text = re.sub(r'\\u\w{4,4}', ' ', text)
	text = re.sub(r'\\n', '.', text)

	#Whitespace Formatting
	text = text.replace('"', ' ')
	text = text.replace('\\', ' ')
	text = text.replace('_', ' ')
	text = text.replace('-', ' ')
	text = re.sub(' +',' ', text)
	
	#Remove Unicode characters
	text = codecs.decode(text, 'unicode-escape')
	text = ''.join([i if ord(i) < 128 else '' for i in text])

	#Remove email addresses
	text = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', ' ', text)
	
	#Remove Twitter Usernames
	text = re.sub(r"(\A|\s)@(\w+)+[a-zA-Z0-9_\.]", ' ', text)

	#Remove urls
	text = re.sub(r'\w+:\/\/\S+', ' ', text)

	return grammarContractions(text)

clean_comments_train = []
df = pd.read_csv("Kaggle/dataset.csv")

#Drop Latin comments (Performed by manual Data Analysis)
df.drop(df.index[[4,798,3127,6183,6347]], inplace=True)
df.reset_index(inplace=True, drop=True)

#Split Training and Testing Set
train_df = df.iloc[:4610]
test_df = df.iloc[4610:]

#Rest the index of the test set
test_df.reset_index(inplace=True, drop=True)

#comment = 'P.S. And drop that loser Suzy.\xa0 You are a LOSER and a trouble maker.\xa0 Exit stage left, and go get a job.\xa0 It is a RAP!!!!!'

for i in range(len(train_df)): 
	clean_comment = cleaning(df['Comment'][i])
	clean_comments_train.append(clean_comment)

# clean_file = open('clean2.txt', 'w')

# for comment in clean_comments:
# 	clean_file.write(comment + "\n")
# print("WRITING TO FILE COMPLETE")

#Implement Bag-Of-Words
vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 1))
data_features_train = vectorizer.fit_transform(clean_comments_train)
data_features_train = data_features_train.toarray()

#Implement TF-IDF Vectorizer
# tf_idf = TfidfVectorizer()
# data_features_train = tf_idf.fit_transform(clean_comments_train)
# data_features_train = data_features_train.toarray()

#Implement Random Forest ML
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( data_features_train, train_df["Insult"] )

#Implement Naive-Bayes Classifier
# nb = GaussianNB()
# nb.fit( data_features_train, train_df["Insult"] )

#Implement SVM Classifier
# model = SVC(kernel='linear')
# model.fit( data_features_train, train_df["Insult"] )


clean_comments_test = []
for i in range(len(test_df)):
	clean_comment = cleaning(test_df['Comment'][i])
	clean_comments_test.append(clean_comment)

data_features_test = vectorizer.transform(clean_comments_test)
data_features_test = data_features_test.toarray()

#MACHINE LEARNING PREDICTION

start_time = time.time()

result = forest.predict(data_features_test)
#result = nb.predict(data_features_test)
#result = model.predict(data_features_test)

end_time = time.time()

duration = round(end_time - start_time, 2)


test_labels = test_df.as_matrix(columns=['Insult'])

score = accuracy_score(y_true=test_df['Insult'], y_pred=result)
print("SCORE: ", round(score*100, 2))
print("TIME TAKEN: ", duration)

# ACCURACY SCORE
# ==============
# BoW + RF = 84% (6000 to 589)
# BoW + NB = 66% (6000 to 589)
# BoW + SVM = 85% (6000 to 589)

# TF-IDF + RF = 82% (6000 to 589)
# TF-IDF + NB = 65% (6000 to 589)
# TF-IDF + SVM = 87% (6000 to 589)

# 2-grams + RF = 80% (6000 to 589)
# 2-grams + NB = 72% (6000 to 589)
# 2-grams + SVM = EMERGENCY!

# 3-grams + RF = EMERGENCY!
# 3-grams + NB = EMERGENCY!
# 3-grams + SVM = EMERGENCY

#NETLINGO for Arconyms
#Spellchecker using regex
#Google Distance
#Frequency Words