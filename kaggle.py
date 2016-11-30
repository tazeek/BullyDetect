import pandas as pd 
import numpy as np 
import re
import codecs
import time
import os

from sklearn import cross_validation
from bs4 import BeautifulSoup
from contractions import contractions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC

def grammarContractions(original_text):

	text = original_text.lower().split()
	reformed = [contractions[word] if word in contractions else word for word in text]

	return " ".join(reformed)

def cleaning(original_text):

	text = BeautifulSoup(original_text,"lxml").get_text()

	# Remove Encodings
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

	#Convert words to lower case
	text = text.lower().split()

	#Remove contractions by expansion of words
	text = [contractions[word] if word in contractions else word for word in text]

	#Rejoin words
	text = " ".join(text)

	#Remove non-alphabets
	text = re.sub("[^a-z\s]", " ", text)

	return " ".join(text.split())

#For Cross Validating and Evaluation of Model	
def crossValidationTest(model, vectorizer, features, df):
	cv = cross_validation.KFold(n=len(df), shuffle=False, n_folds=10)
	features = np.asarray(features)

	accuracy_array = []
	precision_array = []
	recall_array = []

	for train_cv, test_cv in cv:
		
		#Use the K-Fold for Training set and Testing set
		X_train, X_test = features[train_cv], features[test_cv]
		y_train, y_test = df.iloc[train_cv]['Insult'], df.iloc[test_cv]['Insult']
	
		#Training set transformed and fitted
		data_features_train = vectorizer.fit_transform(X_train)
		data_features_train = data_features_train.toarray()
		
		#Model trained
		model.fit(data_features_train, y_train)
		
		#Testing set transformed
		data_features_test = vectorizer.transform(X_test)
		data_features_test = data_features_test.toarray()
		
		#Predict results
		result = model.predict(data_features_test)
		
		#Get accuracy
		accuracy = accuracy_score(y_true=y_test, y_pred=result)
		recall = recall_score(y_true=y_test, y_pred=result)
		precision = precision_score(y_true=y_test, y_pred=result)
		
		accuracy_array.append(accuracy)
		recall_array.append(recall)
		precision_array.append(precision)

	print("AVERAGE ACCURACY: ", round(np.average(accuracy_array)*100, 2))
	print("AVERAGE RECALL: ", round(np.average(recall_array)*100, 2))
	print("AVERAGE PRECISION: ", round(np.average(precision_array)*100, 2))


#Read in the CSV file
df = pd.read_csv("Kaggle/dataset.csv")

# One array to store clean comments, the other to store indexes of empty comments
clean_comments = []
empty_comments = []

#Drop Latin comments (Performed by manual Data Analysis)
df.drop(df.index[[4,798,3127,6183,6347]], inplace=True)
df.reset_index(inplace=True, drop=True)

#Split Training and Testing Set
#train_df = df.iloc[:3900]

for i in range(len(df)): 
	clean_comment = cleaning(df['Comment'][i])
	# Add comment if it is not empty
	if clean_comment != "":
		clean_comments.append(clean_comment)
	else:
		empty_comments.append(i)



# Drop columns of empty comments
df.drop(df.index[empty_comments], inplace=True)
df.reset_index(inplace=True, drop=True)

# Create a new dataframe
clean_df = pd.DataFrame(data={'Insult':df['Insult'], 'Comment': clean_comments})
print("\n\n",len(clean_df))
# Save the cleaned dataset
clean_df.to_csv('clean_dataset.csv', index=False)

#Rest the index of the test set


#Implement Bag-Of-Words/N-grams/TF-IDF
#vectorizer = CountVectorizer(analyzer="word", ngram_range=(3, 3))
#vectorizer = TfidfVectorizer(stop_words='english')

#Implement Classifier here
#model = MultinomialNB()
#model = RandomForestClassifier(n_estimators=100)
#model = SVC(kernel='linear')

#crossValidationTest(model, vectorizer, clean_comments_train, train_df)
#exit()

#data_features_train = vectorizer.fit_transform(clean_comments_train)
#data_features_train = data_features_train.toarray()

#model.fit(data_features_train, train_df['Insult'])

#Implement SVM Classifier
#model = SVC(kernel='linear')
#model.fit( data_features_train, train_df["Insult"] )

#test_df = df.iloc[3900:]
#test_df.reset_index(inplace=True, drop=True)

#clean_comments_test = []
#for i in range(len(test_df)):
#	clean_comment = cleaning(test_df['Comment'][i])
#	clean_comments_test.append(clean_comment)

#data_features_test = tf_idf.transform(clean_comments_test)	
#data_features_test = vectorizer.transform(clean_comments_test)
#data_features_test = data_features_test.toarray()

#result = model.predict(data_features_test)

#test_labels = test_df.as_matrix(columns=['Insult'])

#accuracy = accuracy_score(y_true=test_df['Insult'], y_pred=result)
#recall = recall_score(y_true=test_df['Insult'], y_pred=result)
#precision = precision_score(y_true=test_df['Insult'], y_pred=result)

#print("TEST ACCURACY: ", round(accuracy*100, 2))
#print("TEST RECALL: ", round(recall*100, 2))
#print("TEST PRECISION: ", round(precision*100, 2))

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