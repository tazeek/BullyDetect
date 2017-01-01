import pandas as pd 
import numpy as np 
import pickle
import os

from gensim.models import Word2Vec as w2v

mean_cos_array = []

# Dr. Soon's idea 
def wordsAverage(word, model):

	global mean_cos_array

	# Pre-initialize an empty numpy array (for speed)
	# 300 is used, as it is the number of vectors in Word2Vec
	avgWordsFeature = np.zeros((300,),dtype="float32")

	# Get words that are similar. This returns tuples in a list
	# Topn refers to the Top N words. 10 is default
	top_n = 5

	# Get words that are similar. This returns tuples in a list
	similar_words = model.most_similar(word, topn=top_n)

	# Calculate the Mean Cosine similarity among words and append it to the array
	mean_cos_distance = np.mean([ cos_distance for word, cos_distance in similar_words ])
	mean_cos_array.append(mean_cos_distance)

	# Get the collected words that are similar above this score. 
	# Get the number of words as well
	words_above_mean = [word for word, cos_distance in similar_words if cos_distance > mean_cos_distance]

	total_words = float(len(words_above_mean))

	# Loop over each word
	for word in words_above_mean:

		# Add the word's vector
		avgWordsFeature = np.add(avgWordsFeature,model[word])

	# Average them out
	avgWordsFeature = np.divide(avgWordsFeature,total_words)

	# Return them
	return avgWordsFeature

# Function to transform the unique words
def createVectorDictionary(unique_words, model):

	vector_dict = {}

	# Loop word by word
	for i, word in enumerate(unique_words):

		# Status checker
		if i % 100 == 0:
			print("%d out of %d preprocessed" % (i, len(unique_words)))

		# Check if word is in model
		if word in model:

			# Get the average feature
			avgWordFeature = wordsAverage(word, model)

			# Add to dictionary
			vector_dict[word] = avgWordFeature

	return vector_dict 

# Function to get the unique words
def getUniqueWords(comments):

	unique_words = []

	# Loop comment by comment
	for comment in comments:

		# Loop word by word
		for word in comment.split():

			# Append the word if not present
			if word not in unique_words:
				unique_words.append(word)

	return unique_words

os.system('cls')

# Load Word2Vec model here
print("LOADING WORD2VEC MODEL\n\n")
FILE = "W2V Models/w2v_reddit_unigram_300d.bin"
model = w2v.load_word2vec_format(FILE, binary=True)

# Load the dataset here and load comments
df = pd.read_csv('clean_dataset.csv')
X = df['Comment']

# Get unique words
print("GETTING UNIQUE WORDS LIST\n\n")
unique_words = getUniqueWords(X)

# Create the dictionary (200K vs 20K)
# After creating, save using pickle
print("CREATING WORD-TRANSFORMED DICTIONARY\n\n")
vector_dict = createVectorDictionary(unique_words, model)

#print("SAVING THE DICTIONARY")
#FILE = "Word Dictionaries/vect_dict_5.p"
#pickle.dump(vector_dict, open(FILE, "wb"))

# Get overall mean cosine similarity 
overall_cos_mean = np.mean(mean_cos_array)
overall_cos_median = np.median(mean_cos_array)
overall_cos_mode = np.bincount(mean_cos_array).argmax()

print("COSINE MEAN: ", round(overall_cos_mean, 4))
print("COSINE MEDIAN: ",  round(overall_cos_median, 4))
print("COSINE MODE: ", round(overall_cos_mode, 4))