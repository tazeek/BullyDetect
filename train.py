#!python3
from gensim.models import Word2Vec, Phrases
from pymongo import MongoClient

import logging
import multiprocessing
import os
import time
import numpy as np

# Utilize the full power of the worker threads available
os.system("taskset -p 0xff %d" % os.getpid())

class MySentences():

	def __iter__(self):

		client = MongoClient() # First: Connect to MongoDB
		db = client['reddit'] # Second: Connect to Database
		collection = db['full_2'] # Third: Get collection list

		for fragments in collection.find():
			for sentence in fragments['sentence_list']:
				yield sentence

		#for month in collection_list:
		#	collection = db[month] # Third: Get the collections


		#	for fragments in collection.find():

		#		for sentence in fragments['sentence_list']:
		#			yield sentence



word_list = MySentences()

print("DATA LOADED SUCCESSFULLY.....\n\n")

#For Logging Parameters
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Word2Vec Parameters (START)

sg = 1 # Use Skip-Gram Model
size = 300 # Dimensionality size of Projection Layer
window = 5 # Window of surrounding words
alpha = 0.025 # Initial learning rate of the Neural Network
min_count = 5 # Minimum Frequency of Words
workers = multiprocessing.cpu_count() # Number of workers
max_vocab_size = 8000000 # Maximum number of Unique Words
negative = 10 # Number of words to be drawn for Negative Sampling
sample = 0.001 # Subsampling of frequent words 
hs = 0 # Negative Sampling to be used
iter = 5 # Iterations over corpus. Also called epochs

#Word2Vec Parameters (END)

# WORD2VEC RAM FORMULA (IN GIGABYTES):
# (Estimated Number of Unique Words x Dimension Size x 12)/1,000,000,000 

#Initialize Bigram Transformer
#bigram_transformer = Phrases(word_list)

os.system('cls')

#Initialize Word2Vec model 
model = Word2Vec(word_list, sg=sg, size=size, window=window, alpha=alpha, min_count=min_count, workers=workers, max_vocab_size=max_vocab_size, hs=hs, iter=iter, sample=sample)
model_name = "reddit_full_UNI"

model.init_sims(replace=True) # Trim down memory size
model.save_word2vec_format(model_name + '.bin', binary=True) 