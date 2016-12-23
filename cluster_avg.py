import pandas as pd 
import numpy as np 

import pickle
import os

from gensim.models import Word2Vec as w2v 

from evaluation import evaluate

def getAverageComment():

	return

def transformData(comments, cluster_dict, cluster_map, num_features):

	return

os.system('cls')
# Load the dataset here
print("LOADING DATASET \n\n")
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']

# Specify cluster file to load
cluster_file = 500

# Specify path and load files using pickle
print("LOADING K-MEANS MODELS \n\n")
FILE_DICT = "C:/Users/MyPC/Desktop/Vegito/Word Dictionaries/trans_dict_" + str(cluster_file) + "C.pk"
FILE_CLUS = "C:/Users/MyPC/Desktop/Vegito/K-Means Models/full_" + str(cluster_file) + "C.pk"

array_dict_cluster = pickle.load(open(FILE_DICT, "rb"))
word_centroid_map =  pickle.load(open(FILE_CLUS,"rb"))

# Load dataset


# Find index number of word 
# Then load all related words 
cluster_num = word_centroid_map[word]
words_list = array_dict_cluster[cluster_num]['average_vector']