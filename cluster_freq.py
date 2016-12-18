import pickle

import pandas as pd
import numpy as np 

# Function to load the cluster dictionary
def loadClusterSet(FILE):

	# File loaded here
	word_centroid_map = pickle.load(open(FILE,"rb"))

	return word_centroid_map

# Load the dataset here
print("LOADING DATASET \n\n")
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']

# Loading the cluster dictionary here
print("LOADING CLUSTER DICTIONARY \n\n")
FILE = "K-Means Models/full_500C.pk"
cluster_dictionary = loadClusterSet(FILE)