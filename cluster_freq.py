import pickle
import os

import pandas as pd
import numpy as np 

# Transform the data
def createBagCentroids(comment):

	# Pre-allocate the bag of centroids vector (for speed)
	bag_of_centroids = np.zeros( total_clusters_500, dtype="float32" )

	# Loop word by word
	for word in SENTENCE.split():
	    
	    # Check if word is in dictionary
	    if word in word_centroid_map_500:
	        
	        # Get index of the word
	        index = word_centroid_map_500[word]
	        
	        # Print for evalution
	        print(word, index)
	        
	        # Increment index of bag_of_centroids
	        bag_of_centroids[index] += 1

	return bag_of_centroids

# Read in comment by comment

# Function to load the cluster dictionary
def loadClusterSet(FILE):

	# File loaded here
	word_centroid_map = pickle.load(open(FILE,"rb"))

	return word_centroid_map

os.system('cls')

# Load the dataset here
print("LOADING DATASET \n\n")
df = pd.read_csv('clean_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']

# Loading the cluster dictionary here
print("LOADING CLUSTER DICTIONARY \n\n")
FILE = "K-Means Models/full_500C.pk"
cluster_dictionary = loadClusterSet(FILE)

# Transform the data 