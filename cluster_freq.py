import pickle
import os

import pandas as pd
import numpy as np 

# Transform the data
def createBagCentroids(comment, clusters, cluster_dictionary):

	# Pre-allocate the bag of centroids vector (for speed)
	bag_of_centroids = np.zeros( clusters, dtype="float32" )

	# Loop word by word
	for word in comment.split():
	    
	    # Check if word is in dictionary
	    if word in cluster_dictionary:
	        
	        # Get index of the word
	        index = cluster_dictionary[word]
	        
	        # Increment index of bag_of_centroids
	        bag_of_centroids[index] += 1

	return bag_of_centroids

# Read in comment by comment
def transformation(comments, cluster_dictionary):

	# Find number of clusters
	clusters = max(cluster_dictionary.values()) + 1
	print(clusters)

	# Pre-allocate an array for the transformation (for speed)
	centroids_bag = np.zeros(len(comments), clusters)

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
print("TRANSFORMING DATA \n\n")
transformation(X, cluster_dictionary)