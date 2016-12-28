import pickle
import os

import pandas as pd
import numpy as np 

from evaluation import evaluate

# Transform the data
def createBagCentroids(comment, clusters, cluster_dictionary):

	# Pre-allocate the bag of centroids vector (for speed)
	bag_of_centroids = np.zeros( clusters, dtype="float32" )

	# Loop word by word
	for word in comment.split():
	    
	    # Assume word is not in dictionary
	    index = -1

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
	# Use extra cluster to store unseen words. This is the last cluster
	clusters = max(cluster_dictionary.values()) + 2

	# Pre-allocate an array for the transformation (for speed)
	centroids_bag = np.zeros((len(comments), clusters), dtype="float32")

	# Initialize counter
	counter = 0

	# Loop over comment by comment
	for comment in comments:

		# Overwrite current row with transformed data
		centroids_bag[counter] = createBagCentroids(comment, clusters, cluster_dictionary)

		# Increment counter
		counter += 1

	return centroids_bag

# Function to load the cluster dictionary
def loadClusterSet(FILE):

	# File loaded here
	word_centroid_map = pickle.load(open(FILE,"rb"))

	return word_centroid_map

os.system('cls')

# Load the dataset here
print("LOADING DATASET \n\n")
df = pd.read_csv('balanced_dataset.csv')

# Separate out comments and labels
X , y = df['Comment'], df['Insult']

# Loading the cluster dictionary here
print("LOADING CLUSTER DICTIONARY \n\n")
FILE = "K-Means Models/full_500C.pk"
cluster_dictionary = loadClusterSet(FILE)

# Transform the data 
print("TRANSFORMING DATA \n\n")
X = transformation(X, cluster_dictionary)

# Get the Python's file name. Remove the .py extension
file_name = os.path.basename(__file__)
file_name = file_name.replace(".py","")

# Send in for evaluation
evaluate(X,y, file_name)