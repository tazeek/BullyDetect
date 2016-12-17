import pickle
import os

os.system('cls')

# Specify the file
FILE = "K-Means Models/full_500C.pk"

# Load using pickle
word_centroid_map =  pickle.load(open(FILE,"rb"))

# Get the total number of clusters
total_clusters = max(word_centroid_map.values()) + 1

# Cluster all the words from various indexes as one. Then store them as dictionary
# Dictionary contents: Cluster number, cluster contents
array_dict = []
    
# Loop cluster by cluster
for cluster_num in range(0, total_clusters):
    
    # Progress report for every 50 clusters
    if (len(array_dict) % 50 == 0):

    	print("%i clusters stored" % (len(array_dict)))

    # Create a dictionary
    cluster_dict = {}

    # Get word list
    word_list = [ word for word, cluster in word_centroid_map.items() if cluster == cluster_num ]
    
    # Store in dictionary 
    cluster_dict["cluster_num"] = cluster_num
    cluster_dict["word_list"] = word_list

    # Append to array
    array_dict.append(cluster_dict)

# Save the array
FILE = "Word Dictionaries/dict_500C.pk"
pickle.dump(array_dict, open("Word Dictionaries/dict_250C.pk", "wb"))