import pickle
import os

os.system('cls')

# Specify the file
FILE = "Word Dictionaries/soundex_dict.pk"

# Load using pickle
soundex_word_dict = pickle.load(open(FILE,"rb")) 

# Get the unique soundex encoders
unique_soundex = list(set([value for key, value in soundex_word_dict.items()]))

# Combine all the soundex encoders as one
# Get the words. Then use the encoder as key and the list as value
soundex_words_list = {}

for index, soundex in enumerate(unique_soundex):

	# Update
	if index % 100 == 0:
		print("%i out of %i done " % (index, len(unique_soundex)))

	# Get the word list
	words_list = [ word for word, encode in soundex_word_dict.items() if encode == soundex ]

	# Store in dictionary. Encoder is key, value is words list
	soundex_words_list[soundex] = words_list


# Save it 
FILE = "Word Dictionaries/soundex_words_list.pk"
pickle.dump(soundex_words_list, open(FILE, "wb"))