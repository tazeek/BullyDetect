from contractions import soundex_dictionary
import pickle

def getSoundex(word):
    
    # Uppercase the word
    word = word.upper()

    # Get the first letter of the word
    soundex = word[0]

    # Skip the following letters
    skip_dict = "HW"
    word = [letter for letter in word[1:] if letter not in skip_dict]
    word = "".join(word)

    # Loop character by character (Start with 2nd character)
    for char in word[0:]:

        code = soundex_dictionary[char]

        if code != soundex[-1]:
            soundex += code

    # Replace period characters
    soundex = soundex.replace(".", "")

    # If the string has only one character, append rest with three 0s.
    soundex = soundex[:4].ljust(4, "0")

    return soundex


# Load K-Means model here
# We use this over W2V model, due to memory constraints and loading time
FILE = "K-Means Models/full_500C.pk"
word_centroid_map =  pickle.load(open(FILE,"rb"))

# Create dictionary
soundex_dict = {}

# Loop one pair at a time
# Word is stored in key
counter = 0

for key, value in word_centroid_map.items():

    soundex_key = getSoundex(key)

    soundex_dict[key] = soundex_key

    if counter % 10000 == 0:
        print("%i words encoded" % (counter))

    counter += 1

# Save dictionary 
FILE = "Word Dictionaries/soundex_dict.pk"
pickle.dump(soundex_dict, open(FILE, "wb"))