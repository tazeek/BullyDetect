from gensim.models import Word2Vec as w2v

def getSoundex(word):
    
    # Uppercase the word
    word = word.upper()

    # Get the first letter of the word
    soundex = word[0]

    # Skip the following letters
    skip_dict = "HW"
    word = [letter for letter in word[1:] if letter not in skip_dict]
    word = "".join(word)

    # Create soundex dictionary
    dictionary = {"BFPV": "1", "CGJKQSXZ":"2", "DT":"3", "L":"4", 
        "MN":"5", "R":"6", "AEIOUHWY":"."}

    # Loop character by character (Start with 2nd character)
    for char in word[1:]:
        
        # Loop key-by-key
        for key in dictionary.keys():
            
            # Check if the character is in key list
            if char in key:
                
                # Variable to store the code
                # Ignore if it is the same as last letter
                code = dictionary[key]
                
                if code != soundex[-1]:
                    soundex += code

    # Replace vowels and HWY 
    soundex = soundex.replace(".", "")

    # If the string has only one character, append rest with three 0s.
    soundex = soundex[:4].ljust(4, "0")

    return soundex

# Testing
print(getSoundex("Gutierrez"))

# Load word2vec model 
print("LOADING MODEL")