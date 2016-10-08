from bs4 import BeautifulSoup
from contractions import contractions
from pymongo import MongoClient

import re
import gc
import sys
import bz2
import ujson
import numpy as np
import nltk.data

# LIBRARIES RELATED FUNCTION (START)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

client = MongoClient() # Create client
db = client['reddit'] # Database is 'Reddit'

REDDIT_FILE = "RC_2015-01.bz2"
month = db['january'] # Collection is stored according to month

# LIBRARIES RELATED FUNCTION (END)

#total = 0

# Function to preprocess a sentence into list of words
def convertToWords(sentence):

    #Extract properly as in HTML
    sentence = BeautifulSoup(sentence, "lxml").get_text()

    #Remove non-alphanumeric
    clean_sentence = re.sub("[^a-zA-Z']", " ", sentence)

    #Remove URLs
    clean_sentence = re.sub(r'\w+:\/\/\S+', ' ', clean_sentence)

    #Convert words to lower case and split them
    words = clean_sentence.lower().split()

    #Remove contractions by expansion of words
    words = [contractions[word] if word in contractions else word for word in words]

    # Rejoin words and then split again (Issues were contractions are treated as a single word)
    words = " ".join(words).split()

    return words

# Function to split a comment into its respective sentence(s)
def convertToSentence(review):

    #1. Split into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    #2. Sentence Looping
    sentences = []

    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0: sentences.append(convertToWords(raw_sentence))

    return sentences

# Check if the comment is English or not
def isEnglish(s):

    try:
        s.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True

# Function to store the sentences into MongoDB
def storeComments(comment_list, fragment_number):

    #global total
    reddit_words = []

    for sentence in comment_list:
        reddit_words += convertToSentence(sentence)

    #total = total + len(reddit_words)

    dic = { 'fragment_number': fragment_number, 'sentence_list': reddit_words }

    month.insert(dic)
    
    del reddit_words
    gc.collect()

# Function to collect comments from the respective bz2 file
def collectComments():

    extract_reddit = bz2.BZ2File(REDDIT_FILE)

    valid_count = 0
    fragment_number = 1
    valid_comments = []

    # Removed comments in reddit have the form '[deleted]'
    deleted = "[deleted]"

    for line in extract_reddit:

        comment_str = ujson.loads(line)['body']

        if isEnglish(comment_str) and comment_str != deleted and len(comment_str) != 0:

            valid_count += 1
            valid_comments.append(comment_str)


        # One fragment = 25,000 COMMENTS
        if valid_count == 25000:

            print("STORING FRAGMENT NUMBER ", fragment_number,"........\n\n")

            storeComments(valid_comments, fragment_number)

            # Restart the storing process
            fragment_number += 1
            valid_count = 0
            valid_comments = []

        if fragment_number == 800:
            break

    # The last remaining comments needs to be stored
    #storeComments(valid_comments, fragment_number)

    return


collectComments()
#print("TOTAL SENTENCES: ", total_comments)

# MONGODB Path
# mongod --dbpath C:\Users\MyPC\Documents\data\db

# MONGODB File Path
# cd C:\Program Files\MongoDB\Server\3.2\bin

# MOVIES TO WATCH:
# A Chinese Odyssey, Flirting Scholar, Red Cliff, I not stupid, A World Without Thieves, Curse of the Golden Flower
# You Are the Apple of My Eye