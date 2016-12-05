from bs4 import BeautifulSoup
from contractions import contractions
from pymongo import MongoClient

import re
import gc
import bz2
import ujson
import time
import nltk.data
import itertools

# LIBRARIES RELATED FUNCTION (START)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

client = MongoClient() # Create client
db = client['reddit'] # Database is 'Reddit'
month = db['full_3'] # Collection is stored according to month

# LIBRARIES RELATED FUNCTION (END)

total_words = 0
total_sentences = 0
total_comments = 0

# Function to preprocess a sentence into list of words
def convertToWords(sentence):

    global total_words

    #Extract properly as in HTML
    sentence = BeautifulSoup(sentence, "lxml").get_text()

    #Remove URLs
    clean_sentence = re.sub(r'\w+:\/\/\S+', ' ', sentence)

    # Word Standardizing (Ex. Looooolll should be Looll)
    clean_sentence = ''.join(''.join(s)[:2] for _, s in itertools.groupby(clean_sentence))

    # Split attached words (Ex. AwesomeDisplay should be Awesome Display)
    #clean_sentence =  " ".join(re.findall("[A-Z][^A-Z]*", clean_sentence))

    #Convert words to lower case and split them
    words = clean_sentence.lower().split()

    #Remove contractions by expansion of words
    words = [contractions[word] if word in contractions else word for word in words]

    # Rejoin words 
    words = " ".join(words)

    # Remove non-alphabets
    words = re.sub("[^a-z\s]", " ", words)

    # Split them one last time
    words = words.split()

    # Add total number of words to total_words variable
    total_words += len(words)

    return words

# Function to split a comment into its respective sentence(s)
def convertToSentence(review):

    global total_sentences

    # 1. Split into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    # 2. Sentence Looping
    sentences = []

    for raw_sentence in raw_sentences:

        if len(raw_sentence) > 0: 

            # Get word list
            word_list = convertToWords(raw_sentence)

            # Append word list if not an empty string
            if len(word_list) > 0:
                sentences.append(word_list)

    
    # Add to total_sentences variable
    total_sentences += len(sentences)

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

    reddit_words = []

    for sentence in comment_list:
        reddit_words += convertToSentence(sentence)

    dic = { 'fragment_number': fragment_number, 'sentence_list': reddit_words }

    month.insert(dic)
    
    del reddit_words
    gc.collect()

# Function to collect comments from the respective bz2 file
def collectComments():

    global total_comments

    # Reddit Files' names
    file_list = ["RC_2015-01.bz2","RC_2015-02.bz2","RC_2015-03.bz2","RC_2015-04.bz2","RC_2015-05.bz2"]

    # Comments related
    valid_count = 0
    fragment_number = 1
    valid_comments = []
    deleted = "[deleted]"

    # Looping starts
    for file in file_list:

        extract_reddit = bz2.BZ2File(file)

        for line in extract_reddit:

            comment_str = ujson.loads(line)['body']

            if isEnglish(comment_str) and comment_str != deleted and len(comment_str) != 0:

                valid_count += 1
                valid_comments.append(comment_str)
                total_comments += 1


            # One fragment = 5,000 COMMENTS 
            if valid_count == 5000:

                print("STORING FRAGMENT NUMBER ", fragment_number, "/ 50000")

                start = time.time()
                storeComments(valid_comments, fragment_number)
                duration = time.time() - start 

                print("TIME TAKEN: ", duration, " seconds")
                print("COMMENTS READ: ", fragment_number * valid_count)
                print("CURRENT FILE:", file,"\n\n")

                # Restart the storing process
                fragment_number += 1
                valid_count = 0
                valid_comments = []

    # The last remaining comments needs to be stored
    print("STORING FRAGMENT NUMBER ", fragment_number)
    storeComments(valid_comments, fragment_number)

    return

#convertToSentence("Super Saiyan Son Goku. Super Saiyan Vegeta")
collectComments()
print("TOTAL COMMENTS: ", total_comments)
print("TOTAL SENTENCES: ", total_sentences)
print("TOTAL WORDS: ", total_words)

# MONGODB Path
# mongod --dbpath "E:\tazeek\Final Year Project\data\db"

# MONGODB File Path
# cd "C:\Program Files\MongoDB\Server\3.2\bin"