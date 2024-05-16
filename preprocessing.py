# Libraries

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
sent_tokenizer = PunktSentenceTokenizer()
from transformers import BertTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from gensim.models import KeyedVectors

import numpy as np
import pandas as pd
import emoji
import string
import random

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Score Converter
    # Create new function to convert Rating scores to 3 categories
    # 1-4, 5-6, 7-10 forms negative, neutral, positive [0,1,2]

def score_convert_senti(score):
        if score <= 4:
            return 0
        elif score <= 6:
            return 1
        else:
            return 2

# Reviews Preprocessor

def stopword_remover(tokenized_comment, stop_words):
    clean_text = []
    for token in tokenized_comment:
        if token not in stop_words:
            clean_text.append(token)
    return clean_text

def reviews_preprocessor(reviews,
                 remove_punctuation = False,
                 lowercase = False,
                 tokenized_output = False,
                 bert_tokenization = False,
                 remove_stopwords = True,
                 lemmatization = False,
                 stemming = False,
                 sentence_output = True):
    
    clean_text = reviews
    stop_words = set(stopwords.words('english'))
    
    
    
    # Punctuation
    if remove_punctuation:
        clean_text = re.compile(r'[^\w\s]').sub(' ', clean_text)
        
    # Lowercase    
    if lowercase:
        clean_text = clean_text.lower()
    
    #Tokenisation  
    clean_text = word_tokenize(str(clean_text))
    
    # Stopwords
    if remove_stopwords:
        clean_text = stopword_remover(clean_text, stop_words)
    
    # Lemmatisation and Stemming
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        clean_text = [lemmatizer.lemmatize(token) for token in clean_text]
        
    elif stemming:
        stemmer = PorterStemmer()
        clean_text = [stemmer.stem(token) for token in clean_text]
        
     # Removing Tokenisation    
    if tokenized_output == False:
        #re-join
        clean_text = " ".join(clean_text)
        #Remove space before punctuation
        clean_text = re.sub(r'(\s)(?!\w)','',clean_text)
        
    elif bert_tokenization:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        clean_text = tokenizer.tokenize(clean_text) 

    if sentence_output:
        clean_text = sent_tokenize(str(clean_text))
    
    
    return clean_text


## LSTM preprocessing

# Load pre-trained GloVe embeddings
def load_glove_embeddings(filepath):
    return KeyedVectors.load_word2vec_format(filepath, binary=False)

# Load GloVe embeddings
glove_embeddings = load_glove_embeddings('path_to_glove/glove.6B.100d.txt')

# Function to get GloVe vector for a token
def get_glove_vector(token):
    try:
        return glove_embeddings[token]
    except KeyError:
        # If token not found in GloVe, return zeros
        return np.zeros(glove_embeddings.vector_size)

def lstm_preprocessing(dataset: pd.DataFrame, tokenizer=word_tokenize):
    random.seed(20)
    np.random.seed(20)

    #Create new column, convert scoring into 3 categories
    dataset["Sentiment"] = dataset["Overall Rating"].apply(score_convert_senti)

    #place reviews column textual data into list
    reviews = dataset["Reviews"]
    reviews_list = list(reviews)


    #check for emojis
    def contain_emoji(review):
        emoList = emoji.emoji_list(review)
        
        if emoList:
            return True
        
        return False

    emoji_check = [contain_emoji(review) for review in reviews_list]
    reviews_list_deemojize = reviews_list.copy()
    for i in range(len(emoji_check)):
        if emoji_check[i] == True:
            reviews_list_deemojize[i] = emoji.demojize(reviews_list_deemojize[i], language='en')


    #Remove Punctuation
    def remove_punc(review):
        ascii_to_translate = str.maketrans("", "", string.punctuation)
        review = review.translate(ascii_to_translate)
        return review

    reviews_list_noPunc = [remove_punc(review) for review in reviews_list_deemojize]


    #Make text all lowercase
    reviews_list_lower = [review.lower() for review in reviews_list_noPunc]


    #Tokenization
    rev_tokenized = [tokenizer(review) for review in reviews_list_lower]


    # GloVe Vectorization
    rev_glove_vectors = []
    for review_tokens in rev_tokenized:
        glove_vectors = [get_glove_vector(token) for token in review_tokens]
        rev_glove_vectors.append(glove_vectors)



    #Output
    ret_dataset = dataset.copy()
    assert type(ret_dataset) is pd.DataFrame
    ret_dataset["Tokenized_Reviews"] = rev_tokenized


    return ret_dataset