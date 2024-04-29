import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

import numpy as np
import pandas as pd
from nltk import word_tokenize

import emoji
import string

import random


def lstm_preprocessing(dataset: pd.DataFrame, tokenizer=word_tokenize):
    random.seed(20)
    np.random.seed(20)

    #Create new function to convert Rating scores to 3 categories
    #1-3, 4-7, 8-10 forms negative, neutral, positive [0,1,2]
    def score_convert_senti(score):
        if score <= 3:
            return 0
        elif score >= 4 and score <= 7:
            return 1
        elif score >= 8:
            return 2

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


    #TODO: Add vectorization (use flag)


    #Output
    ret_dataset = dataset.copy()
    assert type(ret_dataset) is pd.DataFrame
    ret_dataset["Tokenized_Reviews"] = rev_tokenized


    return ret_dataset