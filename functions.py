### Libraries
import matplotlib.pyplot as plt

# stopwords
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

### Preprocessing Function

################ Imports ################

# pandas and numpy 
import pandas as pd
import numpy as np

# regex
import regex as re

# Preprocessing
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
sent_tokenizer = PunktSentenceTokenizer()
import unicodedata
lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import ast
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Visualization


################ Functions ################


def stopword_remover(tokenized_comment,stop_words = stop_words):
    clean_comment = []
    for token in tokenized_comment:
        if token not in stop_words:
            clean_comment.append(token)
    return clean_comment

def preprocessor(raw_text, 
                 lowercase=True, 
                 leave_punctuation = False, 
                 remove_stopwords = True,
                 stop_words = None,
                 correct_spelling = False, 
                 lemmatization=False, 
                 porter_stemming=False,
                 tokenized_output=False, 
                 sentence_output=False
                 ):
    

    clean_text = raw_text

    if lowercase == True:
        
    #convert to lowercase
        if any(ord(char) > 127 for char in clean_text):
            clean_text =  ''.join([unicodedata.normalize('NFKD', char).lower() for char in clean_text])
        else:
            clean_text = clean_text.lower()
        
    #remove newline characters
    clean_text = re.sub(r'(\**\\[nrt]|</ul>)',' ',clean_text) 
    
    if leave_punctuation == False:    
    #remove punctuation:
        clean_text = re.compile(r'[^\w\s]').sub(' ', clean_text)
        #clean_text = re.sub(r'([\.\,\;\?\!\:\'])',' ',clean_text) acho que o de cima funciona melhor pq retira { e (
        
    #remove url:
    clean_text = re.sub(r'(\bhttp[^\s]+\b)',' ',clean_text)
    
    #remove isolated consonants:
    clean_text = re.sub(r'\b([^aeiou-])\b',' ',clean_text)

    #correct spelling
    if correct_spelling == True:
        incorrect_text = TextBlob(clean_text)
        clean_text = incorrect_text.correct()
        
    #tokenize
    clean_text = word_tokenize(str(clean_text))
    
    #remove stopwords
    if remove_stopwords == True:
    	clean_text = stopword_remover(clean_text,stop_words)
        
    #lemmatize
    if lemmatization == True:
        for pos_tag in ["v","n","a"]:
            clean_text = [lemmatizer.lemmatize(token, pos=pos_tag) for token in clean_text]
            
    elif porter_stemming == True:  
        porter_stemmer = PorterStemmer()
        clean_text = [porter_stemmer.stem(token) for token in clean_text]
    
    if tokenized_output == False:
    #re-join
        clean_text = " ".join(clean_text)
    #Remove space before punctuation
        clean_text = re.sub(r'(\s)(?!\w)','',clean_text)

    if sentence_output == True:
        #split into sentences:
        clean_text = sent_tokenizer.tokenize(str(clean_text))
    
    return clean_text


### Histogram 

def histogram(data):
    plt.hist(data, bins=22, color='#FF914D', ec='#F16007', alpha=0.7, label='Data Points')  # Adjust the number of bins
    plt.title('Histogram')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()

    # Set x-axis limits to include a larger range
    plt.xlim(min(data), max(data)) 

    plt.show()