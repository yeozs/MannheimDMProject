### Libraries
import matplotlib.pyplot as plt


### Preprocessing Function Libraries
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
sent_tokenizer = PunktSentenceTokenizer()
from transformers import BertTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

### Preprocessing function


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