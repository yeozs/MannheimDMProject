### Libraries
import matplotlib.pyplot as plt

# stopwords
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

### Preprocessing Function

################ Imports ################
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import regex as re
import unicodedata
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob
import emoji

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Visualization


################ Functions ################



def stopword_remover(tokenized_comment, stop_words):
    clean_comment = []
    for token in tokenized_comment:
        if token not in stop_words:
            clean_comment.append(token)
    return clean_comment

def preprocessor(raw_text, 
                 lowercase=True, 
                 leave_punctuation=False, 
                 remove_stopwords=True,
                 stop_words=None,
                 correct_spelling=False, 
                 lemmatization=False, 
                 porter_stemming=False,
                 tokenized_output=False, 
                 sentence_output=False,
                 remove_emojis = False
                 ):
    
    clean_text = raw_text

    if lowercase:
        if any(ord(char) > 127 for char in clean_text):
            clean_text = ''.join([unicodedata.normalize('NFKD', char).lower() for char in clean_text])
        else:
            clean_text = clean_text.lower()

    clean_text = re.sub(r'(\**\\[nrt]|</ul>)',' ',clean_text) 
    
    if not leave_punctuation:    
        clean_text = re.compile(r'[^\w\s]').sub(' ', clean_text)
    
    clean_text = re.sub(r'(\bhttp[^\s]+\b)',' ',clean_text)
    clean_text = re.sub(r'\b([^aeiou-])\b',' ',clean_text)

    if correct_spelling:
        incorrect_text = TextBlob(clean_text)
        clean_text = incorrect_text.correct()
        
    clean_text = word_tokenize(str(clean_text))
    
    if remove_stopwords:
        clean_text = stopword_remover(clean_text, stop_words)
        
    if lemmatization:
        for pos_tag in ["v","n","a"]:
            clean_text = [lemmatizer.lemmatize(token, pos=pos_tag) for token in clean_text]
    elif porter_stemming:  
        porter_stemmer = PorterStemmer()
        clean_text = [porter_stemmer.stem(token) for token in clean_text]
    
    if not tokenized_output:
        clean_text = " ".join(clean_text)
        clean_text = re.sub(r'(\s)(?!\w)','',clean_text)

    if sentence_output:
        clean_text = sent_tokenizer.tokenize(str(clean_text))
    
    # Adding emoji check
    if remove_emojis:  # Check if removing emojis is requested
        clean_text = [token for token in clean_text if not token in emoji.UNICODE_EMOJI]
    
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