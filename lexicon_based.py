from nltk.sentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()
from textblob import TextBlob

# VADER 
def vader_algorithm(review, compound = True):

    # Get the polarity scores (negative, neutral and positive) for the review
    polarity_scores_ = vader.polarity_scores(review)
    
    # compound scores
    if compound:
        polarity = polarity_scores_["compound"]

    else:
        # The three separated scores
        polarity = polarity_scores_

    return polarity

# TextBlob
def textblob_sa(song):

    # Get the polarity (compounded polarity scores) for the song
    polarity = TextBlob(song).sentiment.polarity

    return polarity
