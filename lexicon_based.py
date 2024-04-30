from nltk.sentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

# VADER 
def vader(review, compound = True):

    # Get the polarity scores (negative, neutral and positive) for the song
    polarity_scores_ = vader.polarity_scores(review)
    
    # If you want to compound the scores into one final score
    if compound:
        polarity = polarity_scores_["compound"]

    # If you want the three scores
    else:
        # The three separated scores
        polarity = polarity_scores_

    return polarity