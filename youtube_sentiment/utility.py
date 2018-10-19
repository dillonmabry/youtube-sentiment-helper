"""
Module of utility functions to handle data manipulation
"""
import itertools
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.externals import joblib
import pkg_resources

def flatten_list(items):
    """
    Method to flatten list of lists using itertools (faster than traditional lists)
    Args:
        items: the list of lists [ [item1],[item2] ]
    """
    if len(items) > 0 and items is not None:
        return list(itertools.chain.from_iterable(items))

def top_freq_words(comments):
    """
    Method to return frequency distribution of words from corpus text
    Args:
        comments: the corpus of comments as a single string
    """
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(comments)
    swords = stopwords.words('english')
    freq_words = FreqDist(w.lower() for w in words if w not in swords)   
    return freq_words

def load_ml_pipeline(filename):
    """
    Method to load ML pipeline model via sklearn joblib (pickle)
    Args:
        file_path: the file path of the .pkl model
    """
    model_path = pkg_resources.resource_filename('youtube_sentiment', 'models/')
    with (open(model_path + filename, "rb")) as f:
        try:
            return joblib.load(f)
        except Exception as e:
            raise e

def total_sentiment(sentiment_scores):
    """
    Method to calculate total sentiment for a video based on threshold of comments
    Returns tuple of number of positive vs. negative sentiment comments
    Example: (pos: 100, neg: 50)
    Args:
        sentiment_scores: list of sentiments for each comment (0 neg, 1 pos)
    """
    counts = np.bincount(sentiment_scores)
    return (counts[1], counts[0]) # pos, neg
    