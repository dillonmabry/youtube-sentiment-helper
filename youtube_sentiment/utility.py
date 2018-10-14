"""
Module of utility functions to handle data manipulation
"""
import itertools
import numpy as np
from sklearn.externals import joblib

def get_key_values(json, key):
    """
    Method to parse json of keys extracted, returns list of values
    Args:
        json: the full json object
        key: the key to search by, returns values for this key
    """
    return list(findkeys(json, key))

def findkeys(node, kv):
    """
    Method to get json from nested json starting from root
    Args:
        node: the json root to start at
        kv: the key to search
    """
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
                yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x

def flatten_list(items):
    """
    Method to flatten list of lists using itertools (faster than traditional lists)
    Args:
        items: the list of lists [ [item1],[item2] ]
    """
    if len(items) > 0 and items is not None:
        return list(itertools.chain.from_iterable(items))

def load_ml_pipeline(file_path):
    """
    Method to load ML pipeline model via sklearn joblib (pickle)
    Args:
        file_path: the file path of the .pkl model
    """
    with (open(file_path, "rb")) as f:
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
    