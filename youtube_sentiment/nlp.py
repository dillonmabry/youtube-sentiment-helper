from youtube_sentiment import load_ml_pipeline
from youtube_sentiment import total_counts
from youtube_sentiment import top_freq_words
from youtube_sentiment import extract_entities

class NLP(object):
    """
    Main class to use NLP structures and data analysis
    Args:
        self
        model: ML model to use under /models
    """
    def __init__(self, model):
        self.model = load_ml_pipeline(model)

    def process_comments(self, comments):
        """
        Return list of tuples of video comments with tagged sentiment
        Ex. [("Great video", 1), ("Terrible video", 0)]
        Args:
            comments: list of comments
        """
        predictions = self.model.predict(comments)
        return list(zip(comments, predictions))

    def process_comments_summary(self, comments):
        """
        Display video sentiment and analytics
        Args:
            comments: list of comments
        """
        # Corpus Summary
        comments_corpus = ' '.join(comments)
        entities = extract_entities(comments_corpus)
        top_words = top_freq_words(comments_corpus, topwords=20)
        # Classify sentiment
        predictions = self.model.predict(comments)
        pos, neg = total_counts(predictions)
        print("""
            Video Summary:
            --------------------------------------
            Total sentiment scores (Pos, Neg): {0}, {1}
            Percentage Negative Sentiment: {2}
            Top words by frequency: {3}
            Key entities: {4}
            """
            .format(pos, neg, (neg / (pos + neg)), top_words, entities))