import argparse
from youtube_sentiment import Youtube
from youtube_sentiment import load_ml_pipeline
from youtube_sentiment import total_counts
from youtube_sentiment import top_freq_words

def process_comments_summary(apiKey, videoId, maxpages, model):
    """
    Display video sentiment and analytics
    Args:
        apiKey: Youtube API key
        videoId: video ID
        maxpages: max pages of comments to scan
        model: the ML model to use for sentiment, model choices under ./models
    """
    # Load video comments
    yt = Youtube('https://www.googleapis.com/youtube/v3/commentThreads', apiKey, maxpages)
    comments = yt.get_comments(videoId)
    comments_corpus = ' '.join(comments)
    top_words = top_freq_words(comments_corpus, topwords=20)
    # Classify sentiment
    model = load_ml_pipeline(model)
    predictions = model.predict(comments)
    pos, neg = total_counts(predictions)
    print("""
        Video Summary:
        --------------------------------------
        Total sentiment scores (Pos, Neg): {0}, {1}
        Percentage Negative Sentiment: {2}
        Top words by frequency: {3}
        """
        .format(pos, neg, (neg / (pos + neg)), top_words))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("apiKey", help="Enter the Youtube API key to use for requests")
    parser.add_argument("videoId", help="Enter the Youtube video ID")
    parser.add_argument("maxpages", help="Enter the max pages returned of comments", type=int)
    parser.add_argument("model", help="Enter the model name to use for sentiment")
    args = parser.parse_args()
    process_comments_summary(args.apiKey, args.videoId, args.maxpages, args.model)

if __name__ == '__main__':
    main()