import argparse
from youtube import Youtube
from utility import load_ml_pipeline
from utility import total_sentiment

def process_video_comments(apiKey, videoId, maxpages):
    # Load video comments
    yt = Youtube('https://www.googleapis.com/youtube/v3/commentThreads', apiKey, maxpages)
    comments = yt.get_comments(videoId)
    # Classify sentiment
    model = load_ml_pipeline("./models/lr_sentiment_basic.pkl")
    predictions = model.predict(comments)
    ts = total_sentiment(predictions)
    print("Total sentiment scores (Pos, Neg): {0}".format(ts))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("apiKey", help="Enter the Youtube API key to use for requests")
    parser.add_argument("videoId", help="Enter the Youtube video ID")
    parser.add_argument("maxpages", help="Enter the max pages returned of comments", type=int)
    args = parser.parse_args()
    process_video_comments(args.apiKey, args.videoId, args.maxpages)